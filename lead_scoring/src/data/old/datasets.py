"""Dataset loading and preparation utilities.

The project uses one common schema across baseline lead scoring and uplift
experiments:

- one row per lead/customer
- `outcome` is the observed conversion label Y
- `treatment` is whether the lead received an intervention A
- feature columns are measured before the intervention/outcome window

These helpers intentionally keep modeling out of the data layer. They return
clean pandas objects that can feed conversion rankers, S/T/X learners, policy
evaluation, and simulation benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


PathLike = str | Path


@dataclass(frozen=True)
class PreparedDataset:
    """Container for a prepared lead allocation dataset."""

    frame: pd.DataFrame
    feature_cols: list[str]
    outcome_col: str = "outcome"
    treatment_col: str | None = "treatment"
    id_col: str | None = "lead_id"
    time_col: str | None = "created_at"
    oracle_uplift_col: str | None = None
    propensity_col: str | None = None

    @property
    def X(self) -> pd.DataFrame:
        return self.frame[self.feature_cols].copy()

    @property
    def y(self) -> pd.Series:
        return self.frame[self.outcome_col].astype(int).copy()

    @property
    def treatment(self) -> pd.Series | None:
        if self.treatment_col is None:
            return None
        return self.frame[self.treatment_col].astype(int).copy()

    @property
    def oracle_uplift(self) -> pd.Series | None:
        if self.oracle_uplift_col is None:
            return None
        return self.frame[self.oracle_uplift_col].copy()


@dataclass(frozen=True)
class FeaturePreprocessor:
    """Small pandas-based encoder for model-ready matrices."""

    numeric_cols: list[str]
    categorical_cols: list[str]
    means: pd.Series
    stds: pd.Series
    dummy_cols: list[str]

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        numeric = frame[self.numeric_cols].copy()
        if self.numeric_cols:
            numeric = numeric.apply(pd.to_numeric, errors="coerce")
            numeric = numeric.fillna(self.means)
            numeric = (numeric - self.means) / self.stds

        categorical = pd.get_dummies(
            frame[self.categorical_cols].fillna("__missing__").astype(str),
            columns=self.categorical_cols,
            dummy_na=False,
        )
        categorical = categorical.reindex(columns=self.dummy_cols, fill_value=0)

        return pd.concat([numeric, categorical], axis=1).astype(float)


def load_generic_leads(
    path: PathLike,
    *,
    outcome_col: str = "converted",
    id_col: str = "lead_id",
    time_col: str | None = "created_at",
    treatment_col: str | None = None,
    exclude_cols: Iterable[str] = (),
) -> PreparedDataset:
    """Load a CRM/exported lead dataset and normalize it to the project schema.

    Expected minimum columns:
    - `lead_id`
    - `converted`
    - pre-outcome feature columns

    Optional columns:
    - `created_at` for time-based train/test splits
    - treatment/intervention flag for uplift experiments
    """

    frame = _read_table(path)
    required = [outcome_col]
    if id_col:
        required.append(id_col)
    if time_col:
        required.append(time_col)
    if treatment_col:
        required.append(treatment_col)
    _require_columns(frame, required)

    frame = frame.copy()
    frame["outcome"] = _to_binary(frame[outcome_col])

    normalized_treatment_col = None
    if treatment_col is not None:
        frame["treatment"] = _to_binary(frame[treatment_col])
        normalized_treatment_col = "treatment"

    normalized_id_col = None
    if id_col != "lead_id":
        frame["lead_id"] = frame[id_col].astype(str)
        normalized_id_col = "lead_id"
    elif id_col in frame:
        normalized_id_col = id_col

    normalized_time_col = None
    if time_col is not None:
        frame["created_at"] = pd.to_datetime(frame[time_col], errors="coerce", utc=True)
        normalized_time_col = "created_at"

    blocked = {
        outcome_col,
        "outcome",
        id_col,
        "lead_id",
        *(["created_at", time_col] if time_col else []),
        *(["treatment", treatment_col] if treatment_col else []),
        *exclude_cols,
    }
    feature_cols = _usable_feature_columns(frame, blocked)

    return PreparedDataset(
        frame=frame,
        feature_cols=feature_cols,
        outcome_col="outcome",
        treatment_col=normalized_treatment_col,
        id_col=normalized_id_col,
        time_col=normalized_time_col,
    )


def load_hillstrom(
    path: PathLike,
    *,
    treatment: str = "any_email",
    outcome_col: str = "conversion",
) -> PreparedDataset:
    """Load Kevin Hillstrom MineThatData email data for uplift modeling.

    Common source columns include:
    `recency`, `history_segment`, `history`, `mens`, `womens`, `zip_code`,
    `newbie`, `channel`, `segment`, `visit`, `conversion`, and `spend`.

    Treatment choices:
    - `any_email`: Mens E-Mail or Womens E-Mail vs No E-Mail
    - `mens_email`: Mens E-Mail vs No E-Mail, dropping Womens E-Mail
    - `womens_email`: Womens E-Mail vs No E-Mail, dropping Mens E-Mail
    """

    frame = _read_table(path).copy()
    _require_columns(frame, ["segment", outcome_col])

    if treatment == "any_email":
        kept = frame
        treatment_flag = kept["segment"].ne("No E-Mail")
    elif treatment == "mens_email":
        kept = frame[frame["segment"].isin(["Mens E-Mail", "No E-Mail"])].copy()
        treatment_flag = kept["segment"].eq("Mens E-Mail")
    elif treatment == "womens_email":
        kept = frame[frame["segment"].isin(["Womens E-Mail", "No E-Mail"])].copy()
        treatment_flag = kept["segment"].eq("Womens E-Mail")
    else:
        raise ValueError(
            "treatment must be one of: any_email, mens_email, womens_email"
        )

    kept = kept.reset_index(drop=True)
    kept["lead_id"] = np.arange(len(kept)).astype(str)
    kept["treatment"] = treatment_flag.astype(int).to_numpy()
    kept["outcome"] = _to_binary(kept[outcome_col])
    kept["propensity"] = kept["treatment"].mean()

    blocked = {
        "lead_id",
        "segment",
        "treatment",
        "outcome",
        "propensity",
        outcome_col,
        "visit",
        "spend",
    }
    feature_cols = _usable_feature_columns(kept, blocked)

    return PreparedDataset(
        frame=kept,
        feature_cols=feature_cols,
        outcome_col="outcome",
        treatment_col="treatment",
        id_col="lead_id",
        time_col=None,
        propensity_col="propensity",
    )


def make_semisynthetic_leads(
    n: int = 50_000,
    *,
    seed: int = 42,
    treatment_rate: float = 0.35,
) -> PreparedDataset:
    """Create semi-synthetic leads with known potential outcomes and uplift.

    This is useful for article demos and regret benchmarks because real data
    reveals only one potential outcome per lead. Here we keep `y0`, `y1`, and
    `tau` so experiments can compare learned policies against an oracle.
    """

    rng = np.random.default_rng(seed)

    company_size = rng.lognormal(mean=3.0, sigma=1.0, size=n).clip(1, 2_000)
    prior_visits = rng.poisson(lam=2.5, size=n)
    pricing_page_views = rng.poisson(lam=0.35 + 0.08 * prior_visits, size=n)
    form_fills = rng.binomial(1, p=_sigmoid(-2.4 + 0.25 * pricing_page_views))
    is_enterprise = rng.binomial(1, p=_sigmoid(-3.0 + 0.004 * company_size))
    source = rng.choice(
        ["paid_search", "organic", "direct", "partner", "social"],
        size=n,
        p=[0.28, 0.30, 0.20, 0.12, 0.10],
    )
    region = rng.choice(["na", "emea", "apac", "latam"], size=n, p=[0.48, 0.25, 0.19, 0.08])

    source_effect = pd.Series(source).map(
        {
            "paid_search": 0.20,
            "organic": 0.05,
            "direct": 0.25,
            "partner": 0.35,
            "social": -0.15,
        }
    ).to_numpy()
    region_effect = pd.Series(region).map(
        {"na": 0.15, "emea": 0.05, "apac": -0.05, "latam": -0.10}
    ).to_numpy()

    baseline_logit = (
        -4.1
        + 0.018 * np.sqrt(company_size)
        + 0.16 * prior_visits
        + 0.40 * pricing_page_views
        + 0.95 * form_fills
        + 0.45 * is_enterprise
        + source_effect
        + region_effect
    )
    p0 = _sigmoid(baseline_logit)

    tau = (
        0.015
        + 0.045 * (pricing_page_views > 0)
        + 0.055 * form_fills
        + 0.030 * is_enterprise
        - 0.030 * (source == "direct")
        - 0.020 * (prior_visits >= 7)
    )
    tau = np.clip(tau + rng.normal(0.0, 0.012, size=n), -0.04, 0.22)
    p1 = np.clip(p0 + tau, 0.001, 0.95)

    treatment = rng.binomial(1, treatment_rate, size=n)
    y0 = rng.binomial(1, p0)
    y1 = rng.binomial(1, p1)
    outcome = np.where(treatment == 1, y1, y0)

    start = pd.Timestamp("2025-01-01", tz="UTC")
    created_at = start + pd.to_timedelta(rng.integers(0, 180, size=n), unit="D")

    frame = pd.DataFrame(
        {
            "lead_id": [f"L{i:07d}" for i in range(n)],
            "created_at": created_at,
            "company_size": company_size.round(0).astype(int),
            "prior_visits": prior_visits,
            "pricing_page_views": pricing_page_views,
            "form_fills": form_fills,
            "is_enterprise": is_enterprise,
            "source": source,
            "region": region,
            "treatment": treatment,
            "outcome": outcome,
            "p0": p0,
            "p1": p1,
            "tau": p1 - p0,
            "y0": y0,
            "y1": y1,
            "propensity": treatment_rate,
        }
    )
    feature_cols = [
        "company_size",
        "prior_visits",
        "pricing_page_views",
        "form_fills",
        "is_enterprise",
        "source",
        "region",
    ]

    return PreparedDataset(
        frame=frame,
        feature_cols=feature_cols,
        outcome_col="outcome",
        treatment_col="treatment",
        id_col="lead_id",
        time_col="created_at",
        oracle_uplift_col="tau",
        propensity_col="propensity",
    )


def build_feature_matrix(
    train: PreparedDataset,
    test: PreparedDataset | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, FeaturePreprocessor]:
    """Fit preprocessing on train features and transform train/test matrices.

    This intentionally uses pandas instead of sklearn so dataset preparation can
    run in minimal environments. Downstream models can consume `.to_numpy()`.
    """

    X_train = train.X
    numeric_cols = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in train.feature_cols if c not in numeric_cols]

    numeric = X_train[numeric_cols].apply(pd.to_numeric, errors="coerce")
    means = numeric.mean()
    stds = numeric.std(ddof=0).replace(0, 1).fillna(1)
    dummy_cols = pd.get_dummies(
        X_train[categorical_cols].fillna("__missing__").astype(str),
        columns=categorical_cols,
    ).columns.tolist()

    preprocessor = FeaturePreprocessor(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        means=means,
        stds=stds,
        dummy_cols=dummy_cols,
    )

    X_train_matrix = preprocessor.transform(X_train)
    X_test_matrix = None if test is None else preprocessor.transform(test.X)
    return X_train_matrix, X_test_matrix, preprocessor


def train_test_split_prepared(
    data: PreparedDataset,
    *,
    test_size: float = 0.25,
    split_time: str | pd.Timestamp | None = None,
    random_state: int = 42,
) -> tuple[PreparedDataset, PreparedDataset]:
    """Split a prepared dataset using time when possible, otherwise stratify."""

    frame = data.frame.copy()

    if split_time is not None:
        if data.time_col is None:
            raise ValueError("split_time was provided, but dataset has no time_col")
        cutoff = pd.Timestamp(split_time)
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize("UTC")
        train_frame = frame[frame[data.time_col] < cutoff].copy()
        test_frame = frame[frame[data.time_col] >= cutoff].copy()
    elif data.time_col is not None:
        frame = frame.sort_values(data.time_col)
        cutoff_idx = int(len(frame) * (1 - test_size))
        train_frame = frame.iloc[:cutoff_idx].copy()
        test_frame = frame.iloc[cutoff_idx:].copy()
    else:
        stratify_cols = [data.outcome_col]
        if data.treatment_col is not None:
            stratify_cols.append(data.treatment_col)
        train_frame, test_frame = _stratified_random_split(
            frame, stratify_cols, test_size=test_size, random_state=random_state
        )

    if train_frame.empty or test_frame.empty:
        raise ValueError("Split produced an empty train or test set")

    return (
        _replace_frame(data, train_frame.reset_index(drop=True)),
        _replace_frame(data, test_frame.reset_index(drop=True)),
    )


def _read_table(path: PathLike) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".json", ".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=suffix in {".jsonl", ".ndjson"})
    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def _require_columns(frame: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c and c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _usable_feature_columns(frame: pd.DataFrame, blocked: set[str]) -> list[str]:
    feature_cols = []
    for col in frame.columns:
        if col in blocked:
            continue
        if pd.api.types.is_datetime64_any_dtype(frame[col]):
            continue
        if frame[col].nunique(dropna=True) <= 1:
            continue
        feature_cols.append(col)
    if not feature_cols:
        raise ValueError("No usable feature columns found")
    return feature_cols


def _to_binary(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int).clip(0, 1)

    normalized = series.astype(str).str.strip().str.lower()
    positives = {"1", "true", "yes", "y", "converted", "treated", "email"}
    return normalized.isin(positives).astype(int)


def _sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-z))


def _replace_frame(data: PreparedDataset, frame: pd.DataFrame) -> PreparedDataset:
    return PreparedDataset(
        frame=frame,
        feature_cols=data.feature_cols,
        outcome_col=data.outcome_col,
        treatment_col=data.treatment_col,
        id_col=data.id_col,
        time_col=data.time_col,
        oracle_uplift_col=data.oracle_uplift_col,
        propensity_col=data.propensity_col,
    )


def _stratified_random_split(
    frame: pd.DataFrame,
    stratify_cols: Sequence[str],
    *,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_state)
    train_parts = []
    test_parts = []

    keys = frame[stratify_cols].astype(str).agg("_".join, axis=1)
    for _, group in frame.groupby(keys, sort=False):
        shuffled_idx = group.index.to_numpy()
        rng.shuffle(shuffled_idx)
        test_count = int(round(len(shuffled_idx) * test_size))
        if len(shuffled_idx) > 1:
            test_count = min(max(test_count, 1), len(shuffled_idx) - 1)

        test_idx = shuffled_idx[:test_count]
        train_idx = shuffled_idx[test_count:]
        if len(train_idx):
            train_parts.append(frame.loc[train_idx])
        if len(test_idx):
            test_parts.append(frame.loc[test_idx])

    return pd.concat(train_parts), pd.concat(test_parts)
