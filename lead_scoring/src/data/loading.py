"""
loading.py
====================================

Dataset loading utilities for:
- custom CRM datasets
- Hillstrom uplift dataset
- generic tabular ingestion

This module ONLY handles:
    - reading files
    - schema normalization
    - validation

It does NOT:
    - generate synthetic data
    - preprocess features
    - split datasets
    - train models
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from schema import PreparedDataset


# =========================================================
# FILE LOADING
# =========================================================


def read_table(path: str | Path) -> pd.DataFrame:
    """
    Read supported tabular formats.

    Supported:
    - csv
    - parquet
    - json
    - jsonl
    - ndjson
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    if suffix in {".json", ".jsonl", ".ndjson"}:
        return pd.read_json(
            path,
            lines=suffix in {".jsonl", ".ndjson"},
        )

    raise ValueError(
        f"Unsupported dataset format: {suffix}"
    )


# =========================================================
# VALIDATION
# =========================================================


def require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
) -> None:
    """
    Validate required columns exist.
    """

    missing = [
        col
        for col in columns
        if col not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )


# =========================================================
# FEATURE DETECTION
# =========================================================


def infer_feature_columns(
    frame: pd.DataFrame,
    blocked_columns: set[str],
) -> list[str]:
    """
    Automatically infer usable feature columns.

    Excludes:
    - target columns
    - treatment columns
    - IDs
    - timestamps
    - constant columns
    """

    feature_cols = []

    for col in frame.columns:

        if col in blocked_columns:
            continue

        if pd.api.types.is_datetime64_any_dtype(frame[col]):
            continue

        if frame[col].nunique(dropna=True) <= 1:
            continue

        feature_cols.append(col)

    if not feature_cols:
        raise ValueError(
            "No usable feature columns found."
        )

    return feature_cols


# =========================================================
# LABEL NORMALIZATION
# =========================================================


def to_binary(series: pd.Series) -> pd.Series:
    """
    Convert common binary labels into {0,1}.
    """

    if (
        pd.api.types.is_bool_dtype(series)
        or pd.api.types.is_numeric_dtype(series)
    ):
        return (
            series
            .fillna(0)
            .astype(int)
            .clip(0, 1)
        )

    normalized = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )

    positives = {
        "1",
        "true",
        "yes",
        "y",
        "converted",
        "treated",
        "email",
    }

    return normalized.isin(positives).astype(int)


# =========================================================
# GENERIC CUSTOM DATASET
# =========================================================


def load_custom_dataset(
    config,
) -> PreparedDataset:
    """
    Load arbitrary CRM / lead datasets
    into the common project schema.
    """

    if config.path is None:
        raise ValueError(
            "config.path is required for custom datasets."
        )

    frame = read_table(config.path).copy()

    required = [
        config.outcome_col,
    ]

    if config.id_col:
        required.append(config.id_col)

    if config.treatment_col:
        required.append(config.treatment_col)

    require_columns(frame, required)

    # =====================================================
    # STANDARDIZE TARGET
    # =====================================================

    frame["outcome"] = to_binary(
        frame[config.outcome_col]
    )

    # =====================================================
    # STANDARDIZE TREATMENT
    # =====================================================

    normalized_treatment_col = None

    if config.treatment_col:

        frame["treatment"] = to_binary(
            frame[config.treatment_col]
        )

        normalized_treatment_col = "treatment"

    # =====================================================
    # STANDARDIZE IDS
    # =====================================================

    normalized_id_col = None

    if config.id_col:

        frame["lead_id"] = (
            frame[config.id_col]
            .astype(str)
        )

        normalized_id_col = "lead_id"

    # =====================================================
    # STANDARDIZE TIMESTAMPS
    # =====================================================

    normalized_time_col = None

    if config.time_col and config.time_col in frame.columns:

        frame["created_at"] = pd.to_datetime(
            frame[config.time_col],
            errors="coerce",
            utc=True,
        )

        normalized_time_col = "created_at"

    # =====================================================
    # FEATURE DETECTION
    # =====================================================

    blocked_columns = {
        config.outcome_col,
        "outcome",

        config.treatment_col,
        "treatment",

        config.id_col,
        "lead_id",

        config.time_col,
        "created_at",
    }

    feature_cols = infer_feature_columns(
        frame=frame,
        blocked_columns=blocked_columns,
    )

    metadata = {
        "dataset_type": "custom",
        "synthetic": False,
        "oracle_available": False,
        "observational": config.observational,
        "n_rows": len(frame),
    }

    return PreparedDataset(
        frame=frame,
        feature_cols=feature_cols,

        outcome_col="outcome",
        treatment_col=normalized_treatment_col,

        id_col=normalized_id_col,
        time_col=normalized_time_col,

        metadata=metadata,
    )


# =========================================================
# HILLSTROM DATASET
# =========================================================


def load_hillstrom(
    config,
    treatment_type: str = "any_email",
    outcome_col: str = "conversion",
) -> PreparedDataset:
    """
    Load Hillstrom MineThatData dataset.

    treatment_type:
        - any_email
        - mens_email
        - womens_email
    """

    if config.path is None:
        raise ValueError(
            "Hillstrom dataset path is required."
        )

    frame = read_table(config.path).copy()

    require_columns(
        frame,
        ["segment", outcome_col],
    )

    frame["normalized_segment"] = frame["segment"].str.strip().str.lower()

    # =====================================================
    # TREATMENT DEFINITIONS
    # =====================================================

    if treatment_type == "any_email":

        kept = frame.copy()

        treatment_flag = (
            kept["normalized_segment"]
            .ne("no e-mail")
        )

    elif treatment_type == "mens_email":

        kept = frame[
            frame["normalized_segment"].isin(
                ["mens e-mail", "no e-mail"]
            )
        ].copy()

        treatment_flag = (
            kept["normalized_segment"]
            .eq("mens e-mail")
        )

    elif treatment_type == "womens_email":

        kept = frame[
            frame["normalized_segment"].isin(
                ["womens e-mail", "no e-mail"]
            )
        ].copy()

        treatment_flag = (
            kept["normalized_segment"]
            .eq("womens e-mail")
        )

    else:
        raise ValueError(
            "Invalid treatment_type."
        )

    # =====================================================
    # STANDARD SCHEMA
    # =====================================================

    kept = kept.reset_index(drop=True)

    kept["lead_id"] = (
        kept.index.astype(str)
    )

    kept["treatment"] = (
        treatment_flag
        .astype(int)
    )

    kept["outcome"] = to_binary(
        kept[outcome_col]
    )

    kept["propensity"] = (
        kept["treatment"]
        .mean()
    )

    # =====================================================
    # FEATURE DETECTION
    # =====================================================

    blocked_columns = {
        "lead_id",
        "segment",
        "treatment",
        "outcome",
        "propensity",

        "visit",
        "spend",

        outcome_col,
    }

    feature_cols = infer_feature_columns(
        kept,
        blocked_columns,
    )

    metadata = {
        "dataset_type": "hillstrom",
        "synthetic": False,
        "oracle_available": False,
        "observational": False,
        "randomized": True,
        "treatment_type": treatment_type,
        "n_rows": len(kept),
    }

    return PreparedDataset(
        frame=kept,
        feature_cols=feature_cols,

        outcome_col="outcome",
        treatment_col="treatment",

        id_col="lead_id",
        time_col=None,

        propensity_col="propensity",

        metadata=metadata,
    )


# =========================================================
# ROUTER
# =========================================================


def load_dataset(
    config,
) -> PreparedDataset:
    """
    Route real dataset loading.
    """

    dataset_type = config.dataset_type.lower()

    if dataset_type == "custom":
        return load_custom_dataset(config)

    if dataset_type == "hillstrom":
        return load_hillstrom(config)

    raise ValueError(
        f"Unsupported real dataset type: {dataset_type}"
    )