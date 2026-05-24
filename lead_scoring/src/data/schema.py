"""
schemas.py
====================================

Shared data containers and config schemas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


# =========================================================
# DATASET CONFIG
# =========================================================


@dataclass
class DatasetConfig:

    # =====================================================
    # DATASET SELECTION
    # =====================================================

    dataset_type: str = "synthetic-01"

    # =====================================================
    # RANDOMNESS
    # =====================================================

    random_state: int = 42

    # =====================================================
    # GENERATION SETTINGS
    # =====================================================

    n_samples: int = 100_000
    treatment_rate: float = 0.50
    noise_std: float = 0.00
    observational: bool = False
    include_revenue: bool = True
    clip_tau: bool = True
    base_conversion_rate: float = 0.10

    # =====================================================
    # EXPERIMENT CONTROL
    # =====================================================

    experiment: int = 1

    # =====================================================
    # HILLSTROM OPTIONS
    # =====================================================

    hillstrom_treatment_type: str = "any_email"

    # =====================================================
    # CUSTOM DATASET OPTIONS
    # =====================================================

    path: str | Path | None = None
    outcome_col: str = "converted"
    treatment_col: str = "treatment"
    id_col: str = "lead_id"
    time_col: str | None = "created_at"

    # =====================================================
    # SPLITTING
    # =====================================================

    test_size: float = 0.25
    stratify: bool = True
    time_split: bool = False

    # =====================================================
    # PREPROCESSING
    # =====================================================

    scale_numeric: bool = True
    one_hot_encode: bool = True
    impute_numeric_strategy: str = "median"
    impute_categorical_strategy: str = "most_frequent"

    # =====================================================
    # OUTPUTS
    # =====================================================

    save_dataset: bool = False
    base_dir: str | Path = "."
    src_dir: Path | None = None
    results_dir: Path | None = None
    data_dir: Path | None = None
    save_dir: Path | None = None


# =========================================================
# PREPARED DATASET
# =========================================================


@dataclass
class PreparedDataset:
    """
    Standard dataset container used across:
    - conversion modeling
    - uplift modeling
    - policy learning
    - benchmarking
    """

    frame: pd.DataFrame
    feature_cols: list[str]
    outcome_col: str = "outcome"
    treatment_col: str | None = "treatment"
    id_col: str | None = "lead_id"
    time_col: str | None = "created_at"
    oracle_tau_col: str | None = None
    propensity_col: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # =====================================================
    # COMMON ACCESSORS
    # =====================================================

    @property
    def X(self) -> pd.DataFrame:
        return self.frame[self.feature_cols].copy()

    @property
    def y(self) -> pd.Series:
        return (
            self.frame[self.outcome_col]
            .astype(int)
            .copy()
        )

    @property
    def treatment(self) -> pd.Series | None:

        if self.treatment_col is None:
            return None

        return (
            self.frame[self.treatment_col]
            .astype(int)
            .copy()
        )

    @property
    def tau(self) -> pd.Series | None:

        if self.oracle_tau_col is None:
            return None

        return (
            self.frame[self.oracle_tau_col]
            .copy()
        )

    @property
    def propensity(self) -> pd.Series | None:

        if self.propensity_col is None:
            return None

        return (
            self.frame[self.propensity_col]
            .copy()
        )

    # =====================================================
    # SUMMARY
    # =====================================================

    @property
    def n_rows(self) -> int:
        return len(self.frame)

    @property
    def n_features(self) -> int:
        return len(self.feature_cols)

    @property
    def treated_rate(self) -> float | None:

        if self.treatment_col is None:
            return None

        return float(
            self.frame[self.treatment_col].mean()
        )

    @property
    def conversion_rate(self) -> float:
        return float(
            self.frame[self.outcome_col].mean()
        )


# =========================================================
# PREPROCESSOR CONTAINER
# =========================================================


@dataclass
class PreprocessorArtifacts:
    """
    Stores fitted preprocessing objects.
    """

    preprocessor: Any
    numeric_cols: list[str]
    categorical_cols: list[str]
    feature_names: list[str]

@dataclass
class DatasetSchema:
    """
    Overall dataset schema, including config and prepared data.
    """

    config: DatasetConfig
    dataset: PreparedDataset
    preprocessor: PreprocessorArtifacts | None = None