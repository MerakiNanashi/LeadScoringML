from pathlib import Path
import os
import yaml

from schema import DatasetConfig


def load_config(path):

    path = Path(path).resolve()

    with open(path, "r") as f:
        raw_config = yaml.safe_load(f)

    config = DatasetConfig(**raw_config)

    # =====================================================
    # DERIVED PATHS
    # =====================================================

    config.base_dir =  Path(__file__).resolve().parent.parent.parent
    config.src_dir = config.base_dir / "src"
    config.results_dir = config.src_dir / "results"
    config.data_dir = config.src_dir / "data"
    config.save_dir = config.data_dir / "datasets"

    dirs_to_create = [

        # Core
        config.src_dir,
        config.results_dir,
        config.data_dir,
        config.save_dir,
    ]

    for directory in dirs_to_create:
        os.makedirs(directory, exist_ok=True)

    return config