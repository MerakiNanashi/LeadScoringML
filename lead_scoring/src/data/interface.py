from pathlib import Path
from config import load_config

ROOT = Path(__file__).resolve().parent

config = load_config(
    ROOT / "config.yaml"
)

print(config)