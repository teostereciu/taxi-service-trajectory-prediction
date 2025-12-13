import os
import yaml
from pathlib import Path

ENV = os.getenv("ENV", "dev")
PROJECT_DIR = Path(os.getenv("PROJECT_DIR", "/app"))
CONFIG_PATH = PROJECT_DIR / "configs" / f"{ENV}.yaml"

with open(CONFIG_PATH) as f:
    _cfg = yaml.safe_load(f)

PATHS = {k: Path(v) for k, v in _cfg["paths"].items()}
PREPROCESSING = _cfg.get("preprocessing", {})
FEATURE_ENG = _cfg.get("feature_engineering", {})
TRAIN = _cfg.get("train", {})
