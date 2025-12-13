import os
import yaml
from pathlib import Path

ENV = os.getenv("ENV", "prod")
PROJECT_DIR = Path(os.getenv("PROJECT_DIR", "/app"))
CONFIG_PATH = PROJECT_DIR / "configs" / f"{ENV}.yaml"

with open(CONFIG_PATH) as f:
    _cfg = yaml.safe_load(f)

PATHS = _cfg["paths"]
API = _cfg["api"]
