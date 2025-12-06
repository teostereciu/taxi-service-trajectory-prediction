import yaml
from pathlib import Path


def load_config(env="dev"):
    config_path = Path(f"/app/configs/{env}.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)
