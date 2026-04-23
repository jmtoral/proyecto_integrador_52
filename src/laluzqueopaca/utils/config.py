from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    """Load a YAML file into a dictionary."""
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)

