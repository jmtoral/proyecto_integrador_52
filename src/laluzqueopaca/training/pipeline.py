from pathlib import Path


def run_experiment(output_dir: str | Path) -> Path:
    """Prepare an experiment directory and return its path."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path

