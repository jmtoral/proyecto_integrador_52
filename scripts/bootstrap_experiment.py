from pathlib import Path

from laluzqueopaca.training.pipeline import run_experiment


if __name__ == "__main__":
    output = run_experiment(Path("artifacts") / "baseline_detection")
    print(f"Experiment directory ready at: {output}")

