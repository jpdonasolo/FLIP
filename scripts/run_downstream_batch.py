#!/usr/bin/env python3

"""
Utility script to run `python run_experiment.py example_downstream`
multiple times while updating the `output_dir` before each run.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import subprocess
from pathlib import Path

import toml


PROJECT_ROOT = Path("/users/eleves-a/2025/iuliia.korotkova/FLIP").resolve()
CONFIG_PATH = PROJECT_ROOT / "experiments" / "example_downstream" / "config.toml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run example_downstream multiple times with unique output dirs."
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of times to run the experiment (default: 10).",
    )
    parser.add_argument(
        "--experiment",
        default="example_downstream",
        help="Experiment name passed to run_experiment.py (default: example_downstream).",
    )
    parser.add_argument(
        "--base-output",
        dest="base_output",
        help=(
            "Base path for generated output dirs. "
            "Defaults to the current train_user.output_dir value with a '_batch' suffix."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found at {CONFIG_PATH}")

    original_text = CONFIG_PATH.read_text()
    original_config = toml.loads(original_text)

    base_output = (
        args.base_output
        if args.base_output is not None
        else original_config["train_user"]["output_dir"].rstrip("/") + "_batch"
    )
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        for idx in range(1, args.runs + 1):
            run_output = f"{base_output}_{timestamp}_run_{idx:02d}"
            Path(run_output).mkdir(parents=True, exist_ok=True)

            updated_config = copy.deepcopy(original_config)
            updated_config["train_user"]["output_dir"] = run_output
            CONFIG_PATH.write_text(toml.dumps(updated_config))

            print(f"[{idx}/{args.runs}] Running with output_dir={run_output}")
            subprocess.run(
                ["python", str(PROJECT_ROOT / "run_experiment.py"), args.experiment],
                check=True,
                cwd=PROJECT_ROOT,
            )
    finally:
        CONFIG_PATH.write_text(original_text)
        print("Restored original config.")


if __name__ == "__main__":
    main()

