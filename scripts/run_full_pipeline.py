#!/usr/bin/env python3

"""
Batch pipeline for running `example_attack` with multiple model/dataset
configurations and evaluating downstream performance across flip budgets.

Workflow per configuration:
1. Update `experiments/example_attack/config.toml` with model/dataset overrides
   and fresh output directories, then run `python run_experiment.py example_attack`.
2. For each flip budget (default: [1500]), update
   `experiments/example_downstream/config.toml`, run the downstream training
   10 times, and collect the final clean/poison accuracies.
3. Persist summary statistics (mean/std) for each (dataset, model, budget).

Use the `--grid` flag to supply a JSON/TOML file describing the set of
configurations to evaluate. See `DEFAULT_COMBOS` for the required structure.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import toml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ATTACK_CONFIG = PROJECT_ROOT / "experiments" / "example_attack" / "config.toml"
DEFAULT_DOWNSTREAM_CONFIG = PROJECT_ROOT / "experiments" / "example_downstream" / "config.toml"
DEFAULT_RESULTS_ROOT = Path("/Data/iuliia.korotkova/FLIP") / "out" / "pipeline_runs"
DEFAULT_SUMMARY_PATH = Path("/Data/iuliia.korotkova/FLIP") / "out" / "pipeline_runs" / "metrics_summary.csv"
DEFAULT_BUDGETS = [1500]


def ensure_trailing_slash(path: Path | str) -> str:
    path_str = str(path)
    return path_str if path_str.endswith("/") else path_str + "/"


def run_cmd(cmd: Sequence[str]) -> None:
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def load_array_last_acc(path: Path) -> float:
    arr = np.load(path, allow_pickle=True)
    last_entry = arr[-1]
    if isinstance(last_entry, (list, tuple)):
        return float(last_entry[0])
    last_arr = np.asarray(last_entry)
    if last_arr.ndim == 0:
        return float(last_arr.item())
    return float(last_arr[0])


def compute_stats(values: Sequence[float]) -> Dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    mean = float(data.mean())
    std = float(data.std(ddof=1)) if data.size > 1 else 0.0
    return {"mean": mean, "std": std}


def load_combo_grid(path: Path | None) -> List[Dict]:
    if path is None:
        return copy.deepcopy(DEFAULT_COMBOS)
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text())
    if path.suffix.lower() == ".toml":
        return list(toml.load(path).values())
    raise ValueError(f"Unsupported grid file extension: {path.suffix}")


@dataclass
class ComboContext:
    name: str
    metadata: Dict
    overrides: Dict[str, Dict]
    attack_dirs: Dict[str, Path]
    downstream_root: Path


DEFAULT_COMBOS: List[Dict] = [
    {
        "name": "cifar_r32p_1xs",
        "metadata": {"dataset": "cifar", "model": "r32p", "poisoner": "1xs"},
        "overrides": {
            "train_expert": {"model": "r32p", "dataset": "cifar", "poisoner": "1xs"},
            "generate_labels": {"expert_model": "r32p", "dataset": "cifar", "poisoner": "1xs"},
            "train_user": {"user_model": "r32p", "dataset": "cifar", "poisoner": "1xs"},
        },
    }
]


def prepare_combo_context(
    combo: Dict,
    results_root: Path,
) -> ComboContext:
    name = combo["name"]
    overrides = combo.get("overrides", {})
    metadata = combo.get("metadata", {})

    combo_root = results_root / name
    attack_root = combo_root / "attack"
    attack_dirs = {
        "root": attack_root,
        "checkpoints": attack_root / "checkpoints",
        "labels": attack_root / "labels",
        "selected": attack_root / "selected",
    }
    downstream_root = combo_root / "downstream"

    for path in attack_dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    downstream_root.mkdir(parents=True, exist_ok=True)

    return ComboContext(
        name=name,
        metadata=metadata,
        overrides=overrides,
        attack_dirs=attack_dirs,
        downstream_root=downstream_root,
    )


def build_attack_config(template: Dict, ctx: ComboContext) -> Dict:
    conf = copy.deepcopy(template)
    for section, values in ctx.overrides.items():
        if section in conf:
            conf[section].update(values)
    checkpoints_dir = ctx.attack_dirs["checkpoints"]
    labels_dir = ctx.attack_dirs["labels"]
    selected_dir = ctx.attack_dirs["selected"]

    conf["train_expert"]["output_dir"] = ensure_trailing_slash(checkpoints_dir) + "0/"
    conf["generate_labels"]["input_pths"] = str(checkpoints_dir / "{}" / "model_{}_{}.pth")
    conf["generate_labels"]["opt_pths"] = str(checkpoints_dir / "{}" / "model_{}_{}_opt.pth")
    conf["generate_labels"]["output_dir"] = ensure_trailing_slash(labels_dir)
    conf["select_flips"]["input_label_glob"] = str(labels_dir / "labels.npy")
    conf["select_flips"]["true_labels"] = str(labels_dir / "true.npy")
    conf["select_flips"]["output_dir"] = ensure_trailing_slash(selected_dir)
    return conf


def build_downstream_config(template: Dict, ctx: ComboContext, overrides: Dict) -> Dict:
    conf = copy.deepcopy(template)
    if "train_user" in ctx.overrides:
        conf["train_user"].update(ctx.overrides["train_user"])
    conf["train_user"].update(overrides)
    return conf


def run_single_experiment(experiment: str) -> None:
    run_cmd(["python", "run_experiment.py", experiment])


def orchestrate_combo(
    ctx: ComboContext,
    attack_template: Dict,
    downstream_template: Dict,
    args: argparse.Namespace,
    summary_rows: List[Dict],
) -> None:
    attack_config = build_attack_config(attack_template, ctx)
    attack_config_path = Path(args.attack_config)
    downstream_config_path = Path(args.downstream_config)

    attack_config_path.write_text(toml.dumps(attack_config))
    print(f"[{ctx.name}] Running example_attack...")
    run_single_experiment("example_attack")

    selected_dir = ctx.attack_dirs["selected"]
    for budget in args.budgets:
        label_path = selected_dir / f"{budget}.npy"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing labels for budget {budget}: {label_path}")

        poison_final_acc, clean_final_acc = [], []
        budget_root = ctx.downstream_root / f"budget_{budget}"
        budget_root.mkdir(parents=True, exist_ok=True)

        for run_idx in range(1, args.runs + 1):
            run_dir = budget_root / f"run_{run_idx:02d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            downstream_overrides = {
                "input_labels": str(label_path),
                "output_dir": ensure_trailing_slash(run_dir),
            }
            downstream_config = build_downstream_config(
                downstream_template, ctx, downstream_overrides
            )
            downstream_config_path.write_text(toml.dumps(downstream_config))

            print(f"[{ctx.name}] Budget {budget} – downstream run {run_idx}/{args.runs}")
            run_single_experiment("example_downstream")

            pacc = load_array_last_acc(run_dir / "paccs.npy")
            cacc = load_array_last_acc(run_dir / "caccs.npy")
            poison_final_acc.append(pacc)
            clean_final_acc.append(cacc)

        poison_stats = compute_stats(poison_final_acc)
        clean_stats = compute_stats(clean_final_acc)

        summary_rows.append(
            {
                "combo": ctx.name,
                "dataset": ctx.metadata.get("dataset"),
                "model": ctx.metadata.get("model"),
                "poisoner": ctx.metadata.get("poisoner"),
                "budget": budget,
                "runs": args.runs,
                "poison_acc_mean": poison_stats["mean"],
                "poison_acc_std": poison_stats["std"],
                "clean_acc_mean": clean_stats["mean"],
                "clean_acc_std": clean_stats["std"],
            }
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run attack/downstream grids and collect stats.")
    parser.add_argument("--attack-config", default=str(DEFAULT_ATTACK_CONFIG))
    parser.add_argument("--downstream-config", default=str(DEFAULT_DOWNSTREAM_CONFIG))
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--summary-path", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--grid", type=str, help="Path to JSON/TOML file defining combo overrides.")
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=int,
        default=DEFAULT_BUDGETS,
        help="Budgets (n flips) to evaluate.",
    )
    parser.add_argument("--runs", type=int, default=10, help="Downstream runs per budget.")
    return parser.parse_args()


def write_summary(rows: List[Dict], destination: Path) -> None:
    if not rows:
        print("No summary rows to write.")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with destination.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary metrics to {destination}")


def main() -> None:
    args = parse_args()
    attack_config_path = Path(args.attack_config)
    downstream_config_path = Path(args.downstream_config)
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    attack_template_text = attack_config_path.read_text()
    downstream_template_text = downstream_config_path.read_text()
    attack_template = toml.loads(attack_template_text)
    downstream_template = toml.loads(downstream_template_text)

    summary_rows: List[Dict] = []
    combo_grid = load_combo_grid(Path(args.grid) if args.grid else None)

    try:
        for combo in combo_grid:
            ctx = prepare_combo_context(combo, results_root)
            orchestrate_combo(ctx, attack_template, downstream_template, args, summary_rows)
    finally:
        attack_config_path.write_text(attack_template_text)
        downstream_config_path.write_text(downstream_template_text)

    write_summary(summary_rows, Path(args.summary_path))


if __name__ == "__main__":
    main()

