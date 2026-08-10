#!/usr/bin/env python3
"""Aggregate independent evaluations and plot the paper's eight metrics."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/yaw-matplotlib")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs" / "paper_experiments.json").read_text(encoding="utf-8"))
METHOD_ORDER = ("outer", "yaw", "uniform", "static", "lirpg", "relara")
METRICS = (
    ("Evaluation/fixed_outer_reward", "Task reward", False),
    ("mean_episode_length", "Episode length", False),
    ("Evaluation/command_success", "Command success", False),
    ("Diagnostics/yaw_cmd_signed_success_rate", "Yaw success", False),
    ("Diagnostics/lin_cmd_success_rate", "Linear success", False),
    ("Evaluation/termination_rate", "Termination", False),
    ("Evaluation/action_saturation_rate", "Action saturation", True),
    ("Diagnostics/planar_position_error", "Planar error", False),
)
COLORS = {
    "outer": "#D1495B",
    "yaw": "#2563EB",
    "uniform": "#2A9D8F",
    "static": "#F4A261",
    "lirpg": "#7B2CBF",
    "relara": "#E76F51",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHOD_ORDER, action="append")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "results" / "evaluation_json",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "figures")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    methods = tuple(args.method) if args.method else METHOD_ORDER
    seeds = CONFIG["evaluation"]["seeds"]
    rows: list[dict[str, object]] = []
    for method in methods:
        label = CONFIG[method]["label"]
        for seed in seeds:
            path = args.input_dir / f"{method}_seed{seed}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            row: dict[str, object] = {
                "method": label,
                "seed": seed,
                "mean_episode_length": payload["mean_episode_length"],
            }
            row.update(payload["metrics"])
            rows.append(row)

    aggregate: list[dict[str, object]] = []
    for method in methods:
        label = CONFIG[method]["label"]
        selected = [row for row in rows if row["method"] == label]
        result: dict[str, object] = {"method": label, "num_seeds": len(selected)}
        for key, _, _ in METRICS:
            values = np.asarray([float(row[key]) for row in selected])
            result[f"{key}_mean"] = float(values.mean())
            result[f"{key}_sd"] = float(values.std(ddof=1))
        aggregate.append(result)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "evaluation_per_seed.csv", rows)
    write_csv(args.output_dir / "evaluation_aggregate.csv", aggregate)

    figure, axes = plt.subplots(2, 4, figsize=(18, 8.5))
    x = np.arange(len(methods))
    for axis, (key, title, log_scale) in zip(axes.flat, METRICS, strict=True):
        groups = [np.asarray([float(row[key]) for row in rows if row["method"] == CONFIG[m]["label"]]) for m in methods]
        axis.bar(
            x,
            [values.mean() for values in groups],
            yerr=[values.std(ddof=1) for values in groups],
            capsize=3,
            color=[COLORS[method] for method in methods],
        )
        axis.set_xticks(
            x,
            [CONFIG[method]["label"] for method in methods],
            rotation=25,
            ha="right",
            fontsize=8,
        )
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
        if log_scale:
            axis.set_yscale("log")
    figure.tight_layout()
    figure.savefig(args.output_dir / "independent_evaluation.png", dpi=220)
    figure.savefig(args.output_dir / "independent_evaluation.svg")
    print(f"Wrote aggregate table and figure to {args.output_dir}")


if __name__ == "__main__":
    main()
