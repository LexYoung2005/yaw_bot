#!/usr/bin/env python3
"""Recreate the six-panel training figure from anonymized submitted curves."""

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
    ("task_reward", "Fixed task reward"),
    ("episode_length", "Episode length"),
    ("command_success", "Command success"),
    ("linear_success", "Linear success"),
    ("yaw_success", "Yaw success"),
    ("termination_rate", "Termination rate"),
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
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "results" / "submitted_training_curves.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "figures",
    )
    parser.add_argument("--method", choices=METHOD_ORDER, action="append")
    parser.add_argument("--smoothing", type=int, default=25)
    return parser.parse_args()


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window, dtype=np.float64) / window
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def main() -> None:
    args = parse_args()
    if args.smoothing <= 0:
        raise ValueError("--smoothing must be positive.")
    methods = tuple(args.method) if args.method else METHOD_ORDER
    data = np.load(args.data)
    steps = np.arange(int(CONFIG["training"]["iterations"]))

    figure, axes = plt.subplots(2, 3, figsize=(17, 9), sharex=True)
    summary_rows: list[dict[str, object]] = []
    for axis, (metric, title) in zip(axes.flat, METRICS, strict=True):
        for method in methods:
            seeds = CONFIG[method]["seeds"]
            raw = np.stack([data[f"training__{method}__{seed}__{metric}"] for seed in seeds])
            curves = np.stack([smooth(values, args.smoothing) for values in raw])
            mean = curves.mean(axis=0)
            standard_deviation = curves.std(axis=0, ddof=1)
            axis.plot(
                steps,
                mean,
                linewidth=1.6,
                color=COLORS[method],
                label=CONFIG[method]["label"],
            )
            axis.fill_between(
                steps,
                mean - standard_deviation,
                mean + standard_deviation,
                color=COLORS[method],
                alpha=0.08,
            )
            summary_rows.append(
                {
                    "method": CONFIG[method]["label"],
                    "metric": metric,
                    "final100_mean": float(raw[:, -100:].mean()),
                    "final100_seed_sd": float(raw[:, -100:].mean(axis=1).std(ddof=1)),
                }
            )
        axis.set_title(title)
        axis.grid(alpha=0.2)
        if "success" in metric or "rate" in metric:
            axis.set_ylim(bottom=0)
    axes[0, 0].legend(fontsize=8, ncol=2)
    for axis in axes[1]:
        axis.set_xlabel("PPO iteration")
    figure.tight_layout()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_dir / "training_curves.png", dpi=220)
    figure.savefig(args.output_dir / "training_curves.svg")
    plt.close(figure)
    with (args.output_dir / "training_final100_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote training-curve analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
