#!/usr/bin/env python3
"""Recreate the raw/effective Composer-weight figure and summary table."""

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
GROUPS = (
    ("stability", "Stability", "#2563EB"),
    ("contact_slip", "Contact / slip", "#2A9D8F"),
    ("linear", "Linear motion", "#F4A261"),
    ("yaw", "Yaw tracking", "#7B2CBF"),
    ("regularization", "Regularization", "#D1495B"),
)


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
    data = np.load(args.data)
    seeds = CONFIG["yaw"]["seeds"]
    iterations = int(CONFIG["training"]["iterations"])
    steps = np.arange(iterations)
    figure, axes = plt.subplots(2, 1, figsize=(11.5, 8.2), sharex=True)

    for axis, prefix, title in (
        (axes[0], "raw", "Raw Composer weights"),
        (axes[1], "effective", "PPO-effective Composer weights"),
    ):
        for key, label, color in GROUPS:
            values = np.stack([smooth(data[f"composer__{seed}__{prefix}_{key}"], args.smoothing) for seed in seeds])
            mean = values.mean(axis=0)
            standard_deviation = values.std(axis=0, ddof=1)
            axis.plot(steps, mean, color=color, linewidth=2.0, label=label)
            axis.fill_between(
                steps,
                mean - standard_deviation,
                mean + standard_deviation,
                color=color,
                alpha=0.12,
                linewidth=0,
            )
        axis.axhline(1.0, color="#4B5563", linestyle=":", linewidth=1.2)
        axis.axhline(0.6, color="#9CA3AF", linestyle="--", linewidth=0.9)
        axis.axhline(1.4, color="#9CA3AF", linestyle="--", linewidth=0.9)
        axis.set_ylabel("Group weight")
        axis.set_ylim(0.55, 1.45)
        axis.set_title(title)
        axis.grid(alpha=0.2)

    beta = np.stack([data[f"composer__{seed}__beta"] for seed in seeds])
    beta_axis = axes[1].twinx()
    beta_axis.plot(
        steps,
        beta.mean(axis=0),
        color="black",
        linestyle="--",
        linewidth=1.8,
        label="beta",
    )
    beta_axis.set_ylabel("Deployment beta")
    beta_axis.set_ylim(-0.04, 1.04)
    axes[1].set_xlabel("PPO iteration")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=5)
    figure.tight_layout(rect=(0, 0, 1, 0.95))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_dir / "composer_weight_evolution.png", dpi=260)
    figure.savefig(args.output_dir / "composer_weight_evolution.svg")
    figure.savefig(args.output_dir / "composer_weight_evolution.pdf")
    plt.close(figure)

    rows: list[dict[str, object]] = []
    for key, label, _ in GROUPS:
        raw = np.stack([data[f"composer__{seed}__raw_{key}"] for seed in seeds])
        effective = np.stack([data[f"composer__{seed}__effective_{key}"] for seed in seeds])
        rows.append(
            {
                "group": label,
                "raw_final100_mean": float(raw[:, -100:].mean()),
                "raw_final100_seed_sd": float(raw[:, -100:].mean(axis=1).std(ddof=1)),
                "effective_final100_mean": float(effective[:, -100:].mean()),
                "raw_observed_min": float(raw.min()),
                "raw_observed_max": float(raw.max()),
            }
        )
    with (args.output_dir / "composer_weight_evolution_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote Composer-weight analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
