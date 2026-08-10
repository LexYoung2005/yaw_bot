#!/usr/bin/env python3
"""Export a fresh paper rerun's TensorBoard scalars to the release NPZ schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs" / "paper_experiments.json").read_text(encoding="utf-8"))
METHODS = ("yaw", "outer", "uniform", "static", "lirpg", "relara")
METRICS = {
    "task_reward": "OuterComposer/outer_reward",
    "episode_length": "Train/mean_episode_length",
    "command_success": "OuterComposer/command_success",
    "linear_success": "Diagnostics/lin_cmd_success_rate",
    "yaw_success": "Diagnostics/yaw_cmd_signed_success_rate",
    "termination_rate": "Diagnostics/termination_rate",
}
GROUPS = ("stability", "contact_slip", "linear", "yaw", "regularization")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-root",
        type=Path,
        default=ROOT / "logs" / "rsl_rl" / "yaw_bot_predictive_gated",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "rerun_training_curves.npz",
    )
    return parser.parse_args()


def complete_scalar(
    accumulator: EventAccumulator,
    tag: str,
    iterations: int,
) -> np.ndarray:
    by_step = {int(event.step): float(event.value) for event in accumulator.Scalars(tag)}
    missing = [step for step in range(iterations) if step not in by_step]
    if missing:
        raise RuntimeError(f"{tag} is missing {len(missing)} iterations.")
    return np.asarray(
        [by_step[step] for step in range(iterations)],
        dtype=np.float64,
    )


def main() -> None:
    args = parse_args()
    iterations = int(CONFIG["training"]["iterations"])
    arrays: dict[str, np.ndarray] = {}
    for method in METHODS:
        for seed in CONFIG[method]["seeds"]:
            run = args.log_root / f"paper_{method}_seed{seed}_{iterations}"
            if not (run / f"model_{iterations - 1}.pt").is_file():
                raise RuntimeError(f"Incomplete paper rerun: {run}")
            accumulator = EventAccumulator(str(run), size_guidance={"scalars": 0})
            accumulator.Reload()
            for metric, tag in METRICS.items():
                arrays[f"training__{method}__{seed}__{metric}"] = complete_scalar(
                    accumulator,
                    tag,
                    iterations,
                )
            if method == "yaw":
                arrays[f"composer__{seed}__beta"] = complete_scalar(
                    accumulator,
                    "OuterComposer/beta",
                    iterations,
                )
                for group in GROUPS:
                    arrays[f"composer__{seed}__raw_{group}"] = complete_scalar(
                        accumulator,
                        f"OuterComposer/weight_{group}",
                        iterations,
                    )
                    arrays[f"composer__{seed}__effective_{group}"] = complete_scalar(
                        accumulator,
                        f"OuterComposer/effective_weight_{group}",
                        iterations,
                    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(f"Wrote {len(arrays)} rerun curves to {args.output}")


if __name__ == "__main__":
    main()
