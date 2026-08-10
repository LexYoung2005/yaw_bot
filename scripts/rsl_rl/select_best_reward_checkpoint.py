"""Select a robust best checkpoint using only the fixed outer objective."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


CHECKPOINT_RE = re.compile(r"model_(\d+)\.pt$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--run-glob", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=100)
    return parser.parse_args()


def read_outer_curve(run: Path) -> dict[int, float]:
    accumulator = EventAccumulator(str(run), size_guidance={"scalars": 0})
    accumulator.Reload()
    for tag in ("OuterComposer/outer_reward", "Evaluation/fixed_outer_reward"):
        if tag in accumulator.Tags().get("scalars", []):
            return {int(event.step): float(event.value) for event in accumulator.Scalars(tag)}
    raise RuntimeError(f"No fixed outer reward scalar in {run}")


def score_checkpoint(curve: dict[int, float], iteration: int, window: int) -> float:
    values = [curve[step] for step in sorted(curve) if iteration - window < step <= iteration]
    if len(values) < min(window, iteration + 1):
        raise RuntimeError(f"Only {len(values)} values available at iteration {iteration}")
    return float(np.mean(values))


def main() -> None:
    args = parse_args()
    candidates: list[dict[str, object]] = []
    seen: set[Path] = set()
    for pattern in args.run_glob:
        for run in sorted(Path().glob(pattern)):
            run = run.resolve()
            if (
                not run.is_dir()
                or run in seen
                or run.name.endswith("_rank1")
                or not (run / "model_1499.pt").exists()
            ):
                continue
            seen.add(run)
            curve = read_outer_curve(run)
            for checkpoint in run.glob("model_*.pt"):
                match = CHECKPOINT_RE.fullmatch(checkpoint.name)
                if match is None:
                    continue
                iteration = int(match.group(1))
                actor = run / f"actor_model_{iteration}.pt"
                pose = run / f"pose_predictor_model_{iteration}.pt"
                depth = run / f"depth_encoder_model_{iteration}.pt"
                if not all(path.exists() for path in (actor, pose, depth)):
                    continue
                candidates.append(
                    {
                        "method": args.method,
                        "run": str(run),
                        "checkpoint": str(checkpoint.resolve()),
                        "iteration": iteration,
                        "trailing_outer_reward": score_checkpoint(curve, iteration, args.window),
                        "selection_window": args.window,
                    }
                )
    if not candidates:
        raise RuntimeError(f"No complete checkpoint candidates for {args.method}")
    best = max(candidates, key=lambda item: (float(item["trailing_outer_reward"]), int(item["iteration"])))
    best["candidate_count"] = len(candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(best, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(best, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
