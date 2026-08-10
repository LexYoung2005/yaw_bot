#!/usr/bin/env python3
"""Run the paper protocol without machine-specific paths."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "paper_experiments.json"
METHOD_ORDER = ("yaw", "outer", "uniform", "static", "lirpg", "relara")
LOG_ROOT = ROOT / "logs" / "rsl_rl" / "yaw_bot_predictive_gated"
ARTIFACT_ROOT = ROOT / "artifacts"


def load_config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="stage", required=True)
    for stage in ("train", "select", "evaluate", "plot"):
        subparser = subparsers.add_parser(stage)
        subparser.add_argument("--method", choices=METHOD_ORDER, action="append")
        subparser.add_argument("--dry-run", action="store_true")
        if stage == "train":
            subparser.add_argument("--num-envs", type=int)
            subparser.add_argument("--iterations", type=int)
            subparser.add_argument("--device", default="cuda:0")
            subparser.add_argument(
                "--gpu-id",
                help="Optional physical GPU id assigned through CUDA_VISIBLE_DEVICES.",
            )
        elif stage == "evaluate":
            subparser.add_argument("--num-envs", type=int)
            subparser.add_argument("--steps", type=int)
            subparser.add_argument("--device", default="cuda:0")
            subparser.add_argument("--gpu-id")
        elif stage == "plot":
            subparser.add_argument(
                "--rerun-data",
                action="store_true",
                help="Plot freshly generated logs/evaluations instead of submitted data.",
            )
    return result


def selected_methods(args: argparse.Namespace) -> tuple[str, ...]:
    return tuple(args.method) if args.method else METHOD_ORDER


def run_name(method: str, seed: int, iterations: int = 1500) -> str:
    return f"{method}_seed{seed}_{iterations}"


def run_directory(method: str, seed: int, iterations: int = 1500) -> Path:
    return LOG_ROOT / f"paper_{run_name(method, seed, iterations)}"


def command_text(command: list[str], environment: dict[str, str] | None = None) -> str:
    prefix = ""
    if environment:
        visible = {
            key: value
            for key, value in environment.items()
            if key in {"CUDA_VISIBLE_DEVICES", "YAWBOT_DISTRIBUTED_RUN_TIMESTAMP"}
        }
        prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in visible.items())
    body = shlex.join(command)
    return f"{prefix} {body}".strip()


def execute(
    command: list[str],
    *,
    dry_run: bool,
    environment: dict[str, str] | None = None,
) -> None:
    print(command_text(command, environment), flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def process_environment(gpu_id: str | None) -> dict[str, str]:
    environment = os.environ.copy()
    environment["YAWBOT_DISTRIBUTED_RUN_TIMESTAMP"] = "paper"
    if gpu_id is not None:
        environment["CUDA_VISIBLE_DEVICES"] = gpu_id
    return environment


def train(args: argparse.Namespace, config: dict[str, object]) -> None:
    training = config["training"]
    assert isinstance(training, dict)
    num_envs = args.num_envs or int(training["num_envs"])
    iterations = args.iterations or int(training["iterations"])
    environment = process_environment(args.gpu_id)
    for method in selected_methods(args):
        method_config = config[method]
        assert isinstance(method_config, dict)
        for seed in method_config["seeds"]:
            seed = int(seed)
            directory = run_directory(method, seed, iterations)
            final_checkpoint = directory / f"model_{iterations - 1}.pt"
            if final_checkpoint.is_file() and not args.dry_run:
                print(f"[skip] complete run: {directory.relative_to(ROOT)}")
                continue
            command = [
                sys.executable,
                "scripts/rsl_rl/train.py",
                "--task",
                str(method_config["task"]),
                "--num_envs",
                str(num_envs),
                "--max_iterations",
                str(iterations),
                "--seed",
                str(seed),
                "--run_name",
                run_name(method, seed, iterations),
                "--device",
                args.device,
                "--headless",
            ]
            if method == "static":
                command.extend(["--static_reward_weights", *[str(value) for value in method_config["weights"]]])
            execute(command, dry_run=args.dry_run, environment=environment)


def select(args: argparse.Namespace, config: dict[str, object]) -> None:
    selection = config["selection"]
    assert isinstance(selection, dict)
    output_dir = ARTIFACT_ROOT / "selections"
    for method in selected_methods(args):
        method_config = config[method]
        assert isinstance(method_config, dict)
        command = [
            sys.executable,
            "scripts/rsl_rl/select_best_reward_checkpoint.py",
            "--method",
            str(method_config["label"]),
            "--window",
            str(selection["trailing_window"]),
        ]
        for seed in method_config["seeds"]:
            relative_run = run_directory(method, int(seed)).relative_to(ROOT)
            command.extend(["--run-glob", str(relative_run)])
        command.extend(["--output", str((output_dir / f"{method}.json").relative_to(ROOT))])
        execute(command, dry_run=args.dry_run)


def evaluate(args: argparse.Namespace, config: dict[str, object]) -> None:
    evaluation = config["evaluation"]
    assert isinstance(evaluation, dict)
    num_envs = args.num_envs or int(evaluation["num_envs"])
    steps = args.steps or int(evaluation["steps"])
    environment = process_environment(args.gpu_id)
    output_dir = ARTIFACT_ROOT / "evaluation"
    for method in selected_methods(args):
        method_config = config[method]
        assert isinstance(method_config, dict)
        selection_path = ARTIFACT_ROOT / "selections" / f"{method}.json"
        if not selection_path.is_file() and not args.dry_run:
            raise FileNotFoundError(f"Missing {selection_path.relative_to(ROOT)}; run the select stage first.")
        checkpoint = (
            Path(json.loads(selection_path.read_text(encoding="utf-8"))["checkpoint"])
            if selection_path.is_file()
            else Path("<selected-checkpoint>")
        )
        for seed in evaluation["seeds"]:
            output_path = output_dir / f"{method}_seed{seed}.json"
            if output_path.is_file() and not args.dry_run:
                print(f"[skip] complete evaluation: {output_path.relative_to(ROOT)}")
                continue
            command = [
                sys.executable,
                "scripts/rsl_rl/play.py",
                "--task",
                str(method_config["task"]),
                "--checkpoint",
                str(checkpoint),
                "--num_envs",
                str(num_envs),
                "--max_steps",
                str(steps),
                "--seed",
                str(seed),
                "--evaluation",
                "--evaluation_output",
                str(output_path),
                "--device",
                args.device,
                "--headless",
            ]
            execute(command, dry_run=args.dry_run, environment=environment)


def plot(args: argparse.Namespace) -> None:
    methods = selected_methods(args)
    training_command = [sys.executable, "scripts/plot_training_curves.py"]
    evaluation_command = [sys.executable, "scripts/plot_results.py"]
    if args.rerun_data:
        rerun_archive = ARTIFACT_ROOT / "rerun_training_curves.npz"
        execute(
            [
                sys.executable,
                "scripts/export_training_curves.py",
                "--output",
                str(rerun_archive),
            ],
            dry_run=args.dry_run,
        )
        training_command.extend(["--data", str(rerun_archive)])
        evaluation_command.extend(["--input-dir", str(ARTIFACT_ROOT / "evaluation")])
    for method in methods:
        training_command.extend(["--method", method])
        evaluation_command.extend(["--method", method])
    execute(training_command, dry_run=args.dry_run)
    if "yaw" in methods:
        composer_command = [sys.executable, "scripts/plot_composer_weights.py"]
        if args.rerun_data:
            composer_command.extend(["--data", str(rerun_archive)])
        execute(composer_command, dry_run=args.dry_run)
    execute(evaluation_command, dry_run=args.dry_run)


def main() -> None:
    args = parser().parse_args()
    config = load_config()
    if args.stage == "plot":
        plot(args)
    else:
        {"train": train, "select": select, "evaluate": evaluate}[args.stage](args, config)


if __name__ == "__main__":
    main()
