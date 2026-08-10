# YAW: Code and Data Appendix

Official implementation and submitted experiment data for **“YAW: Predictive
Reward Composition via Task-Advantage Alignment for Wheel-Legged
Locomotion.”** YAW (You Always Walk) learns a bounded, mean-one composition of
22 locomotion rewards and validates the induced virtual PPO update with a fixed
task advantage on complementary environment indices.

Authors: Lexing Yang, Houbao Ji, Shaolong Shen, Hongji Huang, Xiangxiao Chen,
and Yufeng Ding.

Language: [简体中文](./README_cn.md)

This repository contains:

- the two-wheel, six-action Isaac Lab task;
- YAW and all five comparison methods used in the paper;
- training, checkpoint-selection, independent-evaluation, and plotting entry points;
- CPU regression tests and machine-readable experiment configurations; and
- the submitted training curves and fresh-seed evaluation results.

Development-machine paths, logs, and checkpoints are intentionally excluded.
Third-party attributions required for reproducibility and license compliance
are retained.

## Environment

The submitted experiments used Ubuntu 22.04, Isaac Sim 5.1, Isaac Lab, Python
3.11, PyTorch 2.7, and `rsl-rl-lib==3.1.2`. Install this project inside a
working Isaac Lab environment:

```bash
python -m pip install -e source/yaw_bot
```

The robot USD, URDF, and meshes are included under `assets/robots/yaw_bot`; no
asset-conversion step is needed. The complete machine specification and tested
dependency versions are in `configs/environment.json`. Additional protocol
details, including terrain-side friction and robot-side domain randomization,
are recorded in `PROTOCOL_NOTES.md`.

## Fast Validation

The CPU-only checks do not launch Isaac Sim:

```bash
python -B -m unittest discover -s tests -p "test_*.py"
python scripts/validate_release.py
```

## Reproduce the Paper Experiments

Print the exact commands without launching training:

```bash
python scripts/reproduce.py train --dry-run
```

Train all six methods for three seeds each:

```bash
python scripts/reproduce.py train
```

Select checkpoints by the highest trailing-100 training task reward, evaluate
the selected checkpoint with three fresh seeds under the matched automatic
command distribution, and recreate the paper tables and figures:

```bash
python scripts/reproduce.py select
python scripts/reproduce.py evaluate
python scripts/reproduce.py plot
```

The plot stage works immediately on the included submitted data and does not
require checkpoints or development-machine logs. After a fresh rerun, analyze
the new logs and evaluation JSON with:

```bash
python scripts/reproduce.py plot --rerun-data
```

Use `--method yaw` (or another method name) to run only one method. Device and
workload sizes can be overridden for smoke tests; the defaults are the paper
values:

```bash
python scripts/reproduce.py train --method yaw --num-envs 2 --iterations 2
```

## Paper Methods and Task IDs

| Paper method | Task ID |
|---|---|
| YAW | `Template-Yaw-Bot-Predictive-Gated-Direct-v0` |
| Outer-only PPO | `Template-Yaw-Bot-Outer-Only-PPO-Direct-v0` |
| Uniform | `Template-Yaw-Bot-Uniform-Reward-PPO-Direct-v0` |
| Static | `Template-Yaw-Bot-Static-Reward-PPO-Direct-v0` |
| PPO-LIRPG | `Template-Yaw-Bot-LIRPG-PPO-Direct-v0` |
| ReLara-PPO | `Template-Yaw-Bot-ReLara-PPO-Direct-v0` |

See `PAPER_CODE_MAP.md` for the equation-by-equation correspondence and
`configs/paper_experiments.json` for the executable protocol. The 22 atomic
reward definitions and scales are available in `configs/reward_terms.json`.

## Result Data

`results/` contains the submitted training curves, 18 independent-play
evaluation JSON files (six methods by three fresh seeds), per-seed tables, and
aggregate statistics used in the paper figures. Checkpoint paths are replaced
with release-safe placeholders because checkpoints are not part of this
repository.

## Citation

```bibtex
@misc{yang2026yaw,
  title  = {YAW: Predictive Reward Composition via Task-Advantage Alignment for Wheel-Legged Locomotion},
  author = {Yang, Lexing and Ji, Houbao and Shen, Shaolong and Huang, Hongji and Chen, Xiangxiao and Ding, Yufeng},
  year   = {2026},
  note   = {Preprint}
}
```

## License and Third-Party Notice

The project code is released under the BSD 3-Clause License. See `LICENSE` and
`NOTICE.md`. Individual upstream-derived files retain their original copyright
and SPDX headers.
