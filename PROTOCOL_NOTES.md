# Exact Environment Protocol Notes

This file records details that are easy to compress too aggressively in the
paper text.

## Friction

The shared terrain material uses static and dynamic coefficients **1.4** and
**1.2**, respectively, exactly as stated in the supplementary material. The
simulator uses multiplicative material combination. At environment startup,
robot rigid-body materials are additionally randomized with static friction in
`[0.8, 1.6]` and dynamic friction in `[0.7, 1.3]`.

Thus, 1.4/1.2 are the terrain-side coefficients, while the checked-in exact
configuration also contains robot-side domain randomization. The code retains
the configuration used by the reported experiments; changing the randomization
to a fixed value would no longer reproduce those runs.

## Other startup and interval randomization

- Body mass receives an additive sample in `[-0.1, 0.2]`.
- Velocity pushes occur every 3–7 seconds with the ranges in `EventCfg`.

These settings are shared by all six methods. They are not reward-method
variables.

## Submitted result data

`results/submitted_training_curves.npz` contains 1500-point release-safe scalar
curves for all 18 training runs and the raw/effective Composer weights for all
three YAW runs. The archive keys encode only method, seed, and metric—no
development paths or timestamps.

`results/evaluation_json/` contains the 18 independent-play evaluations.
Checkpoint paths have been replaced by a release-safe placeholder while
checkpoint iteration, seed, workload, completed episodes, and all measured
metrics are preserved.
