# Submitted Result Data

- `evaluation_json/`: 18 independent-play evaluations (six methods by three
  fresh seeds), with unavailable checkpoint paths replaced by release-safe
  placeholders.
- `evaluation_per_seed.csv`: the same per-seed metrics in tabular form.
- `evaluation_aggregate.csv`: means and sample standard deviations used by the
  paper's eight-metric evaluation figure.
- `submitted_training_curves.npz`: 141 arrays of length 1500. Keys follow
  `training__<method>__<seed>__<metric>` and
  `composer__<seed>__<quantity>`.
- `training_per_seed.csv`: fixed-task reward AUC and final-100 mean for each
  training run.
- `composer_weight_evolution_summary.csv`: final-100 and observed Composer
  weight statistics.

The NPZ and JSON data contain no user name, host path, timestamp, private
repository remote, or server credential.
