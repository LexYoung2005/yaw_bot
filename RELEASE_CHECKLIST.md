# Public Release Checklist

- Public package metadata identifies the YAW project and repository.
- No personal home paths, user names, e-mail addresses, private repository
  URLs, or server credentials are included.
- No development Git history, training logs, checkpoints, cache files, or
  machine-specific timestamps are included.
- Result CSV files contain aggregate numeric results only.
- Per-seed JSON files retain the numeric protocol and results but replace
  checkpoint paths with release-safe placeholders.
- Submitted training curves use only method, seed, metric, and iteration keys.
- Launchers resolve paths from this repository at runtime.
- Necessary upstream citations, implementation provenance, and license notices
  are retained.

Run `python scripts/validate_release.py` before packaging or publishing.
