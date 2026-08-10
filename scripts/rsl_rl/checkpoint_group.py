"""Atomic checkpoint-group helpers shared by training and playback."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


CHECKPOINT_GROUP_FORMAT_VERSION = 1


def checkpoint_manifest_path(policy_checkpoint_path: str | Path) -> Path:
    policy_path = Path(policy_checkpoint_path)
    return policy_path.with_name(f"checkpoint_group_{policy_path.stem}.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_checkpoint_group(
    policy_checkpoint_path: str | Path,
    writers: Mapping[str, tuple[str | Path, Callable[[str, str], None]]],
) -> str:
    """Write all artifacts to temporary files and publish a manifest last."""

    policy_path = Path(policy_checkpoint_path)
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    group_id = uuid.uuid4().hex
    temporary_paths: dict[str, Path] = {}
    final_paths: dict[str, Path] = {}
    temporary_manifest: Path | None = None
    try:
        for role, (final_path_value, writer) in writers.items():
            final_path = Path(final_path_value)
            if final_path.parent.resolve() != policy_path.parent.resolve():
                raise ValueError("All checkpoint-group artifacts must share the policy checkpoint directory.")
            temporary_path = final_path.with_name(f".{final_path.name}.{group_id}.tmp")
            temporary_paths[role] = temporary_path
            final_paths[role] = final_path
            writer(str(temporary_path), group_id)
            if not temporary_path.is_file():
                raise RuntimeError(f"Checkpoint writer did not create {temporary_path}.")

        entries = {
            role: {
                "filename": final_paths[role].name,
                "size": temporary_paths[role].stat().st_size,
                "sha256": _sha256(temporary_paths[role]),
            }
            for role in writers
        }
        for role in writers:
            os.replace(temporary_paths[role], final_paths[role])

        manifest = {
            "format_version": CHECKPOINT_GROUP_FORMAT_VERSION,
            "checkpoint_group_id": group_id,
            "artifacts": entries,
        }
        manifest_path = checkpoint_manifest_path(policy_path)
        temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.{group_id}.tmp")
        with temporary_manifest.open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_manifest, manifest_path)
        return group_id
    finally:
        for temporary_path in temporary_paths.values():
            temporary_path.unlink(missing_ok=True)
        if temporary_manifest is not None:
            temporary_manifest.unlink(missing_ok=True)


def validate_checkpoint_group(
    policy_checkpoint_path: str | Path,
    required_roles: tuple[str, ...] = ("policy", "predictor", "encoder", "pose", "actor"),
) -> dict[str, Any]:
    """Reject missing, incomplete, or mismatched checkpoint groups."""

    policy_path = Path(policy_checkpoint_path)
    manifest_path = checkpoint_manifest_path(policy_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Checkpoint group completion manifest is missing: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("format_version") != CHECKPOINT_GROUP_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint-group format {manifest.get('format_version')!r}; "
            f"expected {CHECKPOINT_GROUP_FORMAT_VERSION}."
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"Checkpoint manifest has no artifact map: {manifest_path}")
    for role in required_roles:
        entry = artifacts.get(role)
        if not isinstance(entry, dict):
            raise ValueError(f"Checkpoint manifest is missing required artifact role {role!r}.")
        artifact_path = manifest_path.parent / entry.get("filename", "")
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Checkpoint artifact is missing: {artifact_path}")
        if artifact_path.stat().st_size != entry.get("size") or _sha256(artifact_path) != entry.get("sha256"):
            raise ValueError(f"Checkpoint artifact failed size/SHA-256 validation: {artifact_path}")
    policy_entry = artifacts.get("policy", {})
    if policy_entry.get("filename") != policy_path.name:
        raise ValueError(
            f"Manifest policy artifact {policy_entry.get('filename')!r} does not match requested {policy_path.name!r}."
        )
    return manifest


def find_latest_complete_checkpoint(
    log_root_path: str | Path,
    run_pattern: str = ".*",
    checkpoint_pattern: str = r"model_\d+\.pt",
) -> str:
    """Return the newest policy checkpoint whose entire group validates."""

    root = Path(log_root_path)
    candidates: list[Path] = []
    if root.is_dir():
        for run_path in root.iterdir():
            if not run_path.is_dir() or re.fullmatch(run_pattern, run_path.name) is None:
                continue
            candidates.extend(
                path for path in run_path.iterdir() if path.is_file() and re.fullmatch(checkpoint_pattern, path.name)
            )
    candidates.sort(key=lambda path: (path.parent.name, path.stat().st_mtime_ns, path.name), reverse=True)
    validation_errors: list[str] = []
    for candidate in candidates:
        try:
            validate_checkpoint_group(candidate)
        except (FileNotFoundError, ValueError) as exc:
            validation_errors.append(f"{candidate}: {exc}")
            continue
        return str(candidate)
    detail = "\n".join(validation_errors[:5])
    raise FileNotFoundError(f"No complete predictive checkpoint group found under {root}.\n{detail}")


__all__ = [
    "checkpoint_manifest_path",
    "find_latest_complete_checkpoint",
    "save_checkpoint_group",
    "validate_checkpoint_group",
]
