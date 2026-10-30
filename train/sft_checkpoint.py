"""Checkpoint checksums and latest.json. Recovery never guesses a directory."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LATEST_SCHEMA = "qwen3-asr-latest-v1"
MANIFEST_SCHEMA = "qwen3-asr-checkpoint-manifest-v1"


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_text(path: Path, payload: str) -> None:
    atomic_write_bytes(path, payload.encode("utf-8"))


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def step_name(step: int) -> str:
    return f"step_{step:06d}"


def file_map(root: Path, relative_paths: list[str]) -> dict[str, str]:
    mapping = {}
    for relative in sorted(relative_paths):
        path = root / relative
        if not path.is_file():
            raise ValueError(f"checkpoint file missing: {relative}")
        mapping[relative] = sha256_file(path)
    return mapping


def directory_sha256(mapping: dict[str, str]) -> str:
    return sha256_text(canonical_json(mapping))


def write_checksums(root: Path, payload_paths: list[str], written_at: str) -> str:
    """Hash payload files, write the manifest, then hash that manifest too."""
    payload_map = file_map(root, payload_paths)
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "files": [
            {
                "path": relative,
                "bytes": (root / relative).stat().st_size,
                "sha256": digest,
                "written_at": written_at,
            }
            for relative, digest in payload_map.items()
        ],
    }
    atomic_write_json(root / "checkpoint_manifest.json", manifest)
    complete = dict(payload_map)
    complete["checkpoint_manifest.json"] = sha256_file(root / "checkpoint_manifest.json")
    digest = directory_sha256(complete)
    atomic_write_json(
        root / "sha256.json",
        {
            "schema_version": "qwen3-asr-checkpoint-sha-v1",
            "files": complete,
            "directory_sha256": digest,
        },
    )
    return digest


def verify_checkpoint_files(root: Path, expected_sha256: str) -> dict[str, Any]:
    """Recompute every listed file and the directory hash."""
    sha_path = root / "sha256.json"
    if not sha_path.is_file():
        raise ValueError(f"checkpoint is missing sha256.json: {root}")
    recorded = json.loads(sha_path.read_text(encoding="utf-8"))
    files = recorded.get("files")
    if not isinstance(files, dict) or "checkpoint_manifest.json" not in files:
        raise ValueError("checkpoint sha256.json is missing its file map")
    recomputed = file_map(root, list(files))
    if recomputed != files:
        raise ValueError("checkpoint file hash mismatch")
    digest = directory_sha256(recomputed)
    if digest != recorded.get("directory_sha256") or digest != expected_sha256:
        raise ValueError("checkpoint directory hash mismatch")
    return recorded


def verify_checkpoint(root: Path, expected_sha256: str) -> dict[str, Any]:
    """Resume only accepts a committed step directory."""
    if ".incoming" in root.parts or root.name.startswith("."):
        raise ValueError(f"refusing to resume from {root}")
    return verify_checkpoint_files(root, expected_sha256)


def write_latest(path: Path, *, step: int, checkpoint_sha256: str) -> dict[str, Any]:
    payload = {
        "schema_version": LATEST_SCHEMA,
        "step_name": step_name(step),
        "relative_path": f"checkpoints/{step_name(step)}",
        "checkpoint_sha256": checkpoint_sha256,
        "updated_at": utc_now(),
    }
    atomic_write_json(path, payload)
    return payload


def read_latest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"latest.json is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != LATEST_SCHEMA:
        raise ValueError("latest.json schema mismatch")
    relative = payload.get("relative_path")
    step = payload.get("step_name")
    digest = payload.get("checkpoint_sha256")
    if not isinstance(relative, str) or not isinstance(step, str) or not isinstance(digest, str):
        raise ValueError("latest.json is missing step_name, relative_path, or checkpoint_sha256")
    if relative != f"checkpoints/{step}" or ".incoming" in Path(relative).parts:
        raise ValueError("latest.json does not point at a committed checkpoint")
    return payload


def commit_checkpoint(staging: Path, run_dir: Path, step: int) -> str:
    """Move a hashed staging dir through .incoming, then point latest.json at it."""
    name = step_name(step)
    incoming_root = run_dir / "checkpoints" / ".incoming"
    incoming = incoming_root / name
    final = run_dir / "checkpoints" / name
    if final.exists() or incoming.exists():
        raise ValueError(f"checkpoint {name} already exists")
    incoming_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(staging, incoming)
    recorded = json.loads((incoming / "sha256.json").read_text(encoding="utf-8"))
    verify_checkpoint_files(incoming, str(recorded["directory_sha256"]))
    os.rename(incoming, final)
    digest = str(recorded["directory_sha256"])
    verify_checkpoint(final, digest)
    return digest


def prune_old_checkpoints(run_dir: Path, *, keep_last: int, current_step: int) -> list[str]:
    """Delete older committed steps. The step named by latest.json stays."""
    checkpoint_root = run_dir / "checkpoints"
    names = []
    if checkpoint_root.is_dir():
        for path in checkpoint_root.iterdir():
            if path.is_dir() and path.name.startswith("step_"):
                names.append(path.name)
    names.sort()
    protected = step_name(current_step)
    removable = [name for name in names if name != protected]
    overflow = removable[: max(0, len(removable) - (keep_last - 1))]
    removed = []
    for name in overflow:
        shutil.rmtree(checkpoint_root / name)
        removed.append(name)
    return removed
