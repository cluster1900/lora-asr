#!/usr/bin/env python3
"""Record the Phase 0 environment before any training.

The probe writes Python, torch, CUDA, GPU, package versions, and Git state.
It does not load Qwen3-ASR and does not restore an optimizer.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.config import load_infer_config  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version
    except ImportError:  # pragma: no cover
        return None
    try:
        return version(name)
    except Exception:
        return None


def _git_sha(cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    sha = result.stdout.strip()
    return sha or None


def _git_dirty(cwd: Path) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(result.stdout.strip())


def collect_environment(repo_root: Path, git_sha_override: str | None) -> dict[str, Any]:
    """Collect software and GPU facts. Torch is optional for local contract tests."""
    gpu: dict[str, Any] = {"available": False}
    torch_version = None
    cuda_version = None
    deterministic_supported: bool | None = None
    deterministic_error = None
    bf16_supported = None
    try:
        import torch

        torch_version = torch.__version__
        cuda_version = torch.version.cuda
        gpu["available"] = bool(torch.cuda.is_available())
        if gpu["available"]:
            properties = torch.cuda.get_device_properties(0)
            gpu.update({
                "name": torch.cuda.get_device_name(0),
                "total_memory_bytes": int(properties.total_memory),
                "device_count": int(torch.cuda.device_count()),
                "capability": [int(properties.major), int(properties.minor)],
            })
            bf16_supported = bool(torch.cuda.is_bf16_supported())
        try:
            torch.use_deterministic_algorithms(True)
            deterministic_supported = True
        except Exception as exc:
            deterministic_supported = False
            deterministic_error = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        gpu["torch_import_error"] = f"{type(exc).__name__}: {exc}"

    disk = shutil.disk_usage(repo_root if repo_root.exists() else Path.cwd())
    return {
        "schema_version": "qwen3-asr-environment-v1",
        "collected_at": _utc_now(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch_version,
        "cuda": cuda_version,
        "qwen_asr": _package_version("qwen-asr"),
        "transformers": _package_version("transformers"),
        "peft": _package_version("peft"),
        "numpy": _package_version("numpy"),
        "pyyaml": _package_version("PyYAML"),
        "soundfile": _package_version("soundfile"),
        "gpu": gpu,
        "bf16_supported": bf16_supported,
        "deterministic_algorithms_supported": deterministic_supported,
        "deterministic_algorithms_error": deterministic_error,
        "git_sha": git_sha_override or _git_sha(repo_root),
        "git_dirty": _git_dirty(repo_root),
        "disk_free_bytes": int(disk.free),
        "world_size": 1,
    }


def write_probe(
    output_dir: Path,
    config_path: Path,
    run_id: str,
    seed: int,
    git_sha: str | None,
    repo_root: Path,
) -> dict[str, Any]:
    """Write request, environment, and resource files for one run."""
    config = load_infer_config(config_path)
    if seed != config["seed"]:
        raise ValueError(f"CLI seed {seed} does not match config seed {config['seed']}")
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = collect_environment(repo_root, git_sha)
    environment["run_id"] = run_id
    environment["seed"] = seed
    environment["decode_eval_sha256"] = config["decode_eval_sha256"]
    request = {
        "schema_version": "qwen3-asr-request-v1",
        "run_id": run_id,
        "stage": "phase0_baseline",
        "seed": seed,
        "allow_dirty": True,
        "config": str(config_path),
        "model_name": config["model"]["name"],
        "created_at": _utc_now(),
    }
    resource = {
        "schema_version": "qwen3-asr-resource-v1",
        "gpu": environment["gpu"],
        "bf16_supported": environment["bf16_supported"],
        "disk_free_bytes": environment["disk_free_bytes"],
        "world_size": 1,
        "dtype_plan": "bfloat16" if environment["bf16_supported"] else "float32",
    }
    (output_dir / "request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "resource_report.json").write_text(
        json.dumps(resource, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "resolved_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return environment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="记录 Phase 0 环境指纹")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--git-sha", default=None)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        environment = write_probe(
            args.output_dir,
            args.config,
            args.run_id,
            args.seed,
            args.git_sha,
            args.repo_root,
        )
    except (OSError, ValueError) as exc:
        print(f"colab_bootstrap failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "run_id": args.run_id,
        "python": environment["python"],
        "torch": environment["torch"],
        "gpu": environment["gpu"].get("name"),
        "git_sha": environment["git_sha"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
