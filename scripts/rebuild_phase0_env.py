#!/usr/bin/env python3
"""Install the portable Phase 0 lock without replacing a matching CUDA torch.

Verify lines are checked against the runtime that is already present. A
mismatch exits before pip runs, so a different preinstalled torch is left
as it is. The full Colab pip freeze is a fingerprint, not this file.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


SCHEMA = "qwen3-asr-portable-lock-v1"
REPORT_SCHEMA = "qwen3-asr-lock-rebuild-v1"


def parse_portable_lock(text: str) -> dict[str, Any]:
    """Split a portable lock into verify pins and exact install pins."""
    schema = None
    verify: list[tuple[str, str]] = []
    install: list[tuple[str, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("# schema:"):
            schema = line.split(":", 1)[1].strip()
            continue
        if line.startswith("# verify:"):
            verify.append(parse_pin(line.split(":", 1)[1].strip()))
            continue
        if line.startswith("#"):
            continue
        install.append(parse_pin(line))
    if schema != SCHEMA:
        raise ValueError(f"lock schema must be {SCHEMA}, got {schema!r}")
    if not verify or not install:
        raise ValueError("lock needs at least one verify pin and one install pin")
    return {"schema": schema, "verify": verify, "install": install}


def parse_pin(text: str) -> tuple[str, str]:
    """Parse one name==version pin."""
    if text.count("==") != 1:
        raise ValueError(f"pin must be name==version: {text}")
    name, pin_version = (part.strip() for part in text.split("=="))
    if not name or not pin_version or any(char in name for char in " <>!"):
        raise ValueError(f"invalid pin: {text}")
    return name, pin_version


def installed_version(name: str) -> str | None:
    """Return the installed distribution version, or None when it is absent."""
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def pin_rows(pins: list[tuple[str, str]], actual: dict[str, str | None]) -> list[dict[str, Any]]:
    """Compare expected pins with a version map."""
    rows = []
    for name, expected in pins:
        seen = actual.get(name)
        rows.append(
            {
                "name": name,
                "expected": expected,
                "actual": seen,
                "match": seen == expected,
            }
        )
    return rows


def collect_versions(pins: list[tuple[str, str]]) -> dict[str, str | None]:
    """Read currently installed versions for the given pins."""
    return {name: installed_version(name) for name, _version in pins}


def torchao_blocks_peft(version: str | None) -> bool:
    """Return whether peft 0.21 would raise on this torchao version.

    A missing package is safe. peft skips torchao when it cannot be found, and
    raises when an installed copy is older than 0.16.0.
    """
    if version is None:
        return False
    numbers: list[int] = []
    for piece in version.split("."):
        digits = ""
        for char in piece:
            if char.isdigit():
                digits += char
            else:
                break
        if not digits:
            break
        numbers.append(int(digits))
    while len(numbers) < 3:
        numbers.append(0)
    return tuple(numbers[:3]) < (0, 16, 0)


def cuda_available() -> bool | None:
    """Return torch CUDA availability. None means torch itself did not import."""
    try:
        import torch
    except Exception:
        return None
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    lock = parse_portable_lock(args.lock.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    before_versions = collect_versions(lock["verify"] + lock["install"])
    cuda_before = cuda_available()
    verify_before = pin_rows(lock["verify"], before_versions)
    verify_ok = all(row["match"] for row in verify_before) and cuda_before is True
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "lock": str(args.lock),
        "cuda_before": cuda_before,
        "verify_before": verify_before,
        "pip_exit_code": None,
        "cuda_after": None,
        "verify_after": None,
        "install_after": None,
    }
    if not verify_ok:
        report["status"] = "verify_mismatch"
        _write_report(args.output_dir, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2

    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        *[f"{name}=={pin_version}" for name, pin_version in lock["install"]],
    ]
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    (args.output_dir / "pip_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (args.output_dir / "pip_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    report["pip_exit_code"] = completed.returncode
    after_versions = collect_versions(lock["verify"] + lock["install"])
    report["cuda_after"] = cuda_available()
    report["verify_after"] = pin_rows(lock["verify"], after_versions)
    report["install_after"] = pin_rows(lock["install"], after_versions)
    pins_ok = all(row["match"] for row in report["verify_after"] + report["install_after"])
    cuda_ok = report["cuda_after"] is True
    if completed.returncode != 0:
        report["status"] = "install_failed"
        code = 1
    elif not pins_ok or not cuda_ok:
        report["status"] = "post_mismatch"
        code = 1
    else:
        report["status"] = "passed"
        code = 0
        seen_torchao = installed_version("torchao")
        report["torchao_before_remove"] = seen_torchao
        if torchao_blocks_peft(seen_torchao):
            removed = subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
                check=False,
                text=True,
                capture_output=True,
            )
            report["torchao_uninstall_exit_code"] = removed.returncode
            report["torchao_after_remove"] = installed_version("torchao")
            if removed.returncode != 0 or torchao_blocks_peft(report["torchao_after_remove"]):
                report["status"] = "torchao_remove_failed"
                code = 1
    _write_report(args.output_dir, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return code


def _write_report(output_dir: Path, report: dict[str, Any]) -> None:
    (output_dir / "rebuild_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
