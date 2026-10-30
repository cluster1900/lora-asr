#!/usr/bin/env python3
"""Run the Phase 0 baseline: probe, two clips, one failure, CER, readback.

The notebook and the Colab CLI both call this script. Model code stays in
inference/ and evaluation/. A Drive root that is not mounted is reported and
does not pretend the Drive contract passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.jsonl_io import read_jsonl  # noqa: E402

CLEAN_ID = "phase0_zh_clean"
NOISE_ID = "phase0_zh_noise"
MISSING_ID = "phase0_zh_missing"
READBACK_NAMES = [
    "environment.json",
    "request.json",
    "resource_report.json",
    "predictions/smoke.jsonl",
    "eval/metrics.json",
    "eval/scored.jsonl",
    "eval/by_scenario.csv",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class CommandLog:
    """Append each child command, its exit code, and both output streams."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.step = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(
                "# Phase 0 命令记录\n\n"
                "本文件由 `scripts/run_phase0_baseline.py` 追加。"
                "每一步保留脚本命令、退出码、stdout 和 stderr。\n\n",
                encoding="utf-8",
            )

    def run(self, title: str, command: list[str]) -> subprocess.CompletedProcess[str]:
        self.step += 1
        started = _utc_now()
        result = subprocess.run(command, cwd=ROOT, check=False, text=True, capture_output=True)
        self._append(
            title,
            command,
            result.returncode,
            started,
            _utc_now(),
            result.stdout,
            result.stderr,
        )
        return result

    def note(self, title: str, body: str) -> None:
        self.step += 1
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(f"## {self.step}. {title}\n\n{body.rstrip()}\n\n")

    def _append(
        self,
        title: str,
        command: list[str],
        exit_code: int,
        started: str,
        finished: str,
        stdout: str,
        stderr: str,
    ) -> None:
        rendered = " ".join(shlex.quote(part) for part in command)
        text = (
            f"## {self.step}. {title}\n\n"
            f"- started_at: `{started}`\n"
            f"- finished_at: `{finished}`\n"
            f"- exit_code: `{exit_code}`\n"
            "- command:\n\n"
            f"```bash\n{rendered}\n```\n\n"
            "### stdout\n\n"
            f"```text\n{stdout.rstrip() or '(empty)'}\n```\n\n"
            "### stderr\n\n"
            f"```text\n{stderr.rstrip() or '(empty)'}\n```\n\n"
        )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(text)


def resolve_drive_destination(
    drive_root: Path | None,
    output_dir: Path,
    run_id: str,
) -> tuple[Path, str]:
    """Use Drive when its parent directory exists. Otherwise keep a local copy."""
    local = output_dir / "readback_local"
    if drive_root is None:
        return local, "unmounted"
    parent = drive_root.parent
    if not drive_root.exists() and not parent.is_dir():
        return local, "missing"
    drive_root.mkdir(parents=True, exist_ok=True)
    probe = drive_root / ".write_probe"
    probe.write_text("ok\n", encoding="utf-8")
    read_back = probe.read_text(encoding="utf-8")
    probe.unlink()
    if read_back != "ok\n":
        return local, "unreadable"
    return drive_root / "runs" / run_id, "mounted"


def _fail(output_dir: Path, message: str, code: int) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "schema_version": "qwen3-asr-status-v1",
        "status": "failed",
        "reason": message,
        "updated_at": _utc_now(),
    }
    (output_dir / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(message, file=sys.stderr)
    return code


def _check_predictions(path: Path) -> dict[str, Any]:
    rows = {row["sample_id"]: row for row in read_jsonl(path)}
    missing_ids = [sample_id for sample_id in (CLEAN_ID, NOISE_ID, MISSING_ID) if sample_id not in rows]
    if missing_ids:
        raise ValueError(f"predictions missing sample ids: {missing_ids}")
    problems: list[str] = []
    for sample_id in (CLEAN_ID, NOISE_ID):
        if rows[sample_id].get("error"):
            problems.append(f"{sample_id} failed: {rows[sample_id]['error']}")
        elif not str(rows[sample_id].get("prediction_raw") or "").strip():
            problems.append(f"{sample_id} produced an empty transcript")
    if not rows[MISSING_ID].get("error"):
        problems.append("the missing-file row did not record an error")
    if problems:
        raise ValueError("; ".join(problems))
    comparisons = []
    for sample_id in (CLEAN_ID, NOISE_ID):
        comparisons.append({
            "sample_id": sample_id,
            "prediction_raw": rows[sample_id].get("prediction_raw"),
            "transcribe_text": rows[sample_id].get("transcribe_text"),
            "match": rows[sample_id].get("transcribe_matches_greedy"),
            "stop_reason": rows[sample_id].get("stop_reason"),
        })
    return {"rows": 3, "comparisons": comparisons, "model_revision": rows[CLEAN_ID].get("model_revision")}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    comparisons = payload["comparisons"]
    lines = [
        "# Phase 0 baseline",
        "",
        f"- run_id: `{payload['run_id']}`",
        f"- status: `{payload['status']}`",
        f"- seed: `{payload['seed']}`",
        f"- git_sha: `{payload.get('git_sha')}`",
        f"- model_revision: `{payload.get('model_revision')}`",
        f"- micro_cer: `{payload.get('micro_cer')}`",
        f"- macro_cer: `{payload.get('macro_cer')}`",
        f"- drive: `{payload['drive_status']}`",
        "",
        "## generate 与 transcribe",
        "",
    ]
    for item in comparisons:
        lines.append(
            f"- `{item['sample_id']}` match={item['match']} stop={item['stop_reason']}"
        )
        lines.append(f"  - greedy: {item['prediction_raw']}")
        lines.append(f"  - transcribe: {item['transcribe_text']}")
    lines.extend([
        "",
        "两条文本不一致时，不能开始 SFT。本报告只记录 Phase 0 的探针、转写和 CER。",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="跑通 Phase 0 环境探针、两条转写、CER 和读回")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/inference/base_eval.yaml")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--manifest-dir", type=Path)
    parser.add_argument("--drive-root", type=Path)
    parser.add_argument("--git-sha", default=None)
    parser.add_argument("--skip-prepare", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log = CommandLog(output_dir / "command_log.md")
    log.note(
        "orchestrator",
        "- script: `scripts/run_phase0_baseline.py`\n"
        f"- run_id: `{args.run_id}`\n"
        f"- seed: `{args.seed}`\n"
        f"- config: `{args.config}`\n"
        f"- drive_root: `{args.drive_root}`\n"
        f"- git_sha: `{args.git_sha}`\n"
        f"- skip_prepare: `{args.skip_prepare}`",
    )
    manifest_dir = args.manifest_dir or (output_dir / "manifest")
    python = sys.executable

    if not args.skip_prepare:
        prepared = log.run("prepare_phase0_audio.py", [
            python,
            str(ROOT / "scripts/prepare_phase0_audio.py"),
            "--manifest-dir",
            str(manifest_dir),
            "--download-dir",
            str(output_dir / "download"),
        ])
        if prepared.returncode != 0:
            sys.stderr.write(prepared.stderr)
            return _fail(output_dir, "audio preparation failed", prepared.returncode or 1)

    manifest = manifest_dir / "smoke.jsonl"
    if not manifest.is_file():
        return _fail(output_dir, f"missing manifest {manifest}", 2)

    bootstrap = log.run("colab_bootstrap.py", [
        python,
        str(ROOT / "scripts/colab_bootstrap.py"),
        "--config",
        str(args.config),
        "--output-dir",
        str(output_dir),
        "--run-id",
        args.run_id,
        "--seed",
        str(args.seed),
        *(["--git-sha", args.git_sha] if args.git_sha else []),
    ])
    if bootstrap.returncode != 0:
        sys.stderr.write(bootstrap.stderr)
        return _fail(output_dir, "environment probe failed", bootstrap.returncode or 1)
    print(bootstrap.stdout.strip())

    predictions = output_dir / "predictions" / "smoke.jsonl"
    inferred = log.run("run_inference.py", [
        python,
        str(ROOT / "inference/run_inference.py"),
        "--config",
        str(args.config),
        "--manifest",
        str(manifest),
        "--output",
        str(predictions),
        "--mode",
        "eval",
        "--seed",
        str(args.seed),
    ])
    sys.stdout.write(inferred.stdout)
    sys.stderr.write(inferred.stderr)
    if inferred.returncode == 2 or not predictions.is_file():
        return _fail(output_dir, "inference failed before writing a usable file", inferred.returncode or 1)
    try:
        prediction_summary = _check_predictions(predictions)
    except ValueError as exc:
        return _fail(output_dir, str(exc), 1)

    evaluated = log.run("eval_cer.py", [
        python,
        str(ROOT / "evaluation/eval_cer.py"),
        "--config",
        str(args.config),
        "--manifest",
        str(manifest),
        "--predictions",
        str(predictions),
        "--output-dir",
        str(output_dir / "eval"),
    ])
    sys.stdout.write(evaluated.stdout)
    sys.stderr.write(evaluated.stderr)
    if evaluated.returncode != 0:
        return _fail(output_dir, "CER evaluation failed", evaluated.returncode or 1)

    manifest_hash = _sha256(manifest)
    (output_dir / "manifest_sha256.json").write_text(
        json.dumps({"smoke.jsonl": {"sha256": manifest_hash, "bytes": manifest.stat().st_size}}, indent=2) + "\n",
        encoding="utf-8",
    )
    metrics = json.loads((output_dir / "eval" / "metrics.json").read_text(encoding="utf-8"))
    if metrics["manifest_sha256"] != manifest_hash:
        return _fail(output_dir, "metrics manifest hash does not match the smoke manifest", 1)

    readback_dest, drive_status = resolve_drive_destination(args.drive_root, output_dir, args.run_id)
    log.note(
        "drive destination",
        f"- drive_status: `{drive_status}`\n- readback_dest: `{readback_dest}`",
    )
    readback = log.run("phase0_readback.py", [
        python,
        str(ROOT / "scripts/phase0_readback.py"),
        "--source-dir",
        str(output_dir),
        "--dest-dir",
        str(readback_dest),
        "--report",
        str(output_dir / "readback.json"),
        *[item for name in READBACK_NAMES for item in ("--name", name)],
    ])
    sys.stdout.write(readback.stdout)
    sys.stderr.write(readback.stderr)
    if readback.returncode != 0:
        return _fail(output_dir, "artifact readback failed", readback.returncode or 1)

    environment = json.loads((output_dir / "environment.json").read_text(encoding="utf-8"))
    status_name = "completed" if drive_status == "mounted" else "completed_local_readback"
    report_payload = {
        "run_id": args.run_id,
        "status": status_name,
        "seed": args.seed,
        "git_sha": environment.get("git_sha"),
        "model_revision": prediction_summary["model_revision"],
        "micro_cer": metrics["micro_cer"],
        "macro_cer": metrics["macro_cer"],
        "drive_status": drive_status,
        "comparisons": prediction_summary["comparisons"],
    }
    _write_report(output_dir / "run_report.md", report_payload)
    status = {
        "schema_version": "qwen3-asr-status-v1",
        "status": status_name,
        "drive_status": drive_status,
        "run_id": args.run_id,
        "updated_at": _utc_now(),
        "transcribe_matches_greedy": all(item["match"] for item in prediction_summary["comparisons"]),
    }
    (output_dir / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False))
    if drive_status != "mounted":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
