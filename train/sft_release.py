#!/usr/bin/env python3
"""Freeze the gate-selected SFT adapter and read its greedy text back.

``latest.json`` stays a resume pointer. This tool never copies that step
unless the gate selected it. ``prepare`` writes ``staged``. ``readback``
sets ``published`` only when every saved val row matches.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.sft_checkpoint import (  # noqa: E402
    atomic_write_json,
    directory_sha256,
    file_map,
    read_latest,
    sha256_file,
    step_name,
    utc_now,
    verify_checkpoint,
)
from train.sft_schema import load_train_config, semantic_config_sha256  # noqa: E402

RELEASE_SCHEMA = "qwen3-asr-sft-release-v1"
RELEASE_SHA_SCHEMA = "qwen3-asr-sft-release-sha-v1"
READBACK_SCHEMA = "qwen3-asr-sft-release-readback-v1"
PAYLOAD_FILES = (
    "adapter/adapter.safetensors",
    "adapter/adapter_config.json",
    "source/gate.json",
    "source/scored.jsonl",
    "source/training_state.json",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path} line {line_number} is not an object")
        rows.append(row)
    return rows


def decision_for_step(rows: list[dict[str, Any]], step: int) -> dict[str, Any]:
    found = [row for row in rows if int(row.get("step", -1)) == step]
    if len(found) != 1:
        raise ValueError(f"expected one decision for step {step}, found {len(found)}")
    return found[0]


def require_passed_decision(decision: dict[str, Any], step: int) -> None:
    if decision.get("passed") is not True or decision.get("risk_stop") is not False:
        raise ValueError(f"step {step} is not a passed checkpoint")


def greedy_fields(rows: list[dict[str, Any]], source: str) -> dict[str, dict[str, str]]:
    """Map sample_id to the greedy text fields a readback must reproduce."""
    mapped: dict[str, dict[str, str]] = {}
    for index, row in enumerate(rows, start=1):
        sample_id = row.get("sample_id")
        prediction = row.get("prediction_raw")
        reason = row.get("stop_reason")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{source} row {index} is missing sample_id")
        if sample_id in mapped:
            raise ValueError(f"{source} repeats sample_id {sample_id}")
        if not isinstance(prediction, str):
            raise ValueError(f"{source} row {sample_id} is missing prediction_raw")
        if not isinstance(reason, str) or not reason:
            raise ValueError(f"{source} row {sample_id} is missing stop_reason")
        error = row.get("error")
        mapped[sample_id] = {
            "prediction_raw": prediction,
            "stop_reason": reason,
            "error": "" if error in (None, "") else str(error),
        }
    if not mapped:
        raise ValueError(f"{source} has no rows")
    return mapped


def compare_greedy_texts(
    saved: dict[str, dict[str, str]],
    fresh: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Return a readback report. One differing row keeps matched false."""
    mismatches = []
    saved_ids = set(saved)
    fresh_ids = set(fresh)
    for sample_id in sorted(saved_ids.symmetric_difference(fresh_ids)):
        mismatches.append({"sample_id": sample_id, "reason": "sample_id set differs"})
    for sample_id in sorted(saved_ids & fresh_ids):
        left = saved[sample_id]
        right = fresh[sample_id]
        fields = [
            name
            for name in ("prediction_raw", "stop_reason", "error")
            if left[name] != right[name]
        ]
        if fields:
            mismatches.append({"sample_id": sample_id, "fields": fields})
    return {
        "schema_version": READBACK_SCHEMA,
        "matched": not mismatches,
        "compared": len(saved_ids & fresh_ids),
        "saved_rows": len(saved),
        "fresh_rows": len(fresh),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:20],
    }


def _payload_record(root: Path) -> dict[str, Any]:
    mapping = file_map(root, list(PAYLOAD_FILES))
    return {
        "schema_version": RELEASE_SHA_SCHEMA,
        "files": mapping,
        "directory_sha256": directory_sha256(mapping),
    }


def _write_payload_sha(root: Path) -> dict[str, Any]:
    record = _payload_record(root)
    atomic_write_json(root / "sha256.json", record)
    return record


def prepare_release(
    *,
    config_path: Path,
    run_dir: Path,
    output_dir: Path,
    release_id: str,
) -> dict[str, Any]:
    """Copy the selected adapter. Refuse a second write or a failed gate."""
    if output_dir.exists():
        raise ValueError(f"release directory already exists: {output_dir}")
    config = load_train_config(config_path)
    if config["profile"] != "pilot":
        raise ValueError("only a pilot profile can be released")
    if config["model"]["parent_release_sha256"] is not None:
        raise ValueError("this SFT release must have a null parent")
    gate = json.loads((run_dir / "gate.json").read_text(encoding="utf-8"))
    if gate.get("status") != "passed":
        raise ValueError("gate status is not passed")
    selected = gate.get("selected_step")
    if not isinstance(selected, int):
        raise ValueError("gate is missing selected_step")
    if release_id != f"sft_06b_zh_{_run_stamp(run_dir)}_step{selected:06d}":
        raise ValueError(f"release id does not match the selected step: {release_id}")
    decisions = read_jsonl(run_dir / "eval_decisions.jsonl")
    decision = decision_for_step(decisions, selected)
    require_passed_decision(decision, selected)
    embedded = gate.get("decisions")
    if isinstance(embedded, list) and embedded:
        require_passed_decision(decision_for_step(embedded, selected), selected)
    checkpoint = run_dir / "checkpoints" / step_name(selected)
    recorded = json.loads((checkpoint / "sha256.json").read_text(encoding="utf-8"))
    verify_checkpoint(checkpoint, str(recorded["directory_sha256"]))
    state = json.loads((checkpoint / "training_state.json").read_text(encoding="utf-8"))
    if state.get("stage") != "sft" or int(state.get("global_step", -1)) != selected:
        raise ValueError("training_state does not match the selected step")
    if state.get("eval_decode_sha256") != config["decode_eval_sha256"]:
        raise ValueError("eval decode hash does not match the config")
    manifest_sha = state.get("manifest_sha256")
    if not isinstance(manifest_sha, dict):
        raise ValueError("training_state is missing manifest hashes")
    if semantic_config_sha256(config, manifest_sha) != state.get("config_sha256"):
        raise ValueError("semantic config hash does not match the checkpoint")
    scored_path = run_dir / "eval" / step_name(selected) / "scored.jsonl"
    greedy_fields(read_jsonl(scored_path), "scored.jsonl")
    latest_step = None
    latest_path = run_dir / "latest.json"
    if latest_path.is_file():
        latest = read_latest(latest_path)
        latest_step = int(str(latest["step_name"]).split("_")[1])
    staging = output_dir.with_name(output_dir.name + ".incoming")
    if staging.exists():
        shutil.rmtree(staging)
    try:
        (staging / "adapter").mkdir(parents=True)
        (staging / "source").mkdir(parents=True)
        shutil.copyfile(checkpoint / "adapter" / "adapter.safetensors", staging / "adapter" / "adapter.safetensors")
        shutil.copyfile(checkpoint / "adapter" / "adapter_config.json", staging / "adapter" / "adapter_config.json")
        shutil.copyfile(run_dir / "gate.json", staging / "source" / "gate.json")
        shutil.copyfile(scored_path, staging / "source" / "scored.jsonl")
        shutil.copyfile(checkpoint / "training_state.json", staging / "source" / "training_state.json")
        payload = _write_payload_sha(staging)
        release = {
            "schema_version": RELEASE_SCHEMA,
            "release_id": release_id,
            "status": "staged",
            "publishable": False,
            "stage": "sft",
            "source_run_id": run_dir.name,
            "source_step": selected,
            "source_checkpoint_sha256": str(recorded["directory_sha256"]),
            "latest_step": latest_step,
            "latest_is_release": False,
            "parent_release_sha256": None,
            "base_model": config["model"]["name"],
            "base_revision": config["model"]["revision"],
            "processor_sha256": config["model"]["processor_sha256"],
            "eval_decode_sha256": config["decode_eval_sha256"],
            "config_sha256": state["config_sha256"],
            "val_manifest_sha256": manifest_sha["val"],
            "adapter_sha256": payload["files"]["adapter/adapter.safetensors"],
            "scored_sha256": payload["files"]["source/scored.jsonl"],
            "gate_sha256": payload["files"]["source/gate.json"],
            "payload_sha256": payload["directory_sha256"],
            "created_at": utc_now(),
        }
        atomic_write_json(staging / "release.json", release)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return release


def _run_stamp(run_dir: Path) -> str:
    name = run_dir.name
    marker = "_sft_06b_zh_seed"
    if marker not in name:
        raise ValueError(f"run directory name is not an SFT run id: {name}")
    return name.split(marker)[0]


def load_release(release_dir: Path) -> dict[str, Any]:
    release = json.loads((release_dir / "release.json").read_text(encoding="utf-8"))
    if release.get("schema_version") != RELEASE_SCHEMA:
        raise ValueError("release schema mismatch")
    record = _payload_record(release_dir)
    saved = json.loads((release_dir / "sha256.json").read_text(encoding="utf-8"))
    if record["files"] != saved.get("files") or record["directory_sha256"] != release.get("payload_sha256"):
        raise ValueError("release payload hash does not match the files")
    if record["files"]["adapter/adapter.safetensors"] != release.get("adapter_sha256"):
        raise ValueError("adapter hash does not match release.json")
    return release


def mark_published(release_dir: Path, readback: dict[str, Any]) -> dict[str, Any]:
    """Flip staged to published without changing the adapter payload hash."""
    release = load_release(release_dir)
    if release.get("status") != "staged" or release.get("publishable") is not False:
        raise ValueError("only a staged release can be published")
    if readback.get("matched") is not True or int(readback.get("compared", 0)) != int(readback.get("saved_rows", -1)):
        raise ValueError("refusing to publish a partial readback")
    before = release["payload_sha256"]
    readback_path = release_dir / "readback.json"
    atomic_write_json(readback_path, readback)
    release["status"] = "published"
    release["publishable"] = True
    release["readback_sha256"] = sha256_file(readback_path)
    release["published_at"] = utc_now()
    atomic_write_json(release_dir / "release.json", release)
    if _payload_record(release_dir)["directory_sha256"] != before:
        raise ValueError("publishing changed the adapter payload")
    return release


def fresh_fields(predictions: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    return greedy_fields(predictions, "readback predictions")


def _match_training_runtime(config: dict[str, Any]) -> None:
    """Use the same deterministic flags the training eval set before generate."""
    if config["runtime"]["deterministic_algorithms"]:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from train.train_sft import set_seeds

    if config["runtime"]["deterministic_algorithms"]:
        torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    set_seeds(int(config["seed"]))
    print(
        "SFT_RELEASE_RUNTIME "
        f"deterministic={bool(config['runtime']['deterministic_algorithms'])} "
        f"cublas={os.environ.get('CUBLAS_WORKSPACE_CONFIG', '')}",
        flush=True,
    )


def _readback_progress(done: int, total: int) -> None:
    print(f"SFT_RELEASE_PROGRESS {done}/{total}", flush=True)


def run_readback(config_path: Path, release_dir: Path, manifest_path: Path) -> dict[str, Any]:
    """Decode the val manifest in this process and compare it to the saved text."""
    from evaluation.jsonl_io import read_jsonl as read_manifest
    from train.train_sft import decode_eval_rows, load_adapter, load_sft_model

    config = load_train_config(config_path)
    release = load_release(release_dir)
    if release.get("status") != "staged":
        raise ValueError("readback requires a staged release")
    if config["model"]["revision"] != release["base_revision"]:
        raise ValueError("config revision does not match the release")
    if config["model"]["processor_sha256"] != release["processor_sha256"]:
        raise ValueError("config processor hash does not match the release")
    if config["decode_eval_sha256"] != release["eval_decode_sha256"]:
        raise ValueError("config decode hash does not match the release")
    saved = greedy_fields(read_jsonl(release_dir / "source" / "scored.jsonl"), "saved scored")
    rows = read_manifest(manifest_path)
    if [str(row.get("sample_id")) for row in rows] != list(saved):
        raise ValueError("val manifest sample_id order does not match the saved scored rows")
    _match_training_runtime(config)
    model, processor, _report = load_sft_model(config)
    load_adapter(model, release_dir / "adapter" / "adapter.safetensors")
    if sha256_file(release_dir / "adapter" / "adapter.safetensors") != release["adapter_sha256"]:
        raise ValueError("adapter file hash changed after load")
    predictions = decode_eval_rows(
        model,
        processor,
        rows,
        manifest_path,
        config,
        int(release["source_step"]),
        str(release["adapter_sha256"]),
        on_progress=_readback_progress,
    )
    report = compare_greedy_texts(saved, fresh_fields(predictions))
    report["release_id"] = release["release_id"]
    report["checked_at"] = utc_now()
    if report["matched"] and report["compared"] == report["saved_rows"] == len(rows):
        mark_published(release_dir, report)
        print(f"SFT_RELEASE_PUBLISHED compared={report['compared']}", flush=True)
        return report
    atomic_write_json(release_dir / "readback.json", report)
    print(
        f"SFT_RELEASE_READBACK_MISMATCH compared={report['compared']} mismatches={report['mismatch_count']}",
        flush=True,
    )
    return report


def completed_hypothesis_rows(path: Path, rows: list[dict[str, Any]]) -> int:
    """Return how many leading manifest rows are already in ``path``."""
    if not path.is_file():
        return 0
    done: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        sample_id = row.get("sample_id") if isinstance(row, dict) else None
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{path} line {line_number} is missing sample_id")
        done.append(sample_id)
    expected = [str(row.get("sample_id")) for row in rows[: len(done)]]
    if done != expected:
        raise ValueError("existing hypotheses are not a prefix of the manifest")
    return len(done)


def _sync_hypotheses(source: Path, sync_path: Path | None, done: int, total: int) -> None:
    if sync_path is None:
        return
    sync_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, sync_path)
    if sha256_file(source) != sha256_file(sync_path):
        print(f"SFT_HYP_SYNC_FAIL {done}/{total}", flush=True)
        return
    print(f"SFT_HYP_SYNC {done}/{total}", flush=True)


def run_hypotheses(
    config_path: Path,
    release_dir: Path,
    manifest_path: Path,
    output_path: Path,
    sync_path: Path | None = None,
) -> int:
    """Greedy-decode a manifest with a published SFT release."""
    from evaluation.jsonl_io import read_jsonl as read_manifest
    from train.train_sft import decode_eval_rows, load_adapter, load_sft_model

    config = load_train_config(config_path)
    release = load_release(release_dir)
    if release.get("status") != "published" or release.get("publishable") is not True:
        raise ValueError("hypotheses require a published release")
    if config["model"]["revision"] != release["base_revision"]:
        raise ValueError("config revision does not match the release")
    if config["model"]["processor_sha256"] != release["processor_sha256"]:
        raise ValueError("config processor hash does not match the release")
    if config["decode_eval_sha256"] != release["eval_decode_sha256"]:
        raise ValueError("config decode hash does not match the release")
    rows = read_manifest(manifest_path)
    if not rows:
        raise ValueError("manifest is empty")
    done = completed_hypothesis_rows(output_path, rows)
    print(f"SFT_HYP_RESUME {done}/{len(rows)}", flush=True)
    if done == len(rows):
        _sync_hypotheses(output_path, sync_path, done, len(rows))
        if sync_path is not None and sha256_file(output_path) != sha256_file(sync_path):
            raise ValueError("hypothesis sync hash mismatch")
        print(f"SFT_HYP_DONE rows={done}", flush=True)
        return done
    output_path.parent.mkdir(parents=True, exist_ok=True)
    handle = output_path.open("a" if done else "w", encoding="utf-8")
    written = done

    def on_row(prediction: dict[str, Any]) -> None:
        nonlocal written
        handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
        handle.flush()
        written += 1

    def on_progress(finished: int, _batch_total: int) -> None:
        absolute = done + finished
        print(f"SFT_HYP_PROGRESS {absolute}/{len(rows)}", flush=True)
        if absolute % 2500 == 0 or absolute == len(rows):
            _sync_hypotheses(output_path, sync_path, absolute, len(rows))

    try:
        _match_training_runtime(config)
        model, processor, _report = load_sft_model(config)
        adapter_path = release_dir / "adapter" / "adapter.safetensors"
        load_adapter(model, adapter_path)
        if sha256_file(adapter_path) != release["adapter_sha256"]:
            raise ValueError("adapter file hash changed after load")
        decode_eval_rows(
            model,
            processor,
            rows[done:],
            manifest_path,
            config,
            int(release["source_step"]),
            str(release["adapter_sha256"]),
            on_progress=on_progress,
            on_row=on_row,
        )
    finally:
        handle.close()
    if written != len(rows):
        raise ValueError(f"hypothesis row count is {written}, expected {len(rows)}")
    _sync_hypotheses(output_path, sync_path, written, len(rows))
    if sync_path is not None and sha256_file(output_path) != sha256_file(sync_path):
        raise ValueError("hypothesis sync hash mismatch")
    print(f"SFT_HYP_DONE rows={written}", flush=True)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="准备或读回 SFT release")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--config", required=True, type=Path)
    prepare.add_argument("--run-dir", required=True, type=Path)
    prepare.add_argument("--output-dir", required=True, type=Path)
    prepare.add_argument("--release-id", required=True)
    readback = sub.add_parser("readback")
    readback.add_argument("--config", required=True, type=Path)
    readback.add_argument("--release-dir", required=True, type=Path)
    readback.add_argument("--manifest", required=True, type=Path)
    hypotheses = sub.add_parser("hypotheses")
    hypotheses.add_argument("--config", required=True, type=Path)
    hypotheses.add_argument("--release-dir", required=True, type=Path)
    hypotheses.add_argument("--manifest", required=True, type=Path)
    hypotheses.add_argument("--output", required=True, type=Path)
    hypotheses.add_argument("--sync-path", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    from train.train_sft import RunStop

    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            release = prepare_release(
                config_path=args.config,
                run_dir=args.run_dir,
                output_dir=args.output_dir,
                release_id=args.release_id,
            )
            print(f"SFT_RELEASE_STAGED step={release['source_step']} id={release['release_id']}", flush=True)
            return 0
        if args.command == "hypotheses":
            run_hypotheses(
                config_path=args.config,
                release_dir=args.release_dir,
                manifest_path=args.manifest,
                output_path=args.output,
                sync_path=args.sync_path,
            )
            return 0
        report = run_readback(args.config, args.release_dir, args.manifest)
    except (OSError, ValueError, json.JSONDecodeError, RunStop) as exc:
        label = "SFT_HYP_FAIL" if getattr(args, "command", "") == "hypotheses" else "SFT_RELEASE_FAIL"
        print(f"{label} {exc}", file=sys.stderr)
        return 2
    return 0 if report["matched"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
