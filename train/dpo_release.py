#!/usr/bin/env python3
"""Freeze the gate-selected DPO adapter and read its greedy text back.

``latest.json`` stays a resume pointer. This tool only releases the step
selected in ``gate.json`` after checking ``eval_decisions.jsonl`` and
``preference.json``. ``prepare`` writes ``staged`` (``publishable=False``).
``readback`` sets ``published`` (``publishable=True``) only when every saved
val row matches in a fresh process, while preserving ``audit_disposition`` as
``unlistened`` and leaving the source run's ``gate.json`` untouched.
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

from train.dpo_schema import load_dpo_config, semantic_dpo_config_sha256  # noqa: E402
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
from train.sft_release import (  # noqa: E402
    decision_for_step,
    fresh_fields,
    greedy_fields,
    read_jsonl,
)

DPO_RELEASE_SCHEMA = "qwen3-asr-dpo-release-v1"
DPO_RELEASE_SHA_SCHEMA = "qwen3-asr-dpo-release-sha-v1"
DPO_READBACK_SCHEMA = "qwen3-asr-dpo-release-readback-v1"
DPO_PAYLOAD_FILES = (
    "adapter/adapter.safetensors",
    "adapter/adapter_config.json",
    "source/gate.json",
    "source/preference.json",
    "source/scored.jsonl",
    "source/training_state.json",
)


def _run_stamp(run_dir: Path) -> str:
    name = run_dir.name
    marker = "_dpo_06b_zh_seed"
    if marker not in name:
        raise ValueError(f"run directory name is not a DPO run id: {name}")
    return name.split(marker)[0]


def require_passed_dpo_decision(decision: dict[str, Any], step: int, min_pref: float) -> None:
    if decision.get("passed") is not True or decision.get("risk_stop") is not False:
        raise ValueError(f"step {step} is not a passed DPO checkpoint")
    pref = float(decision.get("preference_accuracy", 0.0))
    if pref < min_pref or pref <= 0.5:
        raise ValueError(f"step {step} preference_accuracy {pref} is below the DPO gate")


def compare_dpo_greedy_texts(
    saved: dict[str, dict[str, str]],
    fresh: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Return a DPO readback report. One differing row keeps matched false."""
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
        "schema_version": DPO_READBACK_SCHEMA,
        "matched": not mismatches,
        "compared": len(saved_ids & fresh_ids),
        "saved_rows": len(saved),
        "fresh_rows": len(fresh),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:20],
    }


def _payload_record(root: Path) -> dict[str, Any]:
    mapping = file_map(root, list(DPO_PAYLOAD_FILES))
    return {
        "schema_version": DPO_RELEASE_SHA_SCHEMA,
        "files": mapping,
        "directory_sha256": directory_sha256(mapping),
    }


def _write_payload_sha(root: Path) -> dict[str, Any]:
    record = _payload_record(root)
    atomic_write_json(root / "sha256.json", record)
    return record


def sync_release_dir(release_dir: Path, sync_dir: Path | None) -> str:
    """Copy a staged or published DPO release to Drive and verify readback hashes."""
    if sync_dir is None:
        return "not_requested"
    incoming = sync_dir.with_name(sync_dir.name + ".incoming")
    if incoming.exists():
        shutil.rmtree(incoming)
    incoming.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(release_dir, incoming)
    loaded = load_dpo_release(incoming)
    if loaded.get("payload_sha256") != load_dpo_release(release_dir).get("payload_sha256"):
        shutil.rmtree(incoming, ignore_errors=True)
        raise ValueError("Drive incoming DPO release payload hash mismatch")
    if sync_dir.exists():
        shutil.rmtree(sync_dir)
    incoming.rename(sync_dir)
    verified = load_dpo_release(sync_dir)
    if verified != load_dpo_release(release_dir):
        raise ValueError("Drive DPO release readback metadata mismatch")
    if (release_dir / "readback.json").is_file():
        if sha256_file(release_dir / "readback.json") != sha256_file(sync_dir / "readback.json"):
            raise ValueError("Drive DPO release readback.json hash mismatch")
    return "synced"


def prepare_dpo_release(
    *,
    config_path: Path,
    run_dir: Path,
    output_dir: Path,
    release_id: str,
    sync_dir: Path | None = None,
) -> dict[str, Any]:
    """Copy the selected DPO adapter. Refuse a second write or a failed gate."""
    if output_dir.exists():
        raise ValueError(f"release directory already exists: {output_dir}")
    config = load_dpo_config(config_path)
    if config["profile"] != "pilot":
        raise ValueError("only a pilot DPO profile can be released")
    if config["model"]["parent_release"] is None or config["model"]["parent_release_sha256"] is None:
        raise ValueError("DPO release requires a pinned SFT parent release")
    gate = json.loads((run_dir / "gate.json").read_text(encoding="utf-8"))
    if gate.get("status") != "passed":
        raise ValueError("gate status is not passed")
    selected = gate.get("selected_step")
    if not isinstance(selected, int) or selected <= 0:
        raise ValueError("gate is missing selected_step")
    expected_id = f"dpo_06b_zh_{_run_stamp(run_dir)}_step{selected:06d}"
    if release_id != expected_id:
        raise ValueError(f"release id does not match the selected step: {release_id} != {expected_id}")
    min_pref = float(config["gate"]["min_preference_accuracy"])
    decisions = read_jsonl(run_dir / "eval_decisions.jsonl")
    decision = decision_for_step(decisions, selected)
    require_passed_dpo_decision(decision, selected, min_pref)
    embedded = gate.get("decisions")
    if isinstance(embedded, list) and embedded:
        require_passed_dpo_decision(decision_for_step(embedded, selected), selected, min_pref)
    checkpoint = run_dir / "checkpoints" / step_name(selected)
    recorded = json.loads((checkpoint / "sha256.json").read_text(encoding="utf-8"))
    verify_checkpoint(checkpoint, str(recorded["directory_sha256"]))
    state = json.loads((checkpoint / "training_state.json").read_text(encoding="utf-8"))
    if state.get("stage") != "dpo" or int(state.get("global_step", -1)) != selected:
        raise ValueError("training_state does not match the selected DPO step")
    if state.get("eval_decode_sha256") != config["decode_eval_sha256"]:
        raise ValueError("eval decode hash does not match the config")
    if state.get("parent_release_sha256") != config["model"]["parent_release_sha256"]:
        raise ValueError("parent_release_sha256 does not match the config")
    manifest_sha = state.get("manifest_sha256")
    if not isinstance(manifest_sha, dict):
        raise ValueError("training_state is missing manifest hashes")
    if semantic_dpo_config_sha256(config, manifest_sha) != state.get("config_sha256"):
        raise ValueError("semantic DPO config hash does not match the checkpoint")
    scored_path = run_dir / "eval" / step_name(selected) / "scored.jsonl"
    greedy_fields(read_jsonl(scored_path), "scored.jsonl")
    pref_path = run_dir / "eval" / step_name(selected) / "preference.json"
    if pref_path.is_file():
        preference = json.loads(pref_path.read_text(encoding="utf-8"))
        pref_acc = float(preference.get("accuracy", preference.get("preference_accuracy", 0.0)))
    elif "preference_accuracy" in decision and isinstance(decision.get("length_buckets"), dict):
        preference = {
            "schema_version": "qwen3-asr-dpo-preference-v1",
            "step": selected,
            "accuracy": float(decision["preference_accuracy"]),
            "preference_accuracy": float(decision["preference_accuracy"]),
            "ties": int(decision.get("preference_ties", 0)),
            "raw_policy_accuracy": float(decision.get("raw_policy_accuracy", 0.0)),
            "length_buckets": decision["length_buckets"],
        }
        pref_acc = float(preference["accuracy"])
    else:
        raise ValueError(f"missing preference.json for step {selected}")
    if pref_acc < min_pref or pref_acc <= 0.5:
        raise ValueError(f"preference.json accuracy {pref_acc} is below the DPO gate")
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
        if pref_path.is_file():
            shutil.copyfile(pref_path, staging / "source" / "preference.json")
        else:
            atomic_write_json(staging / "source" / "preference.json", preference)
        shutil.copyfile(scored_path, staging / "source" / "scored.jsonl")
        shutil.copyfile(checkpoint / "training_state.json", staging / "source" / "training_state.json")
        payload = _write_payload_sha(staging)
        release = {
            "schema_version": DPO_RELEASE_SCHEMA,
            "release_id": release_id,
            "status": "staged",
            "publishable": False,
            "stage": "dpo",
            "source_run_id": run_dir.name,
            "source_step": selected,
            "source_checkpoint_sha256": str(recorded["directory_sha256"]),
            "latest_step": latest_step,
            "latest_is_release": bool(latest_step == selected),
            "parent_release": config["model"]["parent_release"],
            "parent_release_sha256": config["model"]["parent_release_sha256"],
            "listen_waiver": config["dpo"]["listen_waiver"],
            "audit_disposition": "unlistened",
            "preference_accuracy": pref_acc,
            "base_model": config["model"]["name"],
            "base_revision": config["model"]["revision"],
            "processor_sha256": config["model"]["processor_sha256"],
            "eval_decode_sha256": config["decode_eval_sha256"],
            "config_sha256": state["config_sha256"],
            "val_manifest_sha256": manifest_sha["val"],
            "adapter_sha256": payload["files"]["adapter/adapter.safetensors"],
            "scored_sha256": payload["files"]["source/scored.jsonl"],
            "preference_sha256": payload["files"]["source/preference.json"],
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
    sync_release_dir(output_dir, sync_dir)
    return release


def load_dpo_release(release_dir: Path) -> dict[str, Any]:
    release = json.loads((release_dir / "release.json").read_text(encoding="utf-8"))
    if release.get("schema_version") != DPO_RELEASE_SCHEMA:
        raise ValueError("DPO release schema mismatch")
    if release.get("stage") != "dpo":
        raise ValueError("release stage is not dpo")
    if release.get("audit_disposition") != "unlistened":
        raise ValueError("DPO release audit_disposition must stay unlistened")
    record = _payload_record(release_dir)
    saved = json.loads((release_dir / "sha256.json").read_text(encoding="utf-8"))
    if record["files"] != saved.get("files") or record["directory_sha256"] != release.get("payload_sha256"):
        raise ValueError("DPO release payload hash does not match the files")
    if record["files"]["adapter/adapter.safetensors"] != release.get("adapter_sha256"):
        raise ValueError("adapter hash does not match release.json")
    return release


def load_published_dpo_release(release_dir: Path) -> dict[str, Any]:
    release = load_dpo_release(release_dir)
    if release.get("status") != "published" or release.get("publishable") is not True:
        raise ValueError("RL requires a published DPO release")
    readback_path = release_dir / "readback.json"
    if not readback_path.is_file():
        raise ValueError("published DPO release is missing readback.json")
    if sha256_file(readback_path) != release.get("readback_sha256"):
        raise ValueError("DPO release readback_sha256 mismatch")
    readback = json.loads(readback_path.read_text(encoding="utf-8"))
    if readback.get("matched") is not True or int(readback.get("mismatch_count", -1)) != 0:
        raise ValueError("DPO release readback is not matched")
    return release


def mark_dpo_published(release_dir: Path, readback: dict[str, Any]) -> dict[str, Any]:
    """Flip staged to published without changing the adapter payload hash."""
    release = load_dpo_release(release_dir)
    if release.get("status") != "staged" or release.get("publishable") is not False:
        raise ValueError("only a staged DPO release can be published")
    if readback.get("matched") is not True or int(readback.get("compared", 0)) != int(readback.get("saved_rows", -1)):
        raise ValueError("refusing to publish a partial DPO readback")
    before = release["payload_sha256"]
    readback_path = release_dir / "readback.json"
    atomic_write_json(readback_path, readback)
    release["status"] = "published"
    release["publishable"] = True
    release["audit_disposition"] = "unlistened"
    release["readback_sha256"] = sha256_file(readback_path)
    release["published_at"] = utc_now()
    atomic_write_json(release_dir / "release.json", release)
    if _payload_record(release_dir)["directory_sha256"] != before:
        raise ValueError("publishing changed the DPO adapter payload")
    return release


def _match_training_runtime(config: dict[str, Any]) -> None:
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
        "DPO_RELEASE_RUNTIME "
        f"deterministic={bool(config['runtime']['deterministic_algorithms'])} "
        f"cublas={os.environ.get('CUBLAS_WORKSPACE_CONFIG', '')}",
        flush=True,
    )


def _readback_progress(done: int, total: int) -> None:
    print(f"DPO_RELEASE_PROGRESS {done}/{total}", flush=True)


def run_dpo_readback(
    config_path: Path,
    release_dir: Path,
    manifest_path: Path,
    sync_dir: Path | None = None,
) -> dict[str, Any]:
    """Decode the val manifest in a fresh process and compare it to the saved DPO text."""
    from evaluation.jsonl_io import read_jsonl as read_manifest
    from train.train_sft import decode_eval_rows, load_adapter, load_sft_model

    config = load_dpo_config(config_path)
    release = load_dpo_release(release_dir)
    if release.get("status") != "staged":
        raise ValueError("readback requires a staged DPO release")
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
    adapter_path = release_dir / "adapter" / "adapter.safetensors"
    load_adapter(model, adapter_path)
    if sha256_file(adapter_path) != release["adapter_sha256"]:
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
    report = compare_dpo_greedy_texts(saved, fresh_fields(predictions))
    report["release_id"] = release["release_id"]
    report["checked_at"] = utc_now()
    if report["matched"] and report["compared"] == report["saved_rows"] == len(rows):
        mark_dpo_published(release_dir, report)
        sync_release_dir(release_dir, sync_dir)
        print(f"DPO_RELEASE_PUBLISHED compared={report['compared']}", flush=True)
        return report
    atomic_write_json(release_dir / "readback.json", report)
    print(
        f"DPO_RELEASE_READBACK_MISMATCH compared={report['compared']} mismatches={report['mismatch_count']}",
        flush=True,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="准备或读回 DPO release")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--config", required=True, type=Path)
    prepare.add_argument("--run-dir", required=True, type=Path)
    prepare.add_argument("--output-dir", required=True, type=Path)
    prepare.add_argument("--release-id", required=True)
    prepare.add_argument("--sync-dir", type=Path)
    readback = sub.add_parser("readback")
    readback.add_argument("--config", required=True, type=Path)
    readback.add_argument("--release-dir", required=True, type=Path)
    readback.add_argument("--manifest", required=True, type=Path)
    readback.add_argument("--sync-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    from train.train_sft import RunStop

    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            release = prepare_dpo_release(
                config_path=args.config,
                run_dir=args.run_dir,
                output_dir=args.output_dir,
                release_id=args.release_id,
                sync_dir=args.sync_dir,
            )
            print(f"DPO_RELEASE_STAGED step={release['source_step']} id={release['release_id']}", flush=True)
            return 0
        report = run_dpo_readback(args.config, args.release_dir, args.manifest, sync_dir=args.sync_dir)
    except (OSError, ValueError, json.JSONDecodeError, RunStop) as exc:
        print(f"DPO_RELEASE_FAIL {exc}", file=sys.stderr)
        return 2
    return 0 if report["matched"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
