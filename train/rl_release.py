#!/usr/bin/env python3
"""Freeze the gate-selected RL adapter and read its greedy text back.

``latest.json`` stays a resume pointer. This tool only releases the step
selected in ``gate.json`` after checking ``eval_decisions.jsonl`` for
``passed=True``, ``risk_stop=False``, ``reward_delta_vs_step0 >= min_reward_improvement``,
``raw_kl <= max_raw_kl``, and ``reference_matched=True``. ``prepare`` writes
``staged`` (``publishable=False``). ``readback`` sets ``published``
(``publishable=True``) only when every saved val row matches in a fresh
process, while preserving ``audit_disposition`` as ``unlistened`` and leaving
the source run's ``gate.json`` untouched.
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

from train.rl_schema import load_rl_config, semantic_rl_config_sha256  # noqa: E402
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

RL_RELEASE_SCHEMA = "qwen3-asr-rl-release-v1"
RL_RELEASE_SHA_SCHEMA = "qwen3-asr-rl-release-sha-v1"
RL_READBACK_SCHEMA = "qwen3-asr-rl-release-readback-v1"
RL_PAYLOAD_FILES = (
    "adapter/adapter.safetensors",
    "adapter/adapter_config.json",
    "source/gate.json",
    "source/decision.json",
    "source/scored.jsonl",
    "source/training_state.json",
)


def _run_stamp(run_dir: Path) -> str:
    name = run_dir.name
    marker = "_rl_06b_zh_seed"
    if marker not in name:
        raise ValueError(f"run directory name is not an RL run id: {name}")
    return name.split(marker)[0]


def require_passed_rl_decision(
    decision: dict[str, Any],
    step: int,
    min_reward_imp: float,
    max_raw_kl: float,
) -> None:
    if decision.get("passed") is not True or decision.get("risk_stop") is not False:
        raise ValueError(f"step {step} is not a passed RL checkpoint")
    if decision.get("reference_matched") is not True:
        raise ValueError(f"step {step} reference_matched is not True")
    reward_delta = float(decision.get("reward_delta_vs_step0", -1.0))
    if reward_delta < min_reward_imp:
        raise ValueError(f"step {step} reward_delta_vs_step0 {reward_delta} is below {min_reward_imp}")
    raw_kl = float(decision.get("raw_kl", 1.0))
    if raw_kl > max_raw_kl:
        raise ValueError(f"step {step} raw_kl {raw_kl} exceeds {max_raw_kl}")


def compare_rl_greedy_texts(
    saved: dict[str, dict[str, str]],
    fresh: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Return an RL readback report. One differing row keeps matched false."""
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
        "schema_version": RL_READBACK_SCHEMA,
        "matched": not mismatches,
        "compared": len(saved_ids & fresh_ids),
        "saved_rows": len(saved),
        "fresh_rows": len(fresh),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:20],
    }


def _payload_record(root: Path) -> dict[str, Any]:
    mapping = file_map(root, list(RL_PAYLOAD_FILES))
    return {
        "schema_version": RL_RELEASE_SHA_SCHEMA,
        "files": mapping,
        "directory_sha256": directory_sha256(mapping),
    }


def _write_payload_sha(root: Path) -> dict[str, Any]:
    record = _payload_record(root)
    atomic_write_json(root / "sha256.json", record)
    return record


def sync_release_dir(release_dir: Path, sync_dir: Path | None) -> str:
    """Copy a staged or published RL release to Drive and verify readback hashes."""
    if sync_dir is None:
        return "not_requested"
    incoming = sync_dir.with_name(sync_dir.name + ".incoming")
    if incoming.exists():
        shutil.rmtree(incoming)
    incoming.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(release_dir, incoming)
    loaded = load_rl_release(incoming)
    if loaded.get("payload_sha256") != load_rl_release(release_dir).get("payload_sha256"):
        shutil.rmtree(incoming, ignore_errors=True)
        raise ValueError("Drive incoming RL release payload hash mismatch")
    if sync_dir.exists():
        shutil.rmtree(sync_dir)
    incoming.rename(sync_dir)
    verified = load_rl_release(sync_dir)
    if verified != load_rl_release(release_dir):
        raise ValueError("Drive RL release readback metadata mismatch")
    if (release_dir / "readback.json").is_file():
        if sha256_file(release_dir / "readback.json") != sha256_file(sync_dir / "readback.json"):
            raise ValueError("Drive RL release readback.json hash mismatch")
    return "synced"


def prepare_rl_release(
    *,
    config_path: Path,
    run_dir: Path,
    output_dir: Path,
    release_id: str,
    sync_dir: Path | None = None,
) -> dict[str, Any]:
    """Copy the selected RL adapter. Refuse a second write or a failed gate."""
    if output_dir.exists():
        raise ValueError(f"release directory already exists: {output_dir}")
    config = load_rl_config(config_path)
    if config["profile"] != "pilot":
        raise ValueError("only a pilot RL profile can be released")
    if config["model"]["parent_release"] is None or config["model"]["parent_release_sha256"] is None:
        raise ValueError("RL release requires a pinned DPO parent release")
    gate = json.loads((run_dir / "gate.json").read_text(encoding="utf-8"))
    if gate.get("status") != "passed":
        raise ValueError("gate status is not passed")
    selected = gate.get("selected_step")
    if not isinstance(selected, int) or selected <= 0:
        raise ValueError("gate is missing selected_step")
    expected_id = f"rl_06b_zh_{_run_stamp(run_dir)}_step{selected:06d}"
    if release_id != expected_id:
        raise ValueError(f"release id does not match the selected step: {release_id} != {expected_id}")
    min_reward_imp = float(config["gate"]["min_reward_improvement"])
    max_raw_kl = float(config["gate"]["raw_kl_max"])
    decisions = read_jsonl(run_dir / "eval_decisions.jsonl")
    decision = decision_for_step(decisions, selected)
    require_passed_rl_decision(decision, selected, min_reward_imp, max_raw_kl)
    embedded = gate.get("decisions")
    if isinstance(embedded, list) and embedded:
        require_passed_rl_decision(decision_for_step(embedded, selected), selected, min_reward_imp, max_raw_kl)
    checkpoint = run_dir / "checkpoints" / step_name(selected)
    recorded = json.loads((checkpoint / "sha256.json").read_text(encoding="utf-8"))
    verify_checkpoint(checkpoint, str(recorded["directory_sha256"]))
    state = json.loads((checkpoint / "training_state.json").read_text(encoding="utf-8"))
    if state.get("stage") != "rl" or int(state.get("global_step", -1)) != selected:
        raise ValueError("training_state does not match the selected RL step")
    if state.get("eval_decode_sha256") != config["decode_eval_sha256"]:
        raise ValueError("eval decode hash does not match the config")
    if state.get("parent_release_sha256") != config["model"]["parent_release_sha256"]:
        raise ValueError("parent_release_sha256 does not match the config")
    manifest_sha = state.get("manifest_sha256")
    if not isinstance(manifest_sha, dict):
        raise ValueError("training_state is missing manifest hashes")
    if semantic_rl_config_sha256(config, manifest_sha) != state.get("config_sha256"):
        raise ValueError("semantic RL config hash does not match the checkpoint")
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
        atomic_write_json(staging / "source" / "decision.json", decision)
        shutil.copyfile(scored_path, staging / "source" / "scored.jsonl")
        shutil.copyfile(checkpoint / "training_state.json", staging / "source" / "training_state.json")
        payload = _write_payload_sha(staging)
        release = {
            "schema_version": RL_RELEASE_SCHEMA,
            "release_id": release_id,
            "status": "staged",
            "publishable": False,
            "stage": "rl",
            "source_run_id": run_dir.name,
            "source_step": selected,
            "source_checkpoint_sha256": str(recorded["directory_sha256"]),
            "latest_step": latest_step,
            "latest_is_release": bool(latest_step == selected),
            "parent_release": config["model"]["parent_release"],
            "parent_release_sha256": config["model"]["parent_release_sha256"],
            "audit_disposition": "unlistened",
            "mean_greedy_reward": float(decision["mean_greedy_reward"]),
            "reward_delta_vs_step0": float(decision["reward_delta_vs_step0"]),
            "raw_kl": float(decision["raw_kl"]),
            "reference_sha256": str(state.get("reference_sha256", "")),
            "base_model": config["model"]["name"],
            "base_revision": config["model"]["revision"],
            "processor_sha256": config["model"]["processor_sha256"],
            "eval_decode_sha256": config["decode_eval_sha256"],
            "config_sha256": state["config_sha256"],
            "val_manifest_sha256": manifest_sha["val"],
            "adapter_sha256": payload["files"]["adapter/adapter.safetensors"],
            "scored_sha256": payload["files"]["source/scored.jsonl"],
            "decision_sha256": payload["files"]["source/decision.json"],
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


def load_rl_release(release_dir: Path) -> dict[str, Any]:
    release = json.loads((release_dir / "release.json").read_text(encoding="utf-8"))
    if release.get("schema_version") != RL_RELEASE_SCHEMA:
        raise ValueError("RL release schema mismatch")
    if release.get("stage") != "rl":
        raise ValueError("release stage is not rl")
    if release.get("audit_disposition") != "unlistened":
        raise ValueError("RL release audit_disposition must stay unlistened")
    record = _payload_record(release_dir)
    saved = json.loads((release_dir / "sha256.json").read_text(encoding="utf-8"))
    if record["files"] != saved.get("files") or record["directory_sha256"] != release.get("payload_sha256"):
        raise ValueError("RL release payload hash does not match the files")
    if record["files"]["adapter/adapter.safetensors"] != release.get("adapter_sha256"):
        raise ValueError("adapter hash does not match release.json")
    return release


def load_published_rl_release(release_dir: Path) -> dict[str, Any]:
    release = load_rl_release(release_dir)
    if release.get("status") != "published" or release.get("publishable") is not True:
        raise ValueError("requires a published RL release")
    readback_path = release_dir / "readback.json"
    if not readback_path.is_file():
        raise ValueError("published RL release is missing readback.json")
    if sha256_file(readback_path) != release.get("readback_sha256"):
        raise ValueError("RL release readback_sha256 mismatch")
    readback = json.loads(readback_path.read_text(encoding="utf-8"))
    if readback.get("matched") is not True or int(readback.get("mismatch_count", -1)) != 0:
        raise ValueError("RL release readback is not matched")
    return release


def mark_rl_published(release_dir: Path, readback: dict[str, Any]) -> dict[str, Any]:
    """Flip staged to published without changing the adapter payload hash."""
    release = load_rl_release(release_dir)
    if release.get("status") != "staged" or release.get("publishable") is not False:
        raise ValueError("only a staged RL release can be published")
    if readback.get("matched") is not True or int(readback.get("compared", 0)) != int(readback.get("saved_rows", -1)):
        raise ValueError("refusing to publish a partial RL readback")
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
        raise ValueError("publishing changed the RL adapter payload")
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
        "RL_RELEASE_RUNTIME "
        f"deterministic={bool(config['runtime']['deterministic_algorithms'])} "
        f"cublas={os.environ.get('CUBLAS_WORKSPACE_CONFIG', '')}",
        flush=True,
    )


def _readback_progress(done: int, total: int) -> None:
    print(f"RL_RELEASE_PROGRESS {done}/{total}", flush=True)


def run_rl_readback(
    config_path: Path,
    release_dir: Path,
    manifest_path: Path,
    sync_dir: Path | None = None,
) -> dict[str, Any]:
    """Decode the val manifest in a fresh process and compare it to the saved RL text."""
    from evaluation.jsonl_io import read_jsonl as read_manifest
    from train.train_sft import decode_eval_rows, load_adapter, load_sft_model

    config = load_rl_config(config_path)
    release = load_rl_release(release_dir)
    if release.get("status") != "staged":
        raise ValueError("readback requires a staged RL release")
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
    report = compare_rl_greedy_texts(saved, fresh_fields(predictions))
    report["release_id"] = release["release_id"]
    report["checked_at"] = utc_now()
    if report["matched"] and report["compared"] == report["saved_rows"] == len(rows):
        mark_rl_published(release_dir, report)
        sync_release_dir(release_dir, sync_dir)
        print(f"RL_RELEASE_PUBLISHED compared={report['compared']}", flush=True)
        return report
    atomic_write_json(release_dir / "readback.json", report)
    print(
        f"RL_RELEASE_READBACK_MISMATCH compared={report['compared']} mismatches={report['mismatch_count']}",
        flush=True,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="准备或读回 RL release")
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
            release = prepare_rl_release(
                config_path=args.config,
                run_dir=args.run_dir,
                output_dir=args.output_dir,
                release_id=args.release_id,
                sync_dir=args.sync_dir,
            )
            print(f"RL_RELEASE_STAGED step={release['source_step']} id={release['release_id']}", flush=True)
            return 0
        report = run_rl_readback(args.config, args.release_dir, args.manifest, sync_dir=args.sync_dir)
    except (OSError, ValueError, json.JSONDecodeError, RunStop) as exc:
        print(f"RL_RELEASE_FAIL {exc}", file=sys.stderr)
        return 2
    return 0 if report["matched"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
