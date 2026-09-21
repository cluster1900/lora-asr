#!/usr/bin/env python3
"""Run multi-GPU parallel sharded inference using run_inference.py."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to input JSONL manifest.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output JSONL predictions.",
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3-ASR-1.7B",
        help="Model identifier or local checkpoint path.",
    )
    parser.add_argument(
        "--revision",
        default="7278e1e70fe206f11671096ffdd38061171dd6e5",
        help="Model commit SHA/revision.",
    )
    parser.add_argument(
        "--adapter-dir",
        default=None,
        help="Path to LoRA adapter directory (optional).",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="List of GPU indices to use (e.g. 0 1 2 3).",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Model compute dtype (default: float16).",
    )
    parser.add_argument(
        "--attention",
        default="eager",
        help="Attention implementation (default: eager).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size per GPU (default: 1).",
    )
    parser.add_argument(
        "--method",
        default=None,
        help="Inference method tag for prediction records (e.g. base, sft_merged, dpo_merged).",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Automatically run evaluation/eval_wer.py on output predictions.",
    )
    parser.add_argument(
        "--temp-dir",
        default=None,
        help="Directory to store shard manifests and outputs (default: temporary dir alongside output).",
    )
    parser.add_argument(
        "--keep-shards",
        action="store_true",
        help="Do not delete temporary shard manifests and outputs upon completion.",
    )
    return parser.parse_args(argv)


def split_manifest(manifest_path: Path, num_shards: int) -> List[List[Dict[str, Any]]]:
    """Load manifest and split items evenly into num_shards."""
    items: List[Dict[str, Any]] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, 1):
            line_s = line.strip()
            if not line_s:
                continue
            item = json.loads(line_s)
            items.append(item)

    if not items:
        raise ValueError(f"Manifest {manifest_path} is empty.")

    total = len(items)
    shards: List[List[Dict[str, Any]]] = [[] for _ in range(num_shards)]
    for i, item in enumerate(items):
        shards[i % num_shards].append(item)

    return shards


def run_parallel_inference(args: argparse.Namespace) -> Path:
    manifest_path = Path(args.manifest).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    # Read all original sample IDs in order
    original_sample_ids: List[str] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                original_sample_ids.append(item["sample_id"])

    total_samples = len(original_sample_ids)
    gpus: List[int] = args.gpus
    num_gpus = len(gpus)
    print(f"Starting parallel inference on {total_samples} samples across {num_gpus} GPUs: {gpus}...")

    # Temp directory for shards
    if args.temp_dir:
        temp_dir = Path(args.temp_dir).resolve()
    else:
        temp_dir = output_path.parent / f".shards_{output_path.stem}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    shards = split_manifest(manifest_path, num_gpus)
    shard_manifest_paths: List[Path] = []
    shard_output_paths: List[Path] = []

    for i, (gpu, shard_items) in enumerate(zip(gpus, shards)):
        s_manifest = temp_dir / f"shard_{i}_gpu{gpu}_manifest.jsonl"
        s_output = temp_dir / f"shard_{i}_gpu{gpu}_predictions.jsonl"
        if not s_manifest.is_file():
            with open(s_manifest, "w", encoding="utf-8") as f:
                for item in shard_items:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        shard_manifest_paths.append(s_manifest)
        shard_output_paths.append(s_output)

    # Launch subprocesses
    processes: List[subprocess.Popen[Any]] = []
    log_files: List[Any] = []
    run_inference_script = REPO_ROOT / "inference" / "run_inference.py"

    start_time = time.time()
    try:
        for i, gpu in enumerate(gpus):
            cmd = [
                sys.executable,
                str(run_inference_script),
                "--manifest", str(shard_manifest_paths[i]),
                "--output", str(shard_output_paths[i]),
                "--model-id", str(args.model_id),
                "--revision", str(args.revision),
                "--device", "cuda:0",  # Since CUDA_VISIBLE_DEVICES isolates the card
                "--dtype", str(args.dtype),
                "--attention", str(args.attention),
                "--batch-size", str(args.batch_size),
            ]
            if args.adapter_dir:
                cmd.extend(["--adapter-dir", str(args.adapter_dir)])
            if args.method:
                cmd.extend(["--method", str(args.method)])

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)

            shard_log = shard_output_paths[i].with_suffix(".log")
            log_f = open(shard_log, "a", encoding="utf-8")
            log_files.append(log_f)

            p = subprocess.Popen(
                cmd,
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
            )
            processes.append(p)
            print(f"[GPU {gpu}] Worker launched (PID {p.pid}) with {len(shards[i])} samples (log: {shard_log.name}).")

        # Wait for all processes to complete
        failed = False
        while True:
            all_done = True
            for i, p in enumerate(processes):
                ret = p.poll()
                if ret is None:
                    all_done = False
                elif ret != 0:
                    failed = True
                    print(f"[GPU {gpus[i]}] FAILED with returncode {ret}")
            if all_done or failed:
                break
            time.sleep(2)

        if failed:
            for p in processes:
                if p.poll() is None:
                    p.terminate()
            raise RuntimeError("One or more inference workers failed. See logs above.")

        # Read back trailing output from log files
        for i, p in enumerate(processes):
            p.wait()
            shard_log = shard_output_paths[i].with_suffix(".log")
            last_line = ""
            if shard_log.is_file():
                try:
                    lines = shard_log.read_text(encoding="utf-8", errors="ignore").splitlines()
                    last_line = lines[-1] if lines else ""
                except Exception:
                    pass
            print(f"[GPU {gpus[i]}] Finished: {last_line}")

    finally:
        for lf in log_files:
            try:
                lf.close()
            except Exception:
                pass

    elapsed = time.time() - start_time
    print(f"All {num_gpus} workers finished in {elapsed:.1f}s ({total_samples / max(0.1, elapsed):.2f} samples/s). Merging outputs...")

    # Load all predictions indexed by sample_id
    predictions_by_id: Dict[str, str] = {}
    for s_out in shard_output_paths:
        if not s_out.is_file():
            raise FileNotFoundError(f"Missing expected shard output: {s_out}")
        with open(s_out, "r", encoding="utf-8") as f:
            for line in f:
                line_s = line.strip()
                if line_s:
                    rec = json.loads(line_s)
                    predictions_by_id[rec["sample_id"]] = line_s

    # Verify coverage
    missing = [sid for sid in original_sample_ids if sid not in predictions_by_id]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} sample predictions in shard outputs (e.g. {missing[:3]})")

    # Write merged predictions in strict original manifest order
    with open(output_path, "w", encoding="utf-8") as f:
        for sid in original_sample_ids:
            f.write(predictions_by_id[sid] + "\n")

    print(f"Successfully merged {len(original_sample_ids)} predictions to {output_path}")

    # Optional evaluation
    if args.eval:
        eval_script = REPO_ROOT / "evaluation" / "eval_wer.py"
        eval_output_dir = output_path.parent / f"{output_path.stem}_eval"
        eval_cmd = [
            sys.executable,
            str(eval_script),
            "--predictions", str(output_path),
            "--output-dir", str(eval_output_dir),
        ]
        print(f"Running evaluation: {' '.join(eval_cmd)}")
        subprocess.run(eval_cmd, check=True)

    # Clean up shards
    if not args.keep_shards:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return output_path


def main() -> None:
    args = parse_args()
    run_parallel_inference(args)


if __name__ == "__main__":
    main()
