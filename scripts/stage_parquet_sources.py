#!/usr/bin/env python3
"""Download, decode, and transcode parquet-based speech datasets for V100 training.

Adheres strictly to docs/qwen3-asr/08_execution_contract.md Stage E1:
- Extracts audio bytes directly from parquet columns.
- Uses ffmpeg to transcode into standard 16kHz mono PCM 16-bit WAV.
- Validates duration [0.5, 30.0] seconds and records SHA-256 checksums.
- Emits canonical staged_<source>.jsonl manifests ready for role partitioning.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Dict, Generator, List, Optional, Tuple
import wave
import yaml


try:
    from huggingface_hub import HfApi, hf_hub_download
except ImportError:
    HfApi = None  # type: ignore
    hf_hub_download = None  # type: ignore

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None  # type: ignore


PINNED_DATASETS = {
    "voices_in_the_wild": {
        "repo_id": "zhifeixie/Voices-in-the-Wild-2M",
        "revision": "a8a35d3319737190d6fd3d39157b258eaab35980",
        "condition_group": "degraded",
        "license": "Open / Research",
    },
    "librispeech_asr": {
        "repo_id": "openslr/librispeech_asr",
        "revision": "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1",
        "condition_group": "clean",
        "language": "en",
        "license": "CC-BY-4.0",
    },
    "aishell1": {
        "repo_id": "knoveleng/aishell1-mandarin",
        "revision": "c6dde006238091dc7c81cf5888208140bba33cba",
        "condition_group": "clean",
        "language": "zh",
        "license": "Apache-2.0",
    },
    "bench_test": {
        "repo_id": "zhifeixie/Voices-in-the-Wild-Bench",
        "revision": "788f5d72c6b0e9091b5c2e432370923b6f9f0660",
        "condition_group": "degraded",
        "license": "Evaluation Only",
    },
}


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def transcode_bytes_to_wav(
    audio_bytes: bytes,
    output_path: Path,
    target_sr: int = 16000,
    min_dur: float = 0.5,
    max_dur: float = 30.0,
) -> Tuple[bool, str, Optional[float], Optional[str]]:
    """Transcode raw audio bytes directly via ffmpeg stdin to 16kHz mono PCM 16-bit WAV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-ac",
        "1",
        "-ar",
        str(target_sr),
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    try:
        proc = subprocess.run(cmd, input=audio_bytes, capture_output=True, check=False)
        if proc.returncode != 0:
            return False, f"ffmpeg error ({proc.returncode}): {proc.stderr.decode('utf-8', errors='ignore').strip()}", None, None
    except Exception as exc:
        return False, f"ffmpeg invocation failed: {exc}", None, None

    # Inspect the generated WAV file
    try:
        with wave.open(str(output_path), "rb") as wf:
            channels = wf.getnchannels()
            framerate = wf.getframerate()
            sampwidth = wf.getsampwidth()
            nframes = wf.getnframes()
            if channels != 1 or framerate != target_sr or sampwidth != 2:
                return False, f"Audio format mismatch: ch={channels}, sr={framerate}, width={sampwidth}", None, None
            duration_s = round(nframes / float(framerate), 4)
            if duration_s < min_dur or duration_s > max_dur:
                output_path.unlink(missing_ok=True)
                return False, f"Duration {duration_s}s out of range [{min_dur}, {max_dur}]", duration_s, None
    except Exception as exc:
        output_path.unlink(missing_ok=True)
        return False, f"Invalid generated WAV: {exc}", None, None

    audio_hash = compute_sha256(output_path)
    return True, "", duration_s, audio_hash


def detect_language(text: str) -> str:
    """Detect if transcript is predominantly Chinese or English."""
    has_cjk = bool(re.search(r"[\u4e00-\u9fff]", text))
    return "zh" if has_cjk else "en"


def parse_bench_subset(subset: str) -> Tuple[str, str, str]:
    """Parse Bench subset name (e.g. 'real-zh-distortion' or 'synthetic-en-noise' or 'sim-en-noise')."""
    parts = subset.split("-")
    if len(parts) >= 3:
        origin_raw = parts[0]
        origin = "synthetic" if origin_raw in ("sim", "synthetic") else "real"
        lang = parts[1]
        scenario = "-".join(parts[2:])
        return origin, lang, scenario
    return "real", "en", subset


class ParquetSourceStager:
    def __init__(self, config_path: Path, endpoint: str = "https://hf-mirror.com"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        storage = self.config.get("storage", {})
        self.data_root = Path(storage.get("data_root", "/data/mega-asr/data"))
        self.manifest_root = Path(storage.get("manifest_root", "/data/mega-asr/manifests"))
        self.cache_root = Path(storage.get("cache_root", "/data/mega-asr/cache"))
        self.endpoint = endpoint
        self.seed = int(self.config.get("seed", 20260722))

        norm = self.config.get("audio_normalization", {})
        self.min_dur = float(norm.get("min_duration_s", 0.5))
        self.max_dur = float(norm.get("max_duration_s", 30.0))

        self.data_root.mkdir(parents=True, exist_ok=True)
        self.manifest_root.mkdir(parents=True, exist_ok=True)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def list_remote_parquet_files(self, repo_id: str, revision: str) -> List[str]:
        api = HfApi(endpoint=self.endpoint)
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)
        return [f for f in files if f.endswith(".parquet")]

    def download_shard(self, repo_id: str, revision: str, filename: str) -> Path:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            revision=revision,
            endpoint=self.endpoint,
            cache_dir=str(self.cache_root / "huggingface"),
        )
        return Path(local_path)

    def stage_source(
        self,
        source_name: str,
        max_samples: Optional[int] = None,
        max_shards: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Stage a single source by streaming parquet files and extracting 16kHz WAVs."""
        if source_name not in PINNED_DATASETS:
            raise ValueError(f"Unknown source: {source_name}")

        info = PINNED_DATASETS[source_name]
        repo_id = info["repo_id"]
        revision = info["revision"]

        print(f"\n==========================================")
        print(f"Staging source: {source_name} ({repo_id}@{revision})")
        print(f"==========================================")

        all_shards = self.list_remote_parquet_files(repo_id, revision)
        print(f"Found {len(all_shards)} remote parquet shards.")

        # Filter relevant shards by dataset
        if source_name == "librispeech_asr":
            # Target train.clean.100 and validation
            train_shards = [f for f in all_shards if "train.clean.100" in f]
            val_shards = [f for f in all_shards if "validation.clean" in f or "clean/validation" in f]
            if max_shards:
                half = max(1, max_shards // 2)
                target_shards = train_shards[:half] + val_shards[:max(1, max_shards - half)]
            else:
                target_shards = train_shards + val_shards
        elif source_name == "aishell1":
            # Target train and validation/dev
            train_shards = [f for f in all_shards if "train-" in f]
            val_shards = [f for f in all_shards if "validation-" in f or "dev-" in f]
            if max_shards:
                n_val = min(len(val_shards), max(2, max_shards // 4))
                n_train = max(1, max_shards - n_val)
                target_shards = train_shards[:n_train] + val_shards[:n_val]
            else:
                target_shards = train_shards + val_shards
        elif source_name == "voices_in_the_wild":
            # Target canonical scenarios for robust speech MVP: distortion, dropout, echo, far_field, noise, recording, obstructed
            canonical_scenarios = [
                "distortion", "dropout", "echo", "far_field", "noise",
                "recording", "obstructed"
            ]
            scenario_shards: List[str] = []
            per_sc = max(1, max_shards // len(canonical_scenarios)) if max_shards else None
            for sc in canonical_scenarios:
                prefix = f"data/{sc}-"
                matches = [f for f in all_shards if f.startswith(prefix)]
                if matches:
                    scenario_shards.extend(matches[:per_sc] if per_sc else matches)
            if max_shards:
                target_shards = scenario_shards[:max_shards]
                if len(target_shards) < max_shards:
                    extra = [f for f in all_shards if f not in target_shards]
                    target_shards.extend(extra[:max_shards - len(target_shards)])
            else:
                target_shards = scenario_shards if scenario_shards else all_shards
        else:
            target_shards = all_shards
            if max_shards:
                target_shards = target_shards[:max_shards]

        output_audio_dir = self.data_root / source_name
        output_audio_dir.mkdir(parents=True, exist_ok=True)

        manifest_file = self.manifest_root / f"staged_{source_name}.jsonl"
        rejects_file = self.manifest_root / f"rejects_{source_name}.jsonl"

        # Read existing records for incremental resumption
        existing_records: Dict[str, Dict[str, Any]] = {}
        if manifest_file.is_file():
            with open(manifest_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        sid = rec.get("sample_id")
                        audio_p = rec.get("audio")
                        if sid and audio_p and os.path.isfile(audio_p):
                            existing_records[sid] = rec
                    except Exception:
                        pass
            print(f"[{source_name}] Found {len(existing_records)} valid existing staged records for resumption.")

        staged_count = 0
        reject_count = 0
        written_sample_ids: Set[str] = set()

        with open(manifest_file, "w", encoding="utf-8") as out_mf, open(rejects_file, "w", encoding="utf-8") as out_rej:
            for shard_idx, shard_name in enumerate(target_shards):
                if max_samples and staged_count >= max_samples:
                    break

                shard_clean = Path(shard_name).stem.replace("-", "_").replace(".", "_")
                print(f"[{source_name}] Downloading & reading shard {shard_idx + 1}/{len(target_shards)}: {shard_name}...")
                shard_path = self.download_shard(repo_id, revision, shard_name)
                pf = pq.ParquetFile(shard_path)

                for rg_idx in range(pf.num_row_groups):
                    if max_samples and staged_count >= max_samples:
                        break

                    table = pf.read_row_group(rg_idx)
                    cols = table.column_names

                    for row_idx in range(table.num_rows):
                        if max_samples and staged_count >= max_samples:
                            break

                        # Extract text
                        text = ""
                        if "text" in cols:
                            text = str(table["text"][row_idx].as_py() or "").strip()
                        elif "answer" in cols:
                            text = str(table["answer"][row_idx].as_py() or "").strip()

                        if not text:
                            out_rej.write(json.dumps({"shard": shard_name, "row": row_idx, "reason": "Empty transcript"}) + "\n")
                            reject_count += 1
                            continue

                        # Extract identifiers & metadata
                        if source_name == "voices_in_the_wild":
                            name = str(table["name"][row_idx].as_py()) if "name" in cols else f"vitw_{shard_clean}_rg{rg_idx}_r{row_idx}"
                            scenario = str(table["subset"][row_idx].as_py()) if "subset" in cols else "noise"
                            sample_id = f"vitw_{name}_{scenario}"
                            utt_id = name
                            condition_group = "degraded"
                            origin = "synthetic" if "synthetic" in name else "real"
                            lang = detect_language(text)
                            split = "train"
                        elif source_name == "bench_test":
                            name = str(table["name"][row_idx].as_py()) if "name" in cols else f"bench_{shard_clean}_rg{rg_idx}_r{row_idx}"
                            subset = str(table["subset"][row_idx].as_py()) if "subset" in cols else "real-en-noise"
                            origin, lang, scenario = parse_bench_subset(subset)
                            sample_id = f"bench_{name}_{scenario}"
                            utt_id = name
                            condition_group = "degraded"
                            split = "test"
                        elif source_name == "librispeech_asr":
                            utt_id = str(table["id"][row_idx].as_py()) if "id" in cols else f"libri_{shard_clean}_rg{rg_idx}_r{row_idx}"
                            sample_id = f"libri_{utt_id}"
                            scenario = "clean"
                            condition_group = "clean"
                            origin = "real"
                            lang = "en"
                            split = "train.100" if "train.clean.100" in shard_name else "validation"
                        elif source_name == "aishell1":
                            path_val = None
                            if "audio" in cols:
                                a_dict = table["audio"][row_idx].as_py()
                                if isinstance(a_dict, dict) and a_dict.get("path"):
                                    path_val = a_dict["path"]
                            if not path_val and "path" in cols:
                                path_val = str(table["path"][row_idx].as_py() or "")

                            if path_val:
                                utt_id = Path(path_val).stem
                            else:
                                utt_id = f"aishell_{shard_clean}_rg{rg_idx}_r{row_idx}"
                            sample_id = f"aishell_{utt_id}"
                            scenario = "clean"
                            condition_group = "clean"
                            origin = "real"
                            lang = "zh"
                            split = "validation" if ("dev-" in shard_name or "validation" in shard_name) else "train"
                        else:
                            utt_id = f"{source_name}_{shard_clean}_rg{rg_idx}_r{row_idx}"
                            sample_id = utt_id
                            scenario = "clean"
                            condition_group = "clean"
                            origin = "real"
                            lang = "en"
                            split = "train"

                        if sample_id in written_sample_ids:
                            continue

                        # Check if sample was previously staged and WAV exists
                        if sample_id in existing_records:
                            cached_rec = existing_records[sample_id]
                            out_mf.write(json.dumps(cached_rec, ensure_ascii=False) + "\n")
                            written_sample_ids.add(sample_id)
                            staged_count += 1
                            continue

                        # Extract audio bytes
                        raw_bytes = None
                        if "audio" in cols:
                            a_val = table["audio"][row_idx].as_py()
                            if isinstance(a_val, dict):
                                raw_bytes = a_val.get("bytes")
                        elif "bytes" in cols:
                            raw_bytes = table["bytes"][row_idx].as_py()

                        if not raw_bytes:
                            out_rej.write(json.dumps({"shard": shard_name, "row": row_idx, "reason": "No audio bytes found"}) + "\n")
                            reject_count += 1
                            continue

                        # Transcode audio to 16kHz WAV
                        target_wav_path = output_audio_dir / f"{sample_id}.wav"
                        success, err, duration_s, audio_hash = transcode_bytes_to_wav(
                            raw_bytes, target_wav_path, target_sr=16000, min_dur=self.min_dur, max_dur=self.max_dur
                        )

                        if not success:
                            out_rej.write(json.dumps({"sample_id": sample_id, "reason": err, "shard": shard_name}) + "\n")
                            reject_count += 1
                            continue

                        record = {
                            "sample_id": sample_id,
                            "audio": str(target_wav_path.resolve()),
                            "text": text,
                            "language": lang,
                            "scenario": scenario,
                            "condition_group": condition_group,
                            "audio_origin": origin,
                            "source_dataset": repo_id,
                            "source_revision": revision,
                            "source_split": split,
                            "source_index": row_idx,
                            "source_utterance_id": utt_id,
                            "duration_s": duration_s,
                            "license": info["license"],
                            "seed": self.seed,
                            "audio_sha256": audio_hash,
                        }
                        out_mf.write(json.dumps(record, ensure_ascii=False) + "\n")
                        written_sample_ids.add(sample_id)
                        staged_count += 1

            # Ensure any existing valid records not touched in this run are preserved
            for sid, cached_rec in existing_records.items():
                if sid not in written_sample_ids:
                    out_mf.write(json.dumps(cached_rec, ensure_ascii=False) + "\n")
                    written_sample_ids.add(sid)
                    staged_count += 1

        print(f"[{source_name}] Completed! Staged: {staged_count}, Rejects: {reject_count}")
        return {
            "source": source_name,
            "repo_id": repo_id,
            "revision": revision,
            "staged_samples": staged_count,
            "rejected_samples": reject_count,
            "manifest_path": str(manifest_file.resolve()),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream, decode and transcode parquet speech datasets.")
    parser.add_argument("--config", type=Path, default=Path("configs/data/public_robust_v100.yaml"), help="Config file")
    parser.add_argument("--source", choices=list(PINNED_DATASETS.keys()) + ["all"], default="all", help="Source dataset to stage")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum samples per source (e.g. for smoke testing)")
    parser.add_argument("--max-shards", type=int, default=None, help="Maximum parquet shards per source")
    parser.add_argument("--endpoint", type=str, default="https://hf-mirror.com", help="HF endpoint")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.config.exists():
        print(f"Error: {args.config} not found", file=sys.stderr)
        return 1

    stager = ParquetSourceStager(args.config, endpoint=args.endpoint)
    sources_to_run = list(PINNED_DATASETS.keys()) if args.source == "all" else [args.source]

    summary = {}
    for src in sources_to_run:
        res = stager.stage_source(src, max_samples=args.max_samples, max_shards=args.max_shards)
        summary[src] = res

    # Record PROCESSED_COMPLETE.json
    status_str = "COMPLETE" if (args.source == "all" and not args.max_samples and not args.max_shards) else "SUBSET"
    gate_file = stager.manifest_root / "PROCESSED_COMPLETE.json"
    with open(gate_file, "w", encoding="utf-8") as f:
        json.dump({
            "status": status_str,
            "sources": summary,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }, f, indent=2)

    print(f"\nAll staging complete! Report saved to {gate_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
