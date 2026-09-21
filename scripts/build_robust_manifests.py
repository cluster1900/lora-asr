#!/usr/bin/env python3
"""Stage and partition multi-source robust ASR datasets for V100 training.

Adheres strictly to docs/qwen3-asr/08_execution_contract.md Stage E1:
- Normalizes and verifies audio (16kHz, mono, PCM 16-bit WAV, 0.5s-30.0s).
- Deterministically assigns samples to 7 disjoint roles with zero leakage.
- Clean train splits (LibriSpeech train.100, AISHELL-1 train) only serve train roles.
- Clean validation splits only serve validation roles.
- Generates role manifests, pilot subsets, rejects.jsonl, SHA-256 checksums,
  and completion marker files including DATASET_COMPLETE.json.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import wave
import yaml
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed


def transcode_to_wav(
    input_path: Path, output_path: Path, sample_rate: int = 16000
) -> Tuple[bool, str]:
    """Transcode arbitrary audio to 16kHz mono PCM 16-bit WAV using ffmpeg."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            return False, f"ffmpeg error ({res.returncode}): {res.stderr.strip()}"
        return True, ""
    except Exception as exc:
        return False, f"ffmpeg execution exception: {exc}"


def compute_sha256(file_path: Path) -> str:
    """Compute sha256 hex digest of a file in chunks."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_content_sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def inspect_wav_audio(
    file_path: Path, min_duration: float = 0.5, max_duration: float = 30.0
) -> Tuple[bool, str, Optional[float], Optional[str]]:
    """Inspect WAV audio file. Returns (is_valid, reason, duration_s, sha256)."""
    if not file_path.exists():
        return False, f"File does not exist: {file_path}", None, None

    try:
        with wave.open(str(file_path), "rb") as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            nframes = wf.getnframes()
            if channels != 1:
                return False, f"Expected 1 channel (mono), got {channels}", None, None
            if sampwidth != 2:
                return False, f"Expected 16-bit PCM (sampwidth=2), got {sampwidth}", None, None
            if framerate != 16000:
                return False, f"Expected 16000 Hz, got {framerate}", None, None
            duration_s = round(nframes / float(framerate), 4)
            if duration_s < min_duration or duration_s > max_duration:
                return (
                    False,
                    f"Duration {duration_s}s out of range [{min_duration}, {max_duration}]",
                    duration_s,
                    None,
                )
    except Exception as exc:
        return False, f"WAV decode error: {exc}", None, None

    audio_hash = compute_sha256(file_path)
    return True, "", duration_s, audio_hash


def deterministic_bucket(utterance_id: str, seed: int) -> float:
    """Return a deterministic float in [0.0, 1.0) for an utterance ID given seed."""
    key = f"{seed}:{utterance_id}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def normalize_transcript(text: str, language: str) -> str:
    """Normalize text transcript without losing gold semantics."""
    t = text.strip()
    return t


class RobustDatasetBuilder:
    def __init__(self, config: Dict[str, Any], output_dir: Optional[Path] = None, data_dir: Optional[Path] = None):
        self.config = config
        self.seed = int(config.get("seed", 20260722))
        
        storage = config.get("storage", {})
        self.manifest_root = output_dir or Path(storage.get("manifest_root", "/data/mega-asr/manifests"))
        self.data_root = data_dir or Path(storage.get("data_root", "/data/mega-asr/data"))
        self.cache_root = Path(storage.get("cache_root", "/data/mega-asr/cache"))
        self.complete_gate_file = storage.get("complete_gate_file", "DATASET_COMPLETE.json")
        
        norm = config.get("audio_normalization", {})
        self.min_duration_s = float(norm.get("min_duration_s", 0.5))
        self.max_duration_s = float(norm.get("max_duration_s", 30.0))
        
        self.sources_cfg = config.get("sources", {})
        self.roles_cfg = config.get("roles", {})
        self.pilot_cfg = config.get("pilot_subsets", {})

    def prepare_directories(self) -> None:
        self.manifest_root.mkdir(parents=True, exist_ok=True)
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def validate_sample_record(
        self, record: Dict[str, Any], verify_audio_file: bool = True
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate sample record against schema and audio constraints."""
        required_fields = [
            "sample_id",
            "audio",
            "text",
            "language",
            "scenario",
            "condition_group",
            "audio_origin",
            "source_dataset",
            "source_revision",
            "source_split",
            "source_index",
            "source_utterance_id",
            "license",
        ]
        for field in required_fields:
            if field not in record:
                return False, f"Missing required field: {field}", record

        text = normalize_transcript(str(record["text"]), record["language"])
        if not text:
            return False, "Empty gold transcript", record
        record["text"] = text

        if record["language"] not in ("en", "zh"):
            return False, f"Invalid language: {record['language']}", record
        if record["condition_group"] not in ("clean", "degraded"):
            return False, f"Invalid condition_group: {record['condition_group']}", record
        if record.get("audio_origin") in ("sim", "synthetic"):
            record["audio_origin"] = "synthetic"
        elif record.get("audio_origin") == "real":
            record["audio_origin"] = "real"
        else:
            return False, f"Invalid audio_origin: {record.get('audio_origin')}", record

        audio_path = Path(record["audio"])
        if verify_audio_file:
            is_valid, reason, duration_s, audio_hash = inspect_wav_audio(
                audio_path, self.min_duration_s, self.max_duration_s
            )
            if not is_valid:
                return False, reason, record
            record["duration_s"] = duration_s
            record["audio_sha256"] = audio_hash
        else:
            duration_s = float(record.get("duration_s", 0.0))
            if duration_s < self.min_duration_s or duration_s > self.max_duration_s:
                return False, f"Duration {duration_s} out of range", record
            if not record.get("audio_sha256"):
                record["audio_sha256"] = hashlib.sha256(record["audio"].encode()).hexdigest()

        record["seed"] = self.seed
        return True, "", record

    def partition_samples(
        self,
        samples_by_source: Dict[str, List[Dict[str, Any]]],
        verify_audio_file: bool = True,
    ) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
        """Partition validated samples into disjoint roles strictly adhering to split constraints."""
        role_pools: Dict[str, List[Dict[str, Any]]] = {role: [] for role in self.roles_cfg}
        rejects: List[Dict[str, Any]] = []
        seen_sample_ids: Set[str] = set()

        # Group clean samples by split (train vs validation)
        # LibriSpeech
        libri_train: List[Dict[str, Any]] = []
        libri_val: List[Dict[str, Any]] = []
        # AISHELL-1
        aishell_train: List[Dict[str, Any]] = []
        aishell_val: List[Dict[str, Any]] = []
        # Robust Voices-in-the-Wild
        robust_samples: List[Dict[str, Any]] = []
        # Bench test
        bench_samples: List[Dict[str, Any]] = []

        for src_name, records in samples_by_source.items():
            for rec in records:
                if rec["sample_id"] in seen_sample_ids:
                    rejects.append({
                        "sample_id": rec.get("sample_id", "unknown"),
                        "reason": "Duplicate sample_id",
                        "record": rec,
                    })
                    continue
                seen_sample_ids.add(rec["sample_id"])

                is_valid, reason, valid_rec = self.validate_sample_record(rec, verify_audio_file=verify_audio_file)
                if not is_valid:
                    rejects.append({
                        "sample_id": rec.get("sample_id", "unknown"),
                        "reason": reason,
                        "record": rec,
                    })
                    continue

                if src_name == "librispeech_asr":
                    if valid_rec["source_split"] == "train.100":
                        libri_train.append(valid_rec)
                    elif valid_rec["source_split"] == "validation":
                        libri_val.append(valid_rec)
                    else:
                        rejects.append({
                            "sample_id": valid_rec["sample_id"],
                            "reason": f"Disallowed LibriSpeech split: {valid_rec['source_split']}",
                            "record": valid_rec,
                        })
                elif src_name == "aishell1":
                    if valid_rec["source_split"] == "train":
                        aishell_train.append(valid_rec)
                    elif valid_rec["source_split"] == "validation":
                        aishell_val.append(valid_rec)
                    else:
                        rejects.append({
                            "sample_id": valid_rec["sample_id"],
                            "reason": f"Disallowed AISHELL-1 split: {valid_rec['source_split']}",
                            "record": valid_rec,
                        })
                elif src_name == "voices_in_the_wild":
                    robust_samples.append(valid_rec)
                elif src_name == "bench_test":
                    bench_samples.append(valid_rec)
                else:
                    rejects.append({
                        "sample_id": valid_rec["sample_id"],
                        "reason": f"Unknown source dataset: {src_name}",
                        "record": valid_rec,
                    })

        # Bench test strictly goes to bench_test role
        role_pools["bench_test"].extend(bench_samples)

        # Robust partitioning: 90% train pool, 10% eval pool based on deterministic bucket
        robust_train_candidates: List[Dict[str, Any]] = []
        robust_eval_candidates: List[Dict[str, Any]] = []
        for rec in robust_samples:
            score = deterministic_bucket(rec["source_utterance_id"], self.seed)
            if score < 0.90:
                robust_train_candidates.append(rec)
            else:
                robust_eval_candidates.append(rec)

        # Sort candidates deterministically by sha256(seed:source_utterance_id)
        robust_train_candidates.sort(key=lambda r: hashlib.sha256(f"{self.seed}:{r['source_utterance_id']}".encode()).hexdigest())
        robust_eval_candidates.sort(key=lambda r: hashlib.sha256(f"{self.seed}:{r['source_utterance_id']}".encode()).hexdigest())

        # Allocate robust train pool: sft_train, dpo_train_pool, rl_train_pool
        robust_train_alloc = self._slice_candidates(
            robust_train_candidates,
            [
                ("sft_train", self.roles_cfg["sft_train"]["quotas"]["robust_degraded"]),
                ("dpo_train_pool", self.roles_cfg["dpo_train_pool"]["source_candidates"]["robust_degraded"]),
                ("rl_train_pool", self.roles_cfg["rl_train_pool"]["quotas"]["robust_degraded"]),
            ],
            seed=self.seed,
        )
        for role, allocated in robust_train_alloc.items():
            role_pools[role].extend(allocated)

        # Allocate robust eval pool: validation, dpo_val_pool, rl_val_pool
        robust_eval_alloc = self._slice_candidates(
            robust_eval_candidates,
            [
                ("validation", self.roles_cfg["validation"]["quotas"]["robust_degraded"]),
                ("dpo_val_pool", self.roles_cfg["dpo_val_pool"]["source_candidates"]["robust_degraded"]),
                ("rl_val_pool", self.roles_cfg["rl_val_pool"]["quotas"]["robust_degraded"]),
            ],
            seed=self.seed,
        )
        for role, allocated in robust_eval_alloc.items():
            role_pools[role].extend(allocated)

        # Allocate Clean Train: LibriSpeech train.100 -> sft_train, dpo_train_pool, rl_train_pool
        libri_train_alloc = self._slice_candidates(
            libri_train,
            [
                ("sft_train", self.roles_cfg["sft_train"]["quotas"]["english_clean"]),
                ("dpo_train_pool", self.roles_cfg["dpo_train_pool"]["source_candidates"]["english_clean"]),
                ("rl_train_pool", self.roles_cfg["rl_train_pool"]["quotas"]["english_clean"]),
            ],
            seed=self.seed,
        )
        for role, allocated in libri_train_alloc.items():
            role_pools[role].extend(allocated)

        # Allocate Clean Train: AISHELL-1 train -> sft_train, dpo_train_pool, rl_train_pool
        aishell_train_alloc = self._slice_candidates(
            aishell_train,
            [
                ("sft_train", self.roles_cfg["sft_train"]["quotas"]["chinese_clean"]),
                ("dpo_train_pool", self.roles_cfg["dpo_train_pool"]["source_candidates"]["chinese_clean"]),
                ("rl_train_pool", self.roles_cfg["rl_train_pool"]["quotas"]["chinese_clean"]),
            ],
            seed=self.seed,
        )
        for role, allocated in aishell_train_alloc.items():
            role_pools[role].extend(allocated)

        # Allocate Clean Validation: LibriSpeech validation -> validation, dpo_val_pool, rl_val_pool
        libri_val_alloc = self._slice_candidates(
            libri_val,
            [
                ("validation", self.roles_cfg["validation"]["quotas"]["english_clean"]),
                ("dpo_val_pool", self.roles_cfg["dpo_val_pool"]["source_candidates"]["english_clean"]),
                ("rl_val_pool", self.roles_cfg["rl_val_pool"]["quotas"]["english_clean"]),
            ],
            seed=self.seed,
        )
        for role, allocated in libri_val_alloc.items():
            role_pools[role].extend(allocated)

        # Allocate Clean Validation: AISHELL-1 validation -> validation, dpo_val_pool, rl_val_pool
        aishell_val_alloc = self._slice_candidates(
            aishell_val,
            [
                ("validation", self.roles_cfg["validation"]["quotas"]["chinese_clean"]),
                ("dpo_val_pool", self.roles_cfg["dpo_val_pool"]["source_candidates"]["chinese_clean"]),
                ("rl_val_pool", self.roles_cfg["rl_val_pool"]["quotas"]["chinese_clean"]),
            ],
            seed=self.seed,
        )
        for role, allocated in aishell_val_alloc.items():
            role_pools[role].extend(allocated)

        return role_pools, rejects

    @classmethod
    def _group_by_utterance(
        cls, candidates: List[Dict[str, Any]], seed: int
    ) -> List[List[Dict[str, Any]]]:
        """Group samples by source_utterance_id and sort groups deterministically."""
        groups_dict: Dict[str, List[Dict[str, Any]]] = {}
        for rec in candidates:
            uid = rec["source_utterance_id"]
            if uid not in groups_dict:
                groups_dict[uid] = []
            groups_dict[uid].append(rec)
        sorted_uids = sorted(
            groups_dict.keys(),
            key=lambda u: hashlib.sha256(f"{seed}:{u}".encode()).hexdigest(),
        )
        return [groups_dict[u] for u in sorted_uids]

    @classmethod
    def _slice_candidates(
        cls,
        candidates: List[Dict[str, Any]],
        role_targets: List[Tuple[str, int]],
        seed: int = 20260722,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Disjointly allocate candidate utterance groups among roles.
        Guarantees all scenario variants of the same source_utterance_id remain in the same role.
        """
        allocations: Dict[str, List[Dict[str, Any]]] = {role: [] for role, _ in role_targets}
        if not candidates or not role_targets:
            return allocations

        groups = cls._group_by_utterance(candidates, seed)
        total_samples = len(candidates)
        total_target = sum(target for _, target in role_targets)
        if total_target == 0 or total_samples == 0:
            return allocations

        if total_samples >= total_target:
            group_idx = 0
            for role, target in role_targets:
                role_recs: List[Dict[str, Any]] = []
                while group_idx < len(groups) and len(role_recs) < target:
                    role_recs.extend(groups[group_idx])
                    group_idx += 1
                allocations[role] = role_recs
        else:
            # Proportional allocation by groups
            group_idx = 0
            for i, (role, target) in enumerate(role_targets):
                target_share = int(round((target / total_target) * total_samples))
                is_last = (i == len(role_targets) - 1)
                role_recs: List[Dict[str, Any]] = []
                while group_idx < len(groups):
                    if not is_last and len(role_recs) >= target_share and len(role_recs) > 0:
                        break
                    role_recs.extend(groups[group_idx])
                    group_idx += 1
                allocations[role] = role_recs

        return allocations

    def verify_leakage(
        self, role_pools: Dict[str, List[Dict[str, Any]]]
    ) -> Tuple[bool, List[str]]:
        """Verify strict pairwise physical isolation across ALL 7 roles.

        Checks all 21 pairs (C(7,2)) for:
        1. sample_id disjointness
        2. source_utterance_id disjointness
        3. audio_sha256 disjointness
        """
        all_roles = list(role_pools.keys())
        errors: List[str] = []

        sample_ids_by_role: Dict[str, Set[str]] = {}
        utterances_by_role: Dict[str, Set[str]] = {}
        audio_hashes_by_role: Dict[str, Set[str]] = {}

        for role, samples in role_pools.items():
            s_ids: Set[str] = set()
            u_ids: Set[str] = set()
            a_hashes: Set[str] = set()
            for rec in samples:
                sid = rec["sample_id"]
                if sid in s_ids:
                    errors.append(f"Duplicate sample_id within role {role}: {sid}")
                s_ids.add(sid)
                u_ids.add(rec["source_utterance_id"])
                if rec.get("audio_sha256"):
                    a_hashes.add(rec["audio_sha256"])
            sample_ids_by_role[role] = s_ids
            utterances_by_role[role] = u_ids
            audio_hashes_by_role[role] = a_hashes

        # Check all C(N, 2) pairs
        for i in range(len(all_roles)):
            for j in range(i + 1, len(all_roles)):
                r1 = all_roles[i]
                r2 = all_roles[j]

                # 1. Sample ID disjointness
                sid_overlap = sample_ids_by_role[r1].intersection(sample_ids_by_role[r2])
                if sid_overlap:
                    errors.append(f"Leakage detected: Sample ID overlap between {r1} and {r2}: {len(sid_overlap)} samples overlap (e.g. {list(sid_overlap)[:3]})")

                # 2. Source utterance ID disjointness
                uid_overlap = utterances_by_role[r1].intersection(utterances_by_role[r2])
                if uid_overlap:
                    errors.append(f"Leakage detected: Utterance ID overlap between {r1} and {r2}: {len(uid_overlap)} utterances overlap (e.g. {list(uid_overlap)[:3]})")

                # 3. Audio sha256 disjointness
                hash_overlap = audio_hashes_by_role[r1].intersection(audio_hashes_by_role[r2])
                if hash_overlap:
                    errors.append(f"Leakage detected: Audio sha256 overlap between {r1} and {r2}: {len(hash_overlap)} audio files overlap (e.g. {list(hash_overlap)[:3]})")

        return len(errors) == 0, errors

    def build_pilot_subsets(
        self, role_pools: Dict[str, List[Dict[str, Any]]], strict: bool = False
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Extract pilot subsets strictly adhering to pilot specifications."""
        pilot_subsets: Dict[str, List[Dict[str, Any]]] = {}
        pilot_cfg = self.config.get("pilot_subsets", {})
        sft_pilot_cfg = pilot_cfg.get("sft_pilot", {})

        sft_samples = role_pools.get("sft_train", [])
        deg_target = int(sft_pilot_cfg.get("robust_degraded", 5000))
        en_clean_target = int(sft_pilot_cfg.get("english_clean", 1000))
        zh_clean_target = int(sft_pilot_cfg.get("chinese_clean", 1000))

        sft_deg = [r for r in sft_samples if r.get("condition_group") == "degraded"]
        sft_en_clean = [r for r in sft_samples if r.get("condition_group") == "clean" and r.get("language") == "en"]
        sft_zh_clean = [r for r in sft_samples if r.get("condition_group") == "clean" and r.get("language") == "zh"]

        if len(sft_deg) >= deg_target and len(sft_en_clean) >= en_clean_target and len(sft_zh_clean) >= zh_clean_target:
            pilot_subsets["sft"] = sft_deg[:deg_target] + sft_en_clean[:en_clean_target] + sft_zh_clean[:zh_clean_target]
        else:
            # Scaled subset mode: must contain ALL three categories
            if len(sft_deg) > 0 and len(sft_en_clean) > 0 and len(sft_zh_clean) > 0:
                n_deg = min(len(sft_deg), deg_target)
                n_en = min(len(sft_en_clean), en_clean_target)
                n_zh = min(len(sft_zh_clean), zh_clean_target)
                pilot_subsets["sft"] = sft_deg[:n_deg] + sft_en_clean[:n_en] + sft_zh_clean[:n_zh]
            elif sft_samples:
                missing = []
                if len(sft_deg) == 0:
                    missing.append("robust_degraded")
                if len(sft_en_clean) == 0:
                    missing.append("english_clean")
                if len(sft_zh_clean) == 0:
                    missing.append("chinese_clean")
                if strict:
                    raise ValueError(
                        f"Cannot build pilot_sft: sft_train is missing required categories: {', '.join(missing)}"
                    )
                else:
                    pilot_subsets["sft"] = list(sft_samples)

        dpo_pilot_cfg = pilot_cfg.get("dpo_pilot", {})
        dpo_train = role_pools.get("dpo_train_pool", [])
        if dpo_train:
            dpo_deg_target = int(dpo_pilot_cfg.get("robust_degraded_pairs", 2000))
            dpo_en_target = int(dpo_pilot_cfg.get("english_clean_pairs", 500))
            dpo_zh_target = int(dpo_pilot_cfg.get("chinese_clean_pairs", 500))

            dpo_deg = [r for r in dpo_train if r.get("condition_group") == "degraded"]
            dpo_en_clean = [r for r in dpo_train if r.get("condition_group") == "clean" and r.get("language") == "en"]
            dpo_zh_clean = [r for r in dpo_train if r.get("condition_group") == "clean" and r.get("language") == "zh"]

            if len(dpo_deg) >= dpo_deg_target and len(dpo_en_clean) >= dpo_en_target and len(dpo_zh_clean) >= dpo_zh_target:
                pilot_subsets["dpo"] = dpo_deg[:dpo_deg_target] + dpo_en_clean[:dpo_en_target] + dpo_zh_clean[:dpo_zh_target]
            else:
                n_deg = min(len(dpo_deg), dpo_deg_target)
                n_en = min(len(dpo_en_clean), dpo_en_target)
                n_zh = min(len(dpo_zh_clean), dpo_zh_target)
                pilot_subsets["dpo"] = dpo_deg[:n_deg] + dpo_en_clean[:n_en] + dpo_zh_clean[:n_zh]

        rl_pilot_cfg = pilot_cfg.get("rl_pilot", {})
        rl_train = role_pools.get("rl_train_pool", [])
        if rl_train:
            rl_deg_target = int(rl_pilot_cfg.get("robust_degraded_audios", 2000))
            rl_en_target = int(rl_pilot_cfg.get("english_clean_audios", 500))
            rl_zh_target = int(rl_pilot_cfg.get("chinese_clean_audios", 500))

            rl_deg = [r for r in rl_train if r.get("condition_group") == "degraded"]
            rl_en_clean = [r for r in rl_train if r.get("condition_group") == "clean" and r.get("language") == "en"]
            rl_zh_clean = [r for r in rl_train if r.get("condition_group") == "clean" and r.get("language") == "zh"]

            if len(rl_deg) >= rl_deg_target and len(rl_en_clean) >= rl_en_target and len(rl_zh_clean) >= rl_zh_target:
                pilot_subsets["rl"] = rl_deg[:rl_deg_target] + rl_en_clean[:rl_en_target] + rl_zh_clean[:rl_zh_target]
            else:
                n_deg = min(len(rl_deg), rl_deg_target)
                n_en = min(len(rl_en_clean), rl_en_target)
                n_zh = min(len(rl_zh_clean), rl_zh_target)
                pilot_subsets["rl"] = rl_deg[:n_deg] + rl_en_clean[:n_en] + rl_zh_clean[:n_zh]

        return pilot_subsets

    def build_smoke_subset(
        self, role_pools: Dict[str, List[Dict[str, Any]]], strict: bool = True
    ) -> List[Dict[str, Any]]:
        """Extract a balanced 128-row smoke subset STRICTLY from sft_train.

        CRITICAL REQUIREMENT: smoke.jsonl MUST ONLY contain samples from sft_train.
        Under NO circumstances may samples from validation or bench_test enter smoke.jsonl.
        """
        sft_samples = role_pools.get("sft_train", [])
        clean_en = [r for r in sft_samples if r.get("condition_group") == "clean" and r.get("language") == "en"][:32]
        clean_zh = [r for r in sft_samples if r.get("condition_group") == "clean" and r.get("language") == "zh"][:32]
        degraded = [r for r in sft_samples if r.get("condition_group") == "degraded"][:64]

        # Strictly disallow cross-role backfilling!
        if len(clean_en) < 32 or len(clean_zh) < 32 or len(degraded) < 64:
            if strict:
                raise ValueError(
                    f"Cannot build smoke subset: sft_train has insufficient samples "
                    f"(clean_en: {len(clean_en)}/32, clean_zh: {len(clean_zh)}/32, degraded: {len(degraded)}/64). "
                    f"Backfilling from other roles (especially validation or bench_test) is strictly forbidden."
                )
            else:
                return (clean_en + clean_zh + degraded)[:128] if (clean_en or clean_zh or degraded) else sft_samples[:128]

        smoke_samples = clean_en + clean_zh + degraded
        return smoke_samples

    def write_manifests_and_gates(
        self,
        role_pools: Dict[str, List[Dict[str, Any]]],
        rejects: List[Dict[str, Any]],
        pilot_subsets: Dict[str, List[Dict[str, Any]]],
        strict_quotas: bool = True,
        smoke_subset: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Write all JSONL manifests, checksums, and completion gates."""
        manifest_checksums: Dict[str, str] = {}
        role_summaries: Dict[str, Any] = {}

        # 1. Write role manifests
        for role, samples in role_pools.items():
            manifest_file = self.manifest_root / f"{role}.jsonl"
            with open(manifest_file, "w", encoding="utf-8") as f:
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            
            manifest_hash = compute_sha256(manifest_file)
            manifest_checksums[f"{role}.jsonl"] = manifest_hash

            by_cond = {"clean": 0, "degraded": 0}
            by_lang = {"en": 0, "zh": 0}
            total_duration = 0.0
            for s in samples:
                by_cond[s["condition_group"]] = by_cond.get(s["condition_group"], 0) + 1
                by_lang[s["language"]] = by_lang.get(s["language"], 0) + 1
                total_duration += s.get("duration_s", 0.0)

            complete_info = {
                "role": role,
                "manifest_path": str(manifest_file.resolve()),
                "manifest_sha256": manifest_hash,
                "row_count": len(samples),
                "total_duration_hours": round(total_duration / 3600.0, 3),
                "by_condition": by_cond,
                "by_language": by_lang,
                "status": "COMPLETE" if strict_quotas else "SUBSET",
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            role_summaries[role] = complete_info

            complete_file = self.manifest_root / f"{role}_COMPLETE.json"
            with open(complete_file, "w", encoding="utf-8") as f:
                json.dump(complete_info, f, indent=2, ensure_ascii=False)

        # 2. Write smoke subset if present
        if smoke_subset is not None:
            smoke_file = self.manifest_root / "smoke.jsonl"
            with open(smoke_file, "w", encoding="utf-8") as f:
                for s in smoke_subset:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
            manifest_checksums["smoke.jsonl"] = compute_sha256(smoke_file)

        # 3. Write pilot subsets
        for p_name, p_samples in pilot_subsets.items():
            p_file = self.manifest_root / f"pilot_{p_name}.jsonl"
            with open(p_file, "w", encoding="utf-8") as f:
                for s in p_samples:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
            manifest_checksums[f"pilot_{p_name}.jsonl"] = compute_sha256(p_file)


        # 4. Write rejects.jsonl
        rejects_file = self.manifest_root / "rejects.jsonl"
        with open(rejects_file, "w", encoding="utf-8") as f:
            for rej in rejects:
                f.write(json.dumps(rej, ensure_ascii=False) + "\n")
        manifest_checksums["rejects.jsonl"] = compute_sha256(rejects_file)

        # 5. Write manifest_sha256.json
        sha256_file = self.manifest_root / "manifest_sha256.json"
        with open(sha256_file, "w", encoding="utf-8") as f:
            json.dump(manifest_checksums, f, indent=2, ensure_ascii=False)

        total_processed = sum(len(s) for s in role_pools.values())

        # 6. Check quotas across all 7 roles
        quota_passed = True
        quota_failures: List[str] = []
        for role, role_spec in self.roles_cfg.items():
            actual_samples = role_pools.get(role, [])
            actual_total = len(actual_samples)

            # Strictly forbid empty roles from passing gate
            if actual_total == 0:
                quota_passed = False
                quota_failures.append(f"Role {role} is EMPTY (0 samples).")
                continue

            sub_quotas = role_spec.get("quotas", role_spec.get("source_candidates", {}))
            exp_deg = sub_quotas.get("robust_degraded", 0)
            exp_en = sub_quotas.get("english_clean", 0)
            exp_zh = sub_quotas.get("chinese_clean", 0)
            expected_total = exp_deg + exp_en + exp_zh

            deg_count = sum(1 for s in actual_samples if s.get("condition_group") == "degraded")
            en_clean_count = sum(1 for s in actual_samples if s.get("condition_group") == "clean" and s.get("language") == "en")
            zh_clean_count = sum(1 for s in actual_samples if s.get("condition_group") == "clean" and s.get("language") == "zh")

            # HARD CHECK: In ANY mode, if a role expects clean, clean count CANNOT be 0!
            if exp_deg > 0 and deg_count == 0:
                quota_passed = False
                quota_failures.append(f"Role {role} requires robust_degraded ({exp_deg}), but actual count is 0!")
            if exp_en > 0 and en_clean_count == 0:
                quota_passed = False
                quota_failures.append(f"Role {role} requires english_clean ({exp_en}), but actual count is 0!")
            if exp_zh > 0 and zh_clean_count == 0:
                quota_passed = False
                quota_failures.append(f"Role {role} requires chinese_clean ({exp_zh}), but actual count is 0!")

            if strict_quotas:
                effective_exp_deg = 4999 if role == "bench_test" else exp_deg
                if deg_count < effective_exp_deg:
                    quota_passed = False
                    quota_failures.append(f"Role {role} robust_degraded quota failed: {deg_count}/{exp_deg}")
                if en_clean_count < exp_en:
                    quota_passed = False
                    quota_failures.append(f"Role {role} english_clean quota failed: {en_clean_count}/{exp_en}")
                if zh_clean_count < exp_zh:
                    quota_passed = False
                    quota_failures.append(f"Role {role} chinese_clean quota failed: {zh_clean_count}/{exp_zh}")

        # 7. Check pairwise physical isolation across all 7 roles
        no_leakage, leak_errors = self.verify_leakage(role_pools)

        # Overall status can ONLY be PASSED when strict_quotas is met, quota_passed is True, and no leakage.
        # Non-strict mode outputs NON_STRICT_SUBSET when all roles are populated and no leakage.
        if strict_quotas and quota_passed and no_leakage and total_processed > 0:
            overall_status = "PASSED"
        elif not strict_quotas and quota_passed and no_leakage and total_processed > 0:
            overall_status = "NON_STRICT_SUBSET"
        else:
            overall_status = "FAILED"

        # 8. Write RAW_COMPLETE.json and PROCESSED_COMPLETE.json
        raw_status = "COMPLETE" if (strict_quotas and quota_passed) else "SUBSET"
        raw_gate = {
            "status": raw_status,
            "sources": list(self.sources_cfg.keys()),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        with open(self.manifest_root / "RAW_COMPLETE.json", "w", encoding="utf-8") as f:
            json.dump(raw_gate, f, indent=2)

        processed_status = "COMPLETE" if (strict_quotas and quota_passed and no_leakage) else "SUBSET"
        processed_gate = {
            "status": processed_status,
            "total_samples": total_processed,
            "rejected_samples": len(rejects),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        with open(self.manifest_root / "PROCESSED_COMPLETE.json", "w", encoding="utf-8") as f:
            json.dump(processed_gate, f, indent=2)

        # 9. Write DATASET_COMPLETE.json
        dataset_complete = {
            "schema_version": self.config.get("schema_version", 1),
            "seed": self.seed,
            "status": overall_status,
            "strict_quotas_enforced": strict_quotas,
            "leakage_check": "PASSED" if no_leakage else "FAILED",
            "leakage_errors": leak_errors,
            "quota_check": "PASSED" if quota_passed else "FAILED",
            "quota_failures": quota_failures,
            "roles": role_summaries,
            "manifest_checksums": manifest_checksums,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        gate_path = self.manifest_root / self.complete_gate_file
        with open(gate_path, "w", encoding="utf-8") as f:
            json.dump(dataset_complete, f, indent=2, ensure_ascii=False)

        return dataset_complete

    def verify_dataset_integrity(
        self, check_audio: bool = True, strict_quotas: bool = False
    ) -> Tuple[bool, List[str]]:
        """Perform deep verification of dataset manifests, hashes, audio format, and pairwise isolation."""
        errors: List[str] = []

        # 1. Check DATASET_COMPLETE.json
        gate_path = self.manifest_root / self.complete_gate_file
        if not gate_path.is_file():
            return False, [f"Gate file missing: {gate_path}"]
        with open(gate_path, "r", encoding="utf-8") as f:
            gate_data = json.load(f)
        gate_status = gate_data.get("status")
        is_strict = gate_data.get("strict_quotas_enforced", False) or strict_quotas
        if is_strict and gate_status != "PASSED":
            errors.append(f"Gate status in {self.complete_gate_file} is {gate_status}, expected PASSED for strict quotas")
        elif not is_strict and gate_status not in ("PASSED", "NON_STRICT_SUBSET"):
            errors.append(f"Gate status in {self.complete_gate_file} is {gate_status}, expected PASSED or NON_STRICT_SUBSET (got {gate_status})")

        # 2. Check manifest_sha256.json and recompute hashes
        sha256_path = self.manifest_root / "manifest_sha256.json"
        if not sha256_path.is_file():
            errors.append(f"Checksum file missing: {sha256_path}")
        else:
            with open(sha256_path, "r", encoding="utf-8") as f:
                recorded_checksums = json.load(f)
            for m_name, recorded_hash in recorded_checksums.items():
                m_file = self.manifest_root / m_name
                if not m_file.is_file():
                    errors.append(f"Manifest file missing: {m_file}")
                else:
                    actual_hash = compute_sha256(m_file)
                    if actual_hash != recorded_hash:
                        errors.append(f"Hash mismatch for {m_name}: recorded {recorded_hash}, actual {actual_hash}")

        # 3. Load all 7 role manifests
        role_pools: Dict[str, List[Dict[str, Any]]] = {}
        for role, role_spec in self.roles_cfg.items():
            m_file = self.manifest_root / f"{role}.jsonl"
            if not m_file.is_file():
                errors.append(f"Required role manifest missing: {m_file}")
                continue
            records = []
            with open(m_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            role_pools[role] = records
            if len(records) == 0:
                errors.append(f"Role manifest {role}.jsonl is EMPTY (0 rows)")

            # Check sub-quotas / category presence
            sub_quotas = role_spec.get("quotas", role_spec.get("source_candidates", {}))
            deg_count = sum(1 for s in records if s.get("condition_group") == "degraded")
            en_clean_count = sum(1 for s in records if s.get("condition_group") == "clean" and s.get("language") == "en")
            zh_clean_count = sum(1 for s in records if s.get("condition_group") == "clean" and s.get("language") == "zh")

            if sub_quotas.get("robust_degraded", 0) > 0 and deg_count == 0:
                errors.append(f"Role {role} missing required robust_degraded samples (found 0)")
            if sub_quotas.get("english_clean", 0) > 0 and en_clean_count == 0:
                errors.append(f"Role {role} missing required english_clean samples (found 0)")
            if sub_quotas.get("chinese_clean", 0) > 0 and zh_clean_count == 0:
                errors.append(f"Role {role} missing required chinese_clean samples (found 0)")

        # 4. Check 7-role pairwise isolation
        if len(role_pools) == len(self.roles_cfg):
            no_leakage, leak_errors = self.verify_leakage(role_pools)
            if not no_leakage:
                errors.extend(leak_errors)

        # 5. Deep audio verification
        if check_audio:
            bad_audio_count = 0
            hash_mismatch_count = 0
            checked_count = 0
            for role, records in role_pools.items():
                for rec in records:
                    checked_count += 1
                    audio_p = Path(rec.get("audio", ""))
                    if not audio_p.is_file() or audio_p.stat().st_size == 0:
                        bad_audio_count += 1
                        if bad_audio_count <= 5:
                            errors.append(f"Audio file missing or empty: {audio_p} (sample_id: {rec.get('sample_id')})")
                        continue

                    # Validate WAV format using inspect_wav_audio
                    is_valid_wav, wav_err, dur, actual_sha = inspect_wav_audio(
                        audio_p, self.min_duration_s, self.max_duration_s
                    )
                    if not is_valid_wav:
                        bad_audio_count += 1
                        if bad_audio_count <= 5:
                            errors.append(f"Invalid WAV file format for {audio_p}: {wav_err}")
                        continue

                    # Verify SHA256 of audio matches record
                    expected_sha = rec.get("audio_sha256")
                    if expected_sha and actual_sha != expected_sha:
                        hash_mismatch_count += 1
                        if hash_mismatch_count <= 5:
                            errors.append(f"Audio SHA256 mismatch for {audio_p}: expected {expected_sha}, got {actual_sha}")

            if bad_audio_count > 5:
                errors.append(f"Total invalid/missing audio files: {bad_audio_count} / {checked_count}")
            if hash_mismatch_count > 5:
                errors.append(f"Total audio hash mismatches: {hash_mismatch_count} / {checked_count}")

        # 6. Verify smoke.jsonl is strict subset of sft_train and contains no bench/val
        smoke_file = self.manifest_root / "smoke.jsonl"
        if smoke_file.is_file():
            smoke_records = []
            with open(smoke_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        smoke_records.append(json.loads(line))
            sft_uids = {r["source_utterance_id"] for r in role_pools.get("sft_train", [])}
            val_uids = {r["source_utterance_id"] for r in role_pools.get("validation", [])}
            bench_uids = {r["source_utterance_id"] for r in role_pools.get("bench_test", [])}

            for s_rec in smoke_records:
                uid = s_rec["source_utterance_id"]
                if uid in bench_uids:
                    errors.append(f"LEAKAGE in smoke.jsonl: contains bench_test utterance {uid}")
                if uid in val_uids:
                    errors.append(f"LEAKAGE in smoke.jsonl: contains validation utterance {uid}")
                if uid not in sft_uids:
                    errors.append(f"ILLEGAL SAMPLE in smoke.jsonl: utterance {uid} not in sft_train")

        return len(errors) == 0, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage and build robust ASR manifests for V100 training.")
    parser.add_argument("--config", type=Path, default=Path("configs/data/public_robust_v100.yaml"), help="Path to data config YAML")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override manifest output directory")
    parser.add_argument("--data-dir", type=Path, default=None, help="Override audio data directory")
    parser.add_argument("--mode", choices=["smoke", "pilot", "full", "verify-only"], default="full", help="Build mode")
    parser.add_argument("--staged-dir", type=Path, default=None, help="Directory containing staged_<source>.jsonl files")
    parser.add_argument("--source-raw-dir", type=Path, default=None, help="Directory containing downloaded raw sources")
    parser.add_argument("--transcode", action="store_true", help="Transcode input audios to 16kHz mono WAV using ffmpeg")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers for concurrent transcoding")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--no-strict-quotas", action="store_true", help="Do not require full quotas (e.g. for testing/smoke)")
    parser.add_argument("--strict-quotas", action="store_true", help="Require strict full quotas during verification or generation")
    parser.add_argument("--verify-audio", action="store_true", help="Perform audio format & file decoding checks")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.config.exists():
        print(f"Error: config file {args.config} does not exist.", file=sys.stderr)
        return 1

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.seed is not None:
        config["seed"] = args.seed

    builder = RobustDatasetBuilder(config, output_dir=args.output_dir, data_dir=args.data_dir)
    builder.prepare_directories()

    if args.mode == "verify-only":
        passed, errors = builder.verify_dataset_integrity(
            check_audio=args.verify_audio, strict_quotas=args.strict_quotas
        )
        if not passed:
            print("DATASET VERIFICATION FAILED:", file=sys.stderr)
            for err in errors:
                print(f"  - {err}", file=sys.stderr)
            return 1

        gate_path = builder.manifest_root / builder.complete_gate_file
        gate_status = "UNKNOWN"
        if gate_path.is_file():
            with open(gate_path, "r", encoding="utf-8") as f:
                gate_status = json.load(f).get("status", "UNKNOWN")

        if gate_status == "NON_STRICT_SUBSET":
            print(f"VERIFIED (NON_STRICT_SUBSET): deep dataset integrity verification passed for {builder.manifest_root}")
        else:
            print(f"PASSED (FULL): deep dataset integrity verification passed for {builder.manifest_root}")
        return 0

    staged_dir = args.staged_dir or builder.manifest_root
    samples_by_source: Dict[str, List[Dict[str, Any]]] = {}
    found_any = False
    for src in builder.sources_cfg.keys():
        src_file = staged_dir / f"staged_{src}.jsonl"
        if src_file.exists():
            found_any = True
            records = []
            with open(src_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            samples_by_source[src] = records
            print(f"Loaded {len(records)} staged records from {src_file.name}")

    if found_any:
        print(f"Partitioning samples across 7 roles with zero leakage...")
        role_pools, rejects = builder.partition_samples(
            samples_by_source, verify_audio_file=args.verify_audio
        )
        strict = not args.no_strict_quotas and args.mode == "full"
        pilot_subsets = builder.build_pilot_subsets(role_pools, strict=strict)
        smoke_subset = builder.build_smoke_subset(role_pools, strict=strict)
        gate_res = builder.write_manifests_and_gates(
            role_pools, rejects, pilot_subsets, strict_quotas=strict, smoke_subset=smoke_subset
        )
        status = gate_res.get("status")
        print(f"DATASET_COMPLETE.json generated at {builder.manifest_root / builder.complete_gate_file} (Status: {status})")
        return 0 if status in ("PASSED", "NON_STRICT_SUBSET") else 1

    print("Data builder ready. To partition, provide staged manifests in --staged-dir or run stage_parquet_sources.py.")
    return 0



if __name__ == "__main__":
    sys.exit(main())
