#!/usr/bin/env python3
"""Materialize the read-only AliMeeting Eval (OpenSLR 119) paired near/far OOD benchmark.

License: CC BY-SA 4.0 — strictly read-only OOD evaluation (`trainable: false`).
Never mix outputs of this script into training, DPO pair building, or RL rollouts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.zh_cer import NORMALIZATION_VERSION, normalize_zh_cer_v1  # noqa: E402
from train.sft_checkpoint import atomic_write_json, sha256_file, utc_now  # noqa: E402

OOD_DATA_CONFIG_SCHEMA = "qwen3-asr-ood-data-v1"
OOD_MANIFEST_ROW_SCHEMA = "qwen3-asr-ood-manifest-v1"
OOD_MANIFEST_SET_SCHEMA = "qwen3-asr-ood-manifest-set-v1"
OOD_PROCESSING_VERSION = "alimeeting-ood-v1"
OOD_SCENARIOS = ("near_field", "far_field")

_SILENCE_ONLY_RE = re.compile(
    r"^(?:\s|<sil>|\(sil\)|sil|<space>|<_>)+$",
    flags=re.IGNORECASE,
)
_STRIP_TAGS_RE = re.compile(
    r"(?:<sil>|<\s*%\s*>|<->|<\$>|<#>|<_>|<space>|\(~\)|\(sil\)|`|&)",
    flags=re.IGNORECASE,
)
_LATIN_RE = re.compile(r"[A-Za-z\uFF21-\uFF3A\uFF41-\uFF5A]")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_SPEAKER_TOKEN_RE = re.compile(r"(SPK[A-Za-z0-9]+)", flags=re.IGNORECASE)
_FORBIDDEN_SUBSTRINGS = ("language chinese", "<asr_text>", "<|")


@dataclass(frozen=True)
class TgInterval:
    xmin: float
    xmax: float
    text: str


@dataclass(frozen=True)
class TgTier:
    name: str
    xmin: float
    xmax: float
    intervals: tuple[TgInterval, ...]


def _stable_sha256_json(obj: Any) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _det_hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _reject_unknown(mapping: dict[str, Any], allowed: set[str], section: str) -> None:
    extra = sorted(set(mapping) - allowed)
    if extra:
        raise ValueError(f"unknown {section} fields: {extra}")


def load_alimeeting_ood_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    _reject_unknown(
        raw,
        {
            "schema_version",
            "dataset_name",
            "license",
            "usage_policy",
            "seed",
            "archive",
            "audio",
            "alignment",
            "text_filter",
            "selection",
        },
        "top-level",
    )
    if raw.get("schema_version") != OOD_DATA_CONFIG_SCHEMA:
        raise ValueError(f"unsupported schema_version: {raw.get('schema_version')!r}")
    if raw.get("license") != "CC-BY-SA-4.0":
        raise ValueError(f"license must be 'CC-BY-SA-4.0', got {raw.get('license')!r}")
    if raw.get("usage_policy") != "read_only_ood_evaluation":
        raise ValueError("usage_policy must be 'read_only_ood_evaluation'")
    seed = raw.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    archive = raw.get("archive")
    if not isinstance(archive, dict):
        raise ValueError("archive must be a mapping")
    _reject_unknown(archive, {"url", "filename", "expected_size_bytes"}, "archive")
    if not isinstance(archive.get("url"), str) or not archive["url"].startswith("https://"):
        raise ValueError("archive.url must be an https:// URL")
    if not isinstance(archive.get("filename"), str) or not archive["filename"].strip():
        raise ValueError("archive.filename must be a non-empty string")
    exp_size = archive.get("expected_size_bytes")
    if exp_size is not None and (isinstance(exp_size, bool) or not isinstance(exp_size, int) or exp_size <= 0):
        raise ValueError("archive.expected_size_bytes must be a positive integer or null")

    audio = raw.get("audio")
    if not isinstance(audio, dict):
        raise ValueError("audio must be a mapping")
    _reject_unknown(
        audio,
        {
            "sample_rate_hz",
            "far_field_channel",
            "min_duration_sec",
            "max_duration_sec",
            "min_peak_abs",
            "min_rms",
        },
        "audio",
    )
    if int(audio.get("sample_rate_hz", 0)) != 16000:
        raise ValueError("audio.sample_rate_hz must be 16000")
    far_ch = audio.get("far_field_channel")
    if isinstance(far_ch, bool) or not isinstance(far_ch, int) or far_ch < 0:
        raise ValueError("audio.far_field_channel must be a non-negative integer")
    min_dur = float(audio["min_duration_sec"])
    max_dur = float(audio["max_duration_sec"])
    if not (0.2 <= min_dur < max_dur <= 30.0):
        raise ValueError("audio duration bounds must satisfy 0.2 <= min_duration_sec < max_duration_sec <= 30.0")

    alignment = raw.get("alignment")
    if not isinstance(alignment, dict):
        raise ValueError("alignment must be a mapping")
    _reject_unknown(
        alignment,
        {"overlap_eps_sec", "max_near_far_sync_drift_sec", "max_near_far_duration_diff_sec"},
        "alignment",
    )
    if float(alignment["overlap_eps_sec"]) < 0.0:
        raise ValueError("alignment.overlap_eps_sec must be >= 0")

    text_filter = raw.get("text_filter")
    if not isinstance(text_filter, dict):
        raise ValueError("text_filter must be a mapping")
    _reject_unknown(
        text_filter,
        {"normalization", "min_normalized_chars", "reject_latin", "reject_overlap_tags"},
        "text_filter",
    )
    if text_filter.get("normalization") != NORMALIZATION_VERSION:
        raise ValueError(f"text_filter.normalization must be {NORMALIZATION_VERSION!r}")
    if int(text_filter.get("min_normalized_chars", 0)) < 1:
        raise ValueError("text_filter.min_normalized_chars must be >= 1")

    selection = raw.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("selection must be a mapping")
    _reject_unknown(
        selection,
        {"scenarios", "max_paired_sources", "min_paired_sources", "stratify_by"},
        "selection",
    )
    if tuple(selection.get("scenarios", ())) != OOD_SCENARIOS:
        raise ValueError(f"selection.scenarios must be {list(OOD_SCENARIOS)}")
    max_src = int(selection["max_paired_sources"])
    min_src = int(selection["min_paired_sources"])
    if min_src < 1 or max_src < min_src:
        raise ValueError("selection min_paired_sources/max_paired_sources are invalid")
    if selection.get("stratify_by") != "meeting_id":
        raise ValueError("selection.stratify_by must be 'meeting_id'")

    normalized = json.loads(json.dumps(raw))
    normalized["config_sha256"] = _stable_sha256_json(raw)
    return normalized


def decode_textgrid_bytes(data: bytes) -> str:
    if data.startswith(b"\xff\xfe"):
        return data.decode("utf-16-le")
    if data.startswith(b"\xfe\xff"):
        return data.decode("utf-16-be")
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig")
    for enc in ("utf-8", "gb18030", "utf-16"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("unable to decode TextGrid file with utf-8, gb18030, or utf-16")


def _unescape_praat_str(raw: str) -> str:
    return raw.replace('""', '"')


def _parse_short_textgrid(content: str) -> list[TgTier]:
    token_re = re.compile(
        r'"((?:[^"]|"")*)"|([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)|(!.*$)',
        flags=re.MULTILINE,
    )
    tokens: list[tuple[str, str]] = []
    for match in token_re.finditer(content):
        if match.group(3) is not None:
            continue
        if match.group(1) is not None:
            tokens.append(("str", _unescape_praat_str(match.group(1))))
        elif match.group(2) is not None:
            tokens.append(("num", match.group(2)))

    idx = 0
    while idx < len(tokens) and not (tokens[idx][0] == "str" and tokens[idx][1] == "TextGrid"):
        idx += 1
    if idx >= len(tokens):
        raise ValueError("missing 'TextGrid' header in TextGrid file")
    idx += 1  # skip "TextGrid"
    # xmin, xmax, "<exists>", size
    if idx + 4 > len(tokens):
        raise ValueError("truncated short TextGrid header")
    idx += 2  # global xmin, xmax
    if tokens[idx][0] == "str" and "exists" in tokens[idx][1].lower():
        idx += 1
    tier_count = int(float(tokens[idx][1]))
    idx += 1
    tiers: list[TgTier] = []
    for _ in range(tier_count):
        if idx + 5 > len(tokens):
            raise ValueError("truncated tier header in short TextGrid")
        tier_class = tokens[idx][1]
        tier_name = tokens[idx + 1][1]
        t_xmin = float(tokens[idx + 2][1])
        t_xmax = float(tokens[idx + 3][1])
        item_count = int(float(tokens[idx + 4][1]))
        idx += 5
        intervals: list[TgInterval] = []
        if tier_class == "IntervalTier":
            for _ in range(item_count):
                if idx + 3 > len(tokens):
                    raise ValueError("truncated interval in short TextGrid")
                i_xmin = float(tokens[idx][1])
                i_xmax = float(tokens[idx + 1][1])
                i_text = tokens[idx + 2][1]
                idx += 3
                intervals.append(TgInterval(xmin=i_xmin, xmax=i_xmax, text=i_text))
            tiers.append(TgTier(name=tier_name, xmin=t_xmin, xmax=t_xmax, intervals=tuple(intervals)))
        else:
            # TextTier / PointTier has (number, mark) pairs
            idx += 2 * item_count
    return tiers


def parse_praat_textgrid(path: Path) -> list[TgTier]:
    """Parse a Praat .TextGrid file (long or short format) using only Python stdlib."""
    content = decode_textgrid_bytes(path.read_bytes()).replace("\r\n", "\n").replace("\r", "\n")
    if "TextGrid" not in content:
        raise ValueError(f"{path} is not a valid Praat TextGrid")

    item_blocks = re.split(r"^\s*item\s*\[\s*\d+\s*\]\s*:", content, flags=re.MULTILINE)
    if len(item_blocks) <= 1:
        return _parse_short_textgrid(content)

    tiers: list[TgTier] = []
    class_re = re.compile(r'class\s*=\s*"((?:[^"]|"")*)"')
    name_re = re.compile(r'name\s*=\s*"((?:[^"]|"")*)"')
    xmin_re = re.compile(r"xmin\s*=\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)")
    xmax_re = re.compile(r"xmax\s*=\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)")
    text_re = re.compile(r'(?:text|mark)\s*=\s*"((?:[^"]|"")*)"', flags=re.DOTALL)

    for block in item_blocks[1:]:
        cls_m = class_re.search(block)
        if cls_m is None or _unescape_praat_str(cls_m.group(1)) != "IntervalTier":
            continue
        interval_parts = re.split(r"^\s*intervals\s*\[\s*\d+\s*\]\s*:", block, flags=re.MULTILINE)
        tier_header = interval_parts[0]
        name_m = name_re.search(tier_header)
        t_xmin_m = xmin_re.search(tier_header)
        t_xmax_m = xmax_re.search(tier_header)
        if name_m is None or t_xmin_m is None or t_xmax_m is None:
            raise ValueError(f"malformed IntervalTier header in {path}")
        tier_name = _unescape_praat_str(name_m.group(1)).strip()
        intervals: list[TgInterval] = []
        for int_block in interval_parts[1:]:
            ixmin = xmin_re.search(int_block)
            ixmax = xmax_re.search(int_block)
            itext = text_re.search(int_block)
            if ixmin is None or ixmax is None or itext is None:
                raise ValueError(f"malformed interval entry in {path}")
            intervals.append(
                TgInterval(
                    xmin=float(ixmin.group(1)),
                    xmax=float(ixmax.group(1)),
                    text=_unescape_praat_str(itext.group(1)),
                )
            )
        tiers.append(
            TgTier(
                name=tier_name,
                xmin=float(t_xmin_m.group(1)),
                xmax=float(t_xmax_m.group(1)),
                intervals=tuple(intervals),
            )
        )
    return tiers


def is_active_speech_mark(raw_text: str) -> bool:
    stripped = str(raw_text or "").strip()
    if not stripped:
        return False
    if _SILENCE_ONLY_RE.match(stripped):
        return False
    return True


def clean_alimeeting_transcript(
    raw_text: str,
    *,
    min_normalized_chars: int = 2,
    reject_latin: bool = True,
    reject_overlap_tags: bool = True,
) -> tuple[str | None, str | None]:
    """Clean AliMeeting annotation tags and return (cleaned_text, reject_reason)."""
    raw = str(raw_text or "").strip()
    if not is_active_speech_mark(raw):
        return None, "silence_interval"
    if reject_overlap_tags and "/overlap" in raw.lower():
        return None, "overlap_tag"
    cleaned = _STRIP_TAGS_RE.sub("", raw)
    cleaned = re.sub(r"\s+", "", cleaned).strip()
    if not cleaned:
        return None, "empty_after_tag_strip"
    lowered = cleaned.lower()
    for bad in _FORBIDDEN_SUBSTRINGS:
        if bad in lowered:
            return None, "forbidden_prompt_token"
    if reject_latin and _LATIN_RE.search(cleaned):
        return None, "contains_latin_code_switch"
    if not _CJK_RE.search(cleaned):
        return None, "no_cjk_char"
    norm = normalize_zh_cer_v1(cleaned)
    if len(norm) < min_normalized_chars:
        return None, "empty_or_too_short_text"
    return cleaned, None


def extract_meeting_id(stem: str) -> str:
    parts = [p for p in stem.strip().split("_") if p]
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return stem.strip()


def extract_speaker_id(name_or_stem: str) -> str | None:
    match = _SPEAKER_TOKEN_RE.search(name_or_stem or "")
    if match is not None:
        return match.group(1).upper()
    return None


def _has_overlap(
    target_key: Any,
    xmin: float,
    xmax: float,
    active_intervals: list[tuple[Any, float, float]],
    overlap_eps_sec: float,
) -> bool:
    for key, other_min, other_max in active_intervals:
        if key == target_key:
            continue
        if other_min >= xmax - overlap_eps_sec:
            # active_intervals are sorted by other_min
            if other_min > xmax:
                break
        if min(xmax, other_max) - max(xmin, other_min) > overlap_eps_sec:
            return True
    return False


def safe_extract_tar(archive_path: Path, dest_dir: Path) -> None:
    dest_resolved = dest_dir.resolve()
    dest_resolved.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:*") as tf:
        for member in tf:
            member_path = (dest_resolved / member.name).resolve()
            if not str(member_path).startswith(str(dest_resolved) + os.sep) and member_path != dest_resolved:
                raise ValueError(f"unsafe tar path traversal in {archive_path}: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"refusing symlink/hardlink in {archive_path}: {member.name}")
        tf.extractall(dest_resolved)


def resolve_eval_ali_dirs(root: Path) -> tuple[Path, Path]:
    candidates = [root, root / "Eval_Ali"]
    for base in candidates:
        near_dir = base / "Eval_Ali_near"
        far_dir = base / "Eval_Ali_far"
        if near_dir.is_dir() and far_dir.is_dir():
            return near_dir, far_dir
    raise ValueError(f"could not locate Eval_Ali_near and Eval_Ali_far under {root}")


def download_archive(url: str, target_path: Path, expected_size_bytes: int | None) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.is_file():
        if expected_size_bytes is None or target_path.stat().st_size == expected_size_bytes:
            return
    part_path = target_path.with_suffix(target_path.suffix + ".part")
    curl_bin = shutil.which("curl")
    print(f"OOD_DOWNLOAD_START url={url} dst={target_path}", flush=True)
    if curl_bin:
        cmd = [
            curl_bin,
            "-fL",
            "--retry",
            "5",
            "--retry-delay",
            "5",
            "-C",
            "-",
            "-o",
            str(part_path),
            url,
        ]
        rc = subprocess.call(cmd)
        if rc != 0:
            raise ValueError(f"curl download failed with exit code {rc}")
    else:
        with urllib.request.urlopen(url, timeout=120) as resp, part_path.open("wb") as out:
            shutil.copyfileobj(resp, out, length=1024 * 1024)
    actual_size = part_path.stat().st_size
    if expected_size_bytes is not None and actual_size != expected_size_bytes:
        raise ValueError(f"downloaded archive size mismatch: {actual_size} != {expected_size_bytes}")
    part_path.replace(target_path)
    print(f"OOD_DOWNLOAD_DONE size={actual_size} dst={target_path}", flush=True)


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    ordered = sorted(values)
    n = len(ordered)

    def pick(q: float) -> float:
        pos = min(n - 1, max(0, int(round(q * (n - 1)))))
        return round(float(ordered[pos]), 4)

    return {
        "min": round(float(ordered[0]), 4),
        "p50": pick(0.50),
        "p90": pick(0.90),
        "max": round(float(ordered[-1]), 4),
    }


def _build_sha256_index(output_dir: Path, rel_paths: list[str]) -> dict[str, Any]:
    files_map: dict[str, str] = {}
    entries: list[dict[str, Any]] = []
    for rel in sorted(rel_paths):
        full = output_dir / rel
        digest = sha256_file(full)
        size = int(full.stat().st_size)
        files_map[rel] = digest
        entries.append({"name": rel, "sha256": digest, "bytes": size})
    return {
        "schema_version": "qwen3-asr-ood-sha256-v1",
        "generated_at": utc_now(),
        "files": files_map,
        "entries": entries,
    }


def try_restore_from_drive_cache(
    drive_cache_dir: Path,
    output_dir: Path,
    *,
    expected_config_sha256: str,
    expected_seed: int,
) -> bool:
    required = ("manifest.json", "ood_eval.jsonl", "rejected.jsonl", "sha256.json", "ood_audio.tar")
    if not all((drive_cache_dir / name).is_file() for name in required):
        return False
    cached_manifest = json.loads((drive_cache_dir / "manifest.json").read_text(encoding="utf-8"))
    if cached_manifest.get("config_sha256") != expected_config_sha256:
        return False
    if int(cached_manifest.get("seed", -1)) != int(expected_seed):
        return False
    if cached_manifest.get("status") != "passed" or cached_manifest.get("trainable") is not False:
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    for meta_name in ("manifest.json", "ood_eval.jsonl", "rejected.jsonl", "sha256.json"):
        shutil.copyfile(drive_cache_dir / meta_name, output_dir / meta_name)
    safe_extract_tar(drive_cache_dir / "ood_audio.tar", output_dir)

    sha_payload = json.loads((output_dir / "sha256.json").read_text(encoding="utf-8"))
    files_map = sha_payload.get("files", {})
    if not isinstance(files_map, dict) or not files_map:
        return False
    for rel, exp_sha in sorted(files_map.items()):
        target = output_dir / rel
        if not target.is_file() or sha256_file(target) != exp_sha:
            raise ValueError(f"Drive OOD cache verification failed for {rel}")
    print(f"OOD_CACHE_RESTORED dir={drive_cache_dir} files={len(files_map)}", flush=True)
    return True


def persist_to_drive_cache(output_dir: Path, drive_cache_dir: Path, wav_rel_paths: list[str]) -> str:
    drive_cache_dir.mkdir(parents=True, exist_ok=True)
    tar_tmp = output_dir / ".ood_audio.tar.tmp"
    with tarfile.open(tar_tmp, "w") as tf:
        for rel in sorted(wav_rel_paths):
            tf.add(output_dir / rel, arcname=rel)
    tar_sha = sha256_file(tar_tmp)
    dst_tar = drive_cache_dir / "ood_audio.tar"
    shutil.copyfile(tar_tmp, dst_tar)
    tar_tmp.unlink(missing_ok=True)
    if sha256_file(dst_tar) != tar_sha:
        raise ValueError("Drive readback sha256 mismatch for ood_audio.tar")

    for name in ("manifest.json", "ood_eval.jsonl", "rejected.jsonl", "sha256.json"):
        src = output_dir / name
        dst = drive_cache_dir / name
        shutil.copyfile(src, dst)
        if sha256_file(src) != sha256_file(dst):
            raise ValueError(f"Drive readback sha256 mismatch for {name}")
    print(f"OOD_CACHE_PERSISTED dir={drive_cache_dir} wavs={len(wav_rel_paths)}", flush=True)
    return "synced"


def prepare_alimeeting_ood(
    *,
    config_path: Path,
    output_dir: Path,
    seed: int,
    extracted_root: Path | None = None,
    archive_path: Path | None = None,
    download_if_missing: bool = False,
    drive_cache_dir: Path | None = None,
) -> tuple[int, dict[str, Any]]:
    import numpy as np
    import soundfile as sf

    cfg = load_alimeeting_ood_config(config_path)
    if int(seed) != int(cfg["seed"]):
        raise ValueError(f"CLI --seed {seed} does not match config seed {cfg['seed']}")

    if drive_cache_dir is not None and try_restore_from_drive_cache(
        drive_cache_dir,
        output_dir,
        expected_config_sha256=cfg["config_sha256"],
        expected_seed=seed,
    ):
        restored = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
        return 0, restored

    temp_extract_ctx: tempfile.TemporaryDirectory[str] | None = None
    archive_meta: dict[str, Any] | None = None
    try:
        if extracted_root is not None:
            if not extracted_root.is_dir():
                raise ValueError(f"--extracted-root does not exist: {extracted_root}")
            raw_root = extracted_root
        else:
            resolved_archive = archive_path
            if resolved_archive is None:
                raise ValueError("either --extracted-root or --archive-path must be provided")
            if not resolved_archive.is_file():
                if not download_if_missing:
                    raise ValueError(
                        f"archive {resolved_archive} does not exist and --download-if-missing is not set"
                    )
                download_archive(
                    str(cfg["archive"]["url"]),
                    resolved_archive,
                    cfg["archive"].get("expected_size_bytes"),
                )
            exp_bytes = cfg["archive"].get("expected_size_bytes")
            actual_bytes = int(resolved_archive.stat().st_size)
            if exp_bytes is not None and actual_bytes != int(exp_bytes):
                raise RuntimeError(f"archive byte size mismatch: {actual_bytes} != {exp_bytes}")
            archive_meta = {
                "filename": resolved_archive.name,
                "bytes": actual_bytes,
                "sha256": sha256_file(resolved_archive),
            }
            temp_extract_ctx = tempfile.TemporaryDirectory(prefix="alimeeting_extract_")
            raw_root = Path(temp_extract_ctx.name)
            print(f"OOD_EXTRACT_START archive={resolved_archive} dst={raw_root}", flush=True)
            safe_extract_tar(resolved_archive, raw_root)
            print("OOD_EXTRACT_DONE", flush=True)

        near_dir, far_dir = resolve_eval_ali_dirs(raw_root)
        near_audio_dir = near_dir / "audio_dir"
        near_tg_dir = near_dir / "textgrid_dir"
        far_audio_dir = far_dir / "audio_dir"
        far_tg_dir = far_dir / "textgrid_dir"
        for required_dir in (near_audio_dir, near_tg_dir, far_audio_dir, far_tg_dir):
            if not required_dir.is_dir():
                raise RuntimeError(f"missing required AliMeeting directory: {required_dir}")

        near_wavs = sorted(p for p in near_audio_dir.iterdir() if p.is_file() and p.suffix.lower() == ".wav")
        near_tgs = {
            p.stem: p
            for p in sorted(near_tg_dir.iterdir())
            if p.is_file() and p.suffix.lower() == ".textgrid"
        }
        far_wavs_by_meeting: dict[str, list[Path]] = {}
        for p in sorted(far_audio_dir.iterdir()):
            if p.is_file() and p.suffix.lower() == ".wav":
                far_wavs_by_meeting.setdefault(extract_meeting_id(p.stem), []).append(p)
        far_tgs_by_meeting: dict[str, Path] = {}
        for p in sorted(far_tg_dir.iterdir()):
            if p.is_file() and p.suffix.lower() == ".textgrid":
                far_tgs_by_meeting.setdefault(extract_meeting_id(p.stem), p)
                far_tgs_by_meeting.setdefault(p.stem, p)

        near_by_meeting: dict[str, list[Path]] = {}
        for wav_p in near_wavs:
            near_by_meeting.setdefault(extract_meeting_id(wav_p.stem), []).append(wav_p)

        meetings = sorted(set(near_by_meeting) | set(far_wavs_by_meeting))
        if not meetings:
            raise RuntimeError("no meeting WAV files found in Eval_Ali_near / Eval_Ali_far")

        overlap_eps = float(cfg["alignment"]["overlap_eps_sec"])
        max_sync_drift = float(cfg["alignment"]["max_near_far_sync_drift_sec"])
        max_dur_diff = float(cfg["alignment"]["max_near_far_duration_diff_sec"])
        min_dur = float(cfg["audio"]["min_duration_sec"])
        max_dur = float(cfg["audio"]["max_duration_sec"])
        min_peak = float(cfg["audio"]["min_peak_abs"])
        min_rms = float(cfg["audio"]["min_rms"])
        far_channel = int(cfg["audio"]["far_field_channel"])
        min_norm_chars = int(cfg["text_filter"]["min_normalized_chars"])
        reject_latin = bool(cfg["text_filter"]["reject_latin"])
        reject_overlap_tags = bool(cfg["text_filter"]["reject_overlap_tags"])

        rejected_rows: list[dict[str, Any]] = []
        rejected_counts: dict[str, int] = {}

        def record_reject(meeting_id: str, speaker_id: str, reason: str, **extra: Any) -> None:
            rejected_counts[reason] = rejected_counts.get(reason, 0) + 1
            rejected_rows.append(
                {
                    "meeting_id": meeting_id,
                    "speaker_id": speaker_id,
                    "reason": reason,
                    "processing_version": OOD_PROCESSING_VERSION,
                    **extra,
                }
            )

        candidates_by_meeting: dict[str, list[dict[str, Any]]] = {}
        seen_speakers: set[str] = set()
        total_near_intervals = 0

        for meeting_id in meetings:
            m_near_wavs = near_by_meeting.get(meeting_id, [])
            m_far_wavs = far_wavs_by_meeting.get(meeting_id, [])
            if not m_near_wavs or not m_far_wavs:
                record_reject(meeting_id, "UNKNOWN", "missing_meeting_near_or_far_wav")
                continue
            far_wav_path = m_far_wavs[0]
            far_tg_path = far_tgs_by_meeting.get(meeting_id) or far_tgs_by_meeting.get(far_wav_path.stem)
            if far_tg_path is None:
                record_reject(meeting_id, "UNKNOWN", "missing_far_textgrid")
                continue

            far_tiers = parse_praat_textgrid(far_tg_path)
            far_active: list[tuple[tuple[int, int], float, float]] = []
            far_Lookup: list[dict[str, Any]] = []
            for t_idx, tier in enumerate(far_tiers):
                tier_spk = extract_speaker_id(tier.name)
                for i_idx, iv in enumerate(tier.intervals):
                    if not is_active_speech_mark(iv.text):
                        continue
                    key = (t_idx, i_idx)
                    far_active.append((key, iv.xmin, iv.xmax))
                    cleaned_far, far_reason = clean_alimeeting_transcript(
                        iv.text,
                        min_normalized_chars=min_norm_chars,
                        reject_latin=reject_latin,
                        reject_overlap_tags=reject_overlap_tags,
                    )
                    far_Lookup.append(
                        {
                            "key": key,
                            "tier_name": tier.name,
                            "speaker_id": tier_spk,
                            "xmin": iv.xmin,
                            "xmax": iv.xmax,
                            "raw_text": iv.text,
                            "cleaned_text": cleaned_far,
                            "reject_reason": far_reason,
                        }
                    )
            far_active.sort(key=lambda item: (item[1], item[2]))

            near_speaker_entries: list[tuple[str, Path, list[TgInterval]]] = []
            near_active: list[tuple[tuple[str, int], float, float]] = []
            for n_wav in m_near_wavs:
                tg_p = near_tgs.get(n_wav.stem)
                spk_id = extract_speaker_id(n_wav.stem) or n_wav.stem.split("_")[-1].upper()
                if tg_p is None:
                    record_reject(meeting_id, spk_id, "missing_near_textgrid", wav=n_wav.name)
                    continue
                n_tiers = parse_praat_textgrid(tg_p)
                merged_ivs: list[TgInterval] = []
                for nt in n_tiers:
                    merged_ivs.extend(nt.intervals)
                merged_ivs.sort(key=lambda x: (x.xmin, x.xmax))
                for idx_iv, iv in enumerate(merged_ivs):
                    if is_active_speech_mark(iv.text):
                        near_active.append(((n_wav.stem, idx_iv), iv.xmin, iv.xmax))
                near_speaker_entries.append((spk_id, n_wav, merged_ivs))
            near_active.sort(key=lambda item: (item[1], item[2]))

            for spk_id, n_wav, merged_ivs in near_speaker_entries:
                seen_speakers.add(f"{meeting_id}:{spk_id}")
                for idx_iv, iv in enumerate(merged_ivs):
                    if not is_active_speech_mark(iv.text):
                        continue
                    total_near_intervals += 1
                    near_key = (n_wav.stem, idx_iv)
                    dur = float(iv.xmax - iv.xmin)
                    cleaned_text, text_reason = clean_alimeeting_transcript(
                        iv.text,
                        min_normalized_chars=min_norm_chars,
                        reject_latin=reject_latin,
                        reject_overlap_tags=reject_overlap_tags,
                    )
                    if text_reason is not None or cleaned_text is None:
                        record_reject(
                            meeting_id,
                            spk_id,
                            text_reason or "invalid_text",
                            xmin=round(iv.xmin, 3),
                            xmax=round(iv.xmax, 3),
                        )
                        continue
                    if dur < min_dur or dur > max_dur:
                        record_reject(
                            meeting_id,
                            spk_id,
                            "duration_out_of_range",
                            duration_sec=round(dur, 4),
                            xmin=round(iv.xmin, 3),
                            xmax=round(iv.xmax, 3),
                        )
                        continue
                    if _has_overlap(near_key, iv.xmin, iv.xmax, near_active, overlap_eps):
                        record_reject(
                            meeting_id,
                            spk_id,
                            "speaker_overlap",
                            timeline="near_field",
                            xmin=round(iv.xmin, 3),
                            xmax=round(iv.xmax, 3),
                        )
                        continue

                    matches = [
                        item
                        for item in far_Lookup
                        if (item["speaker_id"] == spk_id or item["speaker_id"] is None)
                        and item["cleaned_text"] == cleaned_text
                        and abs(float(item["xmin"]) - iv.xmin) <= max_sync_drift
                        and abs((float(item["xmax"]) - float(item["xmin"])) - dur) <= max_dur_diff
                    ]
                    if any(m["speaker_id"] == spk_id for m in matches):
                        matches = [m for m in matches if m["speaker_id"] == spk_id]
                    if len(matches) != 1:
                        record_reject(
                            meeting_id,
                            spk_id,
                            "unpaired_far_interval",
                            match_count=len(matches),
                            xmin=round(iv.xmin, 3),
                            xmax=round(iv.xmax, 3),
                        )
                        continue
                    far_item = matches[0]
                    if _has_overlap(
                        far_item["key"],
                        float(far_item["xmin"]),
                        float(far_item["xmax"]),
                        far_active,
                        overlap_eps,
                    ):
                        record_reject(
                            meeting_id,
                            spk_id,
                            "speaker_overlap",
                            timeline="far_field",
                            xmin=round(float(far_item["xmin"]), 3),
                            xmax=round(float(far_item["xmax"]), 3),
                        )
                        continue

                    start_cs = int(round(iv.xmin * 100.0))
                    end_cs = int(round(iv.xmax * 100.0))
                    source_id = f"alimeeting_{meeting_id}_{spk_id}_{start_cs:07d}_{end_cs:07d}"
                    candidates_by_meeting.setdefault(meeting_id, []).append(
                        {
                            "source_utterance_id": source_id,
                            "meeting_id": meeting_id,
                            "speaker_id": spk_id,
                            "text": cleaned_text,
                            "near_wav": n_wav,
                            "near_xmin": iv.xmin,
                            "near_xmax": iv.xmax,
                            "far_wav": far_wav_path,
                            "far_xmin": float(far_item["xmin"]),
                            "far_xmax": float(far_item["xmax"]),
                        }
                    )

        # Deduplicate any identical source_utterance_id deterministically and rank per meeting
        ranked_by_meeting: dict[str, list[dict[str, Any]]] = {}
        seen_source_ids: set[str] = set()
        for meeting_id in sorted(candidates_by_meeting):
            unique_cands: list[dict[str, Any]] = []
            for cand in sorted(candidates_by_meeting[meeting_id], key=lambda c: c["source_utterance_id"]):
                sid = cand["source_utterance_id"]
                if sid in seen_source_ids:
                    record_reject(meeting_id, cand["speaker_id"], "duplicate_source_id", source_utterance_id=sid)
                    continue
                seen_source_ids.add(sid)
                unique_cands.append(cand)
            unique_cands.sort(
                key=lambda c: (
                    _det_hash_text(f"{seed}:{c['source_utterance_id']}:{OOD_PROCESSING_VERSION}"),
                    c["source_utterance_id"],
                )
            )
            ranked_by_meeting[meeting_id] = unique_cands

        # Stratified round-robin slicing with audio quality check so rejected silent/OOB clips get backfilled
        output_dir.mkdir(parents=True, exist_ok=True)
        for sc in OOD_SCENARIOS:
            (output_dir / "wavs" / sc).mkdir(parents=True, exist_ok=True)

        max_paired = int(cfg["selection"]["max_paired_sources"])
        min_paired = int(cfg["selection"]["min_paired_sources"])
        available_candidates = sum(len(v) for v in ranked_by_meeting.values())

        cursors = {m: 0 for m in sorted(ranked_by_meeting)}
        selected_sources: list[dict[str, Any]] = []
        manifest_rows: list[dict[str, Any]] = []
        wav_rel_paths: list[str] = []
        durations_by_scenario: dict[str, list[float]] = {sc: [] for sc in OOD_SCENARIOS}
        per_meeting_selected: dict[str, int] = {}

        open_sf_cache: dict[Path, Any] = {}
        try:
            def get_sf(wav_p: Path) -> Any:
                handle = open_sf_cache.get(wav_p)
                if handle is None:
                    handle = sf.SoundFile(str(wav_p), mode="r")
                    open_sf_cache[wav_p] = handle
                return handle

            made_progress = True
            while len(selected_sources) < max_paired and made_progress:
                made_progress = False
                for meeting_id in sorted(ranked_by_meeting):
                    if len(selected_sources) >= max_paired:
                        break
                    bucket = ranked_by_meeting[meeting_id]
                    while cursors[meeting_id] < len(bucket):
                        cand = bucket[cursors[meeting_id]]
                        cursors[meeting_id] += 1
                        near_sf = get_sf(cand["near_wav"])
                        far_sf = get_sf(cand["far_wav"])
                        if near_sf.samplerate != 16000 or far_sf.samplerate != 16000:
                            record_reject(
                                meeting_id,
                                cand["speaker_id"],
                                "bad_sample_rate",
                                near_sr=near_sf.samplerate,
                                far_sr=far_sf.samplerate,
                            )
                            continue
                        if far_channel >= far_sf.channels:
                            record_reject(
                                meeting_id,
                                cand["speaker_id"],
                                "far_channel_out_of_range",
                                far_channels=far_sf.channels,
                            )
                            continue
                        n_start = int(round(cand["near_xmin"] * 16000))
                        n_end = int(round(cand["near_xmax"] * 16000))
                        f_start = int(round(cand["far_xmin"] * 16000))
                        f_end = int(round(cand["far_xmax"] * 16000))
                        if not (0 <= n_start < n_end <= near_sf.frames and 0 <= f_start < f_end <= far_sf.frames):
                            record_reject(meeting_id, cand["speaker_id"], "slice_out_of_wav_bounds")
                            continue

                        near_sf.seek(n_start)
                        near_arr = near_sf.read(n_end - n_start, dtype="float32", always_2d=True)[:, 0]
                        far_sf.seek(f_start)
                        far_arr = far_sf.read(f_end - f_start, dtype="float32", always_2d=True)[:, far_channel]

                        if not (np.isfinite(near_arr).all() and np.isfinite(far_arr).all()):
                            record_reject(meeting_id, cand["speaker_id"], "non_finite_waveform")
                            continue
                        near_peak = float(np.max(np.abs(near_arr))) if near_arr.size else 0.0
                        far_peak = float(np.max(np.abs(far_arr))) if far_arr.size else 0.0
                        near_rms = float(math.sqrt(float(np.mean(np.square(near_arr))))) if near_arr.size else 0.0
                        far_rms = float(math.sqrt(float(np.mean(np.square(far_arr))))) if far_arr.size else 0.0
                        if near_peak < min_peak or far_peak < min_peak or near_rms < min_rms or far_rms < min_rms:
                            record_reject(
                                meeting_id,
                                cand["speaker_id"],
                                "silent_or_low_energy_slice",
                                near_rms=round(near_rms, 6),
                                far_rms=round(far_rms, 6),
                            )
                            continue

                        sid = cand["source_utterance_id"]
                        near_sample_id = f"{sid}__near_field"
                        far_sample_id = f"{sid}__far_field"
                        near_rel = f"wavs/near_field/{near_sample_id}.wav"
                        far_rel = f"wavs/far_field/{far_sample_id}.wav"
                        sf.write(output_dir / near_rel, np.clip(near_arr, -1.0, 1.0), 16000, subtype="PCM_16", format="WAV")
                        sf.write(output_dir / far_rel, np.clip(far_arr, -1.0, 1.0), 16000, subtype="PCM_16", format="WAV")

                        near_dur = round(float(near_arr.shape[0]) / 16000.0, 4)
                        far_dur = round(float(far_arr.shape[0]) / 16000.0, 4)
                        durations_by_scenario["near_field"].append(near_dur)
                        durations_by_scenario["far_field"].append(far_dur)
                        wav_rel_paths.extend([near_rel, far_rel])

                        manifest_rows.append(
                            {
                                "schema_version": OOD_MANIFEST_ROW_SCHEMA,
                                "sample_id": near_sample_id,
                                "source_utterance_id": sid,
                                "audio": near_rel,
                                "text": cand["text"],
                                "language": "zh",
                                "scenario": "near_field",
                                "split": "ood_eval",
                                "meeting_id": meeting_id,
                                "speaker_id": f"{meeting_id}_{cand['speaker_id']}",
                                "duration_sec": near_dur,
                                "far_field_channel": None,
                                "license": "CC-BY-SA-4.0",
                            }
                        )
                        manifest_rows.append(
                            {
                                "schema_version": OOD_MANIFEST_ROW_SCHEMA,
                                "sample_id": far_sample_id,
                                "source_utterance_id": sid,
                                "audio": far_rel,
                                "text": cand["text"],
                                "language": "zh",
                                "scenario": "far_field",
                                "split": "ood_eval",
                                "meeting_id": meeting_id,
                                "speaker_id": f"{meeting_id}_{cand['speaker_id']}",
                                "duration_sec": far_dur,
                                "far_field_channel": far_channel,
                                "license": "CC-BY-SA-4.0",
                            }
                        )
                        selected_sources.append(cand)
                        per_meeting_selected[meeting_id] = per_meeting_selected.get(meeting_id, 0) + 1
                        made_progress = True
                        break
        finally:
            for handle in open_sf_cache.values():
                handle.close()

        manifest_rows.sort(key=lambda r: (r["source_utterance_id"], 0 if r["scenario"] == "near_field" else 1))
        ood_jsonl_path = output_dir / "ood_eval.jsonl"
        rejected_jsonl_path = output_dir / "rejected.jsonl"
        with ood_jsonl_path.open("w", encoding="utf-8") as h:
            for row in manifest_rows:
                h.write(json.dumps(row, ensure_ascii=False) + "\n")
        with rejected_jsonl_path.open("w", encoding="utf-8") as h:
            for row in rejected_rows:
                h.write(json.dumps(row, ensure_ascii=False) + "\n")

        passed_quota = len(selected_sources) >= min_paired
        status_str = "passed" if passed_quota else "insufficient_data"
        manifest_summary: dict[str, Any] = {
            "schema_version": OOD_MANIFEST_SET_SCHEMA,
            "dataset_name": cfg["dataset_name"],
            "license": cfg["license"],
            "usage_policy": cfg["usage_policy"],
            "trainable": False,
            "processing_version": OOD_PROCESSING_VERSION,
            "config_sha256": cfg["config_sha256"],
            "seed": int(seed),
            "status": status_str,
            "archive": archive_meta,
            "meetings_discovered": len(meetings),
            "meetings_with_selected_samples": len(per_meeting_selected),
            "speakers_discovered": len(seen_speakers),
            "total_active_near_intervals": total_near_intervals,
            "available_paired_candidates_pre_audio": available_candidates,
            "selected_paired_sources": len(selected_sources),
            "total_ood_rows": len(manifest_rows),
            "far_field_channel": far_channel,
            "per_meeting_selected_sources": dict(sorted(per_meeting_selected.items())),
            "by_scenario": {
                sc: {
                    "rows": len(durations_by_scenario[sc]),
                    "total_duration_sec": round(sum(durations_by_scenario[sc]), 4),
                    "duration_quantiles_sec": _quantiles(durations_by_scenario[sc]),
                }
                for sc in OOD_SCENARIOS
            },
            "rejected_total": len(rejected_rows),
            "rejected_by_reason": dict(sorted(rejected_counts.items())),
        }
        atomic_write_json(output_dir / "manifest.json", manifest_summary)

        all_tracked_rels = ["ood_eval.jsonl", "rejected.jsonl", "manifest.json", *wav_rel_paths]
        sha_index = _build_sha256_index(output_dir, all_tracked_rels)
        atomic_write_json(output_dir / "sha256.json", sha_index)

        if passed_quota and drive_cache_dir is not None:
            persist_to_drive_cache(output_dir, drive_cache_dir, wav_rel_paths)

        print(
            f"OOD_PREPARE_DONE status={status_str} sources={len(selected_sources)} "
            f"rows={len(manifest_rows)} meetings={len(per_meeting_selected)} rejected={len(rejected_rows)}",
            flush=True,
        )
        return (0 if passed_quota else 3), manifest_summary
    finally:
        if temp_extract_ctx is not None:
            temp_extract_ctx.cleanup()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="构建 AliMeeting Eval 只读近场/远场配对 OOD 评测集")
    parser.add_argument("--config", default=Path("configs/data/alimeeting_ood_eval.yaml"), type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--extracted-root", type=Path)
    parser.add_argument("--archive-path", type=Path)
    parser.add_argument("--download-if-missing", action="store_true")
    parser.add_argument("--drive-cache-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        code, _summary = prepare_alimeeting_ood(
            config_path=args.config,
            output_dir=args.output_dir,
            seed=args.seed,
            extracted_root=args.extracted_root,
            archive_path=args.archive_path,
            download_if_missing=bool(args.download_if_missing),
            drive_cache_dir=args.drive_cache_dir,
        )
        return code
    except ValueError as exc:
        print(f"OOD_PREPARE_CONFIG_FAIL {exc}", file=sys.stderr, flush=True)
        return 2
    except (OSError, RuntimeError, tarfile.TarError, json.JSONDecodeError) as exc:
        print(f"OOD_PREPARE_FAIL {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
