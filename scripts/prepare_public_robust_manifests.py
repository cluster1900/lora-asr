#!/usr/bin/env python3
"""Prepare and validate the fixed public-data manifests for fast A2S training.

The tool deliberately separates remote metadata inspection from audio
materialization. ``probe`` only asks ``datasets`` for builder metadata; ``build``
and ``smoke`` consume local candidate JSONL files produced by the Colab staging
step and never call the Hub.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


LANGUAGES = ("en", "zh")
CONDITION_GROUPS = {"clean", "atomic", "compound"}
AUDIO_ORIGINS = {"clean", "real", "synthetic"}
DERIVED_ROLES = {"canary", "curriculum"}
PINNED_REVISION_LENGTH = 40


class ConfigError(RuntimeError):
    """Raised when the fixed data config is incomplete or inconsistent."""


class ManifestError(RuntimeError):
    """Raised when candidate or canonical manifest data fails a hard gate."""


def load_config(path: Path) -> dict[str, Any]:
    """Load JSON with stdlib or YAML with an explicit PyYAML dependency."""
    if not path.exists():
        raise ConfigError(f"config does not exist: {path}")
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ConfigError(
                "YAML config requires PyYAML; install it with `pip install PyYAML` "
                "or pass an equivalent JSON config."
            ) from exc
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError("config root must be a mapping")
    return data


def require_mapping(value: Any, name: str) -> dict[str, Any]:
    """Return a mutable mapping or raise a readable config error."""
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a mapping")
    return value


def config_section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    """Read one required top-level config section."""
    return require_mapping(config.get(name), name)


def assert_pinned_revision(revision: Any, source_name: str) -> str:
    """Require a full hexadecimal Hub commit rather than a moving branch."""
    value = str(revision or "")
    if len(value) != PINNED_REVISION_LENGTH or any(
        char not in "0123456789abcdef" for char in value.lower()
    ):
        raise ConfigError(
            f"sources.{source_name}.revision must be a 40-character commit hash"
        )
    return value


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    """Yield JSON objects with source line numbers."""
    if not path.exists():
        raise ManifestError(f"JSONL does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ManifestError(f"{path}:{line_number}: row must be a JSON object")
            yield line_number, row


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into memory."""
    return [row for _, row in iter_jsonl(path)]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Atomically write JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    """Atomically write formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(data), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    """Hash a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(seed: int, namespace: str, identity: Any) -> str:
    """Return a cross-process deterministic ordering key."""
    payload = f"{seed}\0{namespace}\0{identity}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_order(
    rows: Iterable[dict[str, Any]], seed: int, namespace: str
) -> list[dict[str, Any]]:
    """Order rows reproducibly by sample identity."""
    return sorted(
        rows,
        key=lambda row: (
            stable_key(seed, namespace, row.get("sample_id", "")),
            str(row.get("sample_id", "")),
        ),
    )


def distribute_total(names: Sequence[str], total: int) -> dict[str, int]:
    """Distribute a total over sorted names, assigning remainder to the front."""
    ordered = sorted(str(name) for name in names)
    if not ordered:
        if total:
            raise ConfigError("cannot distribute a non-zero total over no names")
        return {}
    if len(set(ordered)) != len(ordered):
        raise ConfigError("quota names must be unique")
    if total < 0:
        raise ConfigError("quota total must be non-negative")
    base, remainder = divmod(total, len(ordered))
    return {
        name: base + int(index < remainder)
        for index, name in enumerate(ordered)
    }


def plan_scenario_quotas(
    split_names: Sequence[str],
    atomic_splits: Sequence[str],
    atomic_per_split: int,
    compound_total: int,
    expected_compound_splits: int = 47,
) -> dict[str, int]:
    """Build the fixed atomic plus sorted-remainder compound quota map."""
    all_splits = {str(name) for name in split_names}
    atomics = {str(name) for name in atomic_splits}
    missing = sorted(atomics - all_splits)
    if missing:
        raise ConfigError(f"missing atomic splits: {', '.join(missing)}")
    compounds = sorted(all_splits - atomics)
    if len(compounds) != expected_compound_splits:
        raise ConfigError(
            f"compound split count {len(compounds)} != expected {expected_compound_splits}"
        )
    quotas = {name: int(atomic_per_split) for name in sorted(atomics)}
    quotas.update(distribute_total(compounds, int(compound_total)))
    return quotas


def plan_language_quotas(scenario_quotas: Mapping[str, int]) -> dict[tuple[str, str], int]:
    """Split each scenario as evenly as possible across English and Chinese."""
    result: dict[tuple[str, str], int] = {}
    for index, scenario in enumerate(sorted(scenario_quotas)):
        total = int(scenario_quotas[scenario])
        half, remainder = divmod(total, 2)
        en_extra = int(bool(remainder) and index % 2 == 0)
        result[(scenario, "en")] = half + en_extra
        result[(scenario, "zh")] = total - half - en_extra
    return result


def infer_language(text: str) -> str | None:
    """Infer a conservative en/zh label from a gold transcript."""
    chinese = 0
    latin = 0
    for char in text:
        codepoint = ord(char)
        if (
            0x3400 <= codepoint <= 0x4DBF
            or 0x4E00 <= codepoint <= 0x9FFF
            or 0xF900 <= codepoint <= 0xFAFF
        ):
            chinese += 1
        elif "LATIN" in unicodedata.name(char, ""):
            latin += 1
    if chinese and chinese >= latin:
        return "zh"
    if latin and not chinese:
        return "en"
    return None


def source_identity(row: Mapping[str, Any]) -> str:
    """Return the required source utterance identity."""
    value = row.get("source_utterance_id") or row.get("name")
    if value not in (None, ""):
        return str(value)
    file_name = row.get("file_name")
    if file_name:
        return Path(str(file_name)).stem
    raise ManifestError("source identity missing (need source_utterance_id, name, or file_name)")


def audio_path_value(row: Mapping[str, Any]) -> str:
    """Extract a materialized audio path from common candidate representations."""
    value = row.get("audio")
    if isinstance(value, str) and value:
        return value
    if isinstance(value, Mapping) and value.get("path"):
        return str(value["path"])
    if row.get("file_name"):
        return str(row["file_name"])
    raise ManifestError("materialized audio path missing")


def finite_float(value: Any, field: str) -> float:
    """Parse a finite numeric value."""
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"{field} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ManifestError(f"{field} must be finite")
    return parsed


def normalize_candidate(
    row: Mapping[str, Any],
    *,
    source_config: Mapping[str, Any],
    source_kind: str,
    atomic_splits: Sequence[str],
    seed: int,
    fallback_index: int,
) -> dict[str, Any]:
    """Convert one staged source row into the canonical project schema."""
    answer = str(row.get("answer") or row.get("text") or "").strip()
    if not answer:
        raise ManifestError("answer is empty")
    language = str(row.get("language") or "").lower()
    if language not in LANGUAGES:
        language = infer_language(answer) or ""
    if language not in LANGUAGES:
        raise ManifestError("language is mixed or could not be inferred")

    source_split = str(
        row.get("source_split") or row.get("subset") or row.get("split") or ""
    )
    scenario = str(row.get("scenario") or source_split or "clean")
    if not source_split:
        source_split = scenario
    if source_kind == "clean":
        condition_group = "clean"
        scenario = "clean"
    elif source_kind == "robust":
        condition_group = "atomic" if scenario in set(atomic_splits) else "compound"
    else:
        condition_group = str(row.get("condition_group") or "atomic")

    source_dataset = str(
        row.get("source_dataset") or source_config.get("dataset_id") or ""
    )
    source_revision = str(
        row.get("source_revision") or source_config.get("revision") or ""
    )
    source_index = row.get("source_index", row.get("index", fallback_index))
    identity = source_identity(row)
    sample_id = str(
        row.get("sample_id")
        or f"{source_dataset}:{source_split}:{source_index}"
    )
    duration = finite_float(
        row.get("duration_s", row.get("duration")), "duration_s"
    )
    default_origin = "clean" if source_kind == "clean" else "synthetic"
    origin = str(row.get("audio_origin") or row.get("origin") or default_origin).lower()
    if origin == "syn":
        origin = "synthetic"

    canonical: dict[str, Any] = {
        "sample_id": sample_id,
        "audio": audio_path_value(row),
        "answer": answer,
        "language": language,
        "scenario": scenario,
        "condition_group": condition_group,
        "audio_origin": origin,
        "source_dataset": source_dataset,
        "source_revision": source_revision,
        "source_split": source_split,
        "source_index": source_index,
        "source_utterance_id": identity,
        "speaker_id": row.get("speaker_id"),
        "duration_s": duration,
        "license": str(row.get("license") or source_config.get("license") or ""),
        "seed": int(row.get("seed", seed)),
        "audio_sha256": str(row.get("audio_sha256") or ""),
    }
    source_name = row.get("source_name", row.get("name"))
    if source_name not in (None, ""):
        canonical["source_name"] = str(source_name)
    if row.get("benchmark_id") not in (None, ""):
        canonical["benchmark_id"] = str(row["benchmark_id"])
    return canonical


def load_candidates(
    path: Path,
    *,
    source_config: Mapping[str, Any],
    source_kind: str,
    atomic_splits: Sequence[str],
    seed: int,
    min_duration_s: float,
    max_duration_s: float,
    rejects: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize staged candidates while recording row-level rejects."""
    rows: list[dict[str, Any]] = []
    for line_number, source_row in iter_jsonl(path):
        try:
            row = normalize_candidate(
                source_row,
                source_config=source_config,
                source_kind=source_kind,
                atomic_splits=atomic_splits,
                seed=seed,
                fallback_index=line_number - 1,
            )
            duration = float(row["duration_s"])
            if not min_duration_s <= duration <= max_duration_s:
                raise ManifestError(
                    f"duration_s {duration} outside [{min_duration_s}, {max_duration_s}]"
                )
            rows.append(row)
        except (ManifestError, TypeError, ValueError) as exc:
            rejects.append(
                {
                    "source_jsonl": str(path),
                    "source_line": line_number,
                    "source_locator": source_row.get("sample_id")
                    or source_row.get("name")
                    or source_row.get("file_name")
                    or source_row.get("index"),
                    "reason": str(exc),
                }
            )
    return rows


def select_stratified(
    rows: Sequence[dict[str, Any]],
    quotas: Mapping[tuple[str, str], int],
    *,
    seed: int,
    namespace: str,
    excluded_source_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Select exact per-scenario/language quotas in deterministic order."""
    excluded = excluded_source_ids or set()
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("source_utterance_id", "")) in excluded:
            continue
        buckets[(str(row.get("scenario", "")), str(row.get("language", "")))].append(row)

    selected: list[dict[str, Any]] = []
    shortages: list[str] = []
    for key in sorted(quotas):
        required = int(quotas[key])
        ordered = stable_order(buckets.get(key, []), seed, f"{namespace}:{key[0]}:{key[1]}")
        if len(ordered) < required:
            shortages.append(f"{key[0]}/{key[1]}={len(ordered)}/{required}")
            continue
        selected.extend(ordered[:required])
    if shortages:
        raise ManifestError("candidate quota shortage: " + ", ".join(shortages[:20]))
    return stable_order(selected, seed, namespace)


def select_clean(
    rows: Sequence[dict[str, Any]],
    *,
    language: str,
    count: int,
    seed: int,
    namespace: str,
    excluded_source_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Select one clean-language quota."""
    excluded = excluded_source_ids or set()
    candidates = [
        row
        for row in rows
        if row.get("language") == language
        and str(row.get("source_utterance_id", "")) not in excluded
    ]
    ordered = stable_order(candidates, seed, namespace)
    if len(ordered) < count:
        raise ManifestError(
            f"clean {language} candidate quota shortage: {len(ordered)}/{count}"
        )
    return ordered[:count]


def select_bench_smoke(
    rows: Sequence[dict[str, Any]], seed: int, per_stratum: int
) -> list[dict[str, Any]]:
    """Select a tiny en/zh by real/synthetic benchmark probe."""
    quotas = {
        (language, origin): per_stratum
        for language in LANGUAGES
        for origin in ("real", "synthetic")
    }
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(str(row.get("language")), str(row.get("audio_origin")))].append(row)
    selected: list[dict[str, Any]] = []
    for key, required in sorted(quotas.items()):
        ordered = stable_order(buckets.get(key, []), seed, f"bench-smoke:{key}")
        if len(ordered) < required:
            raise ManifestError(
                f"bench smoke quota shortage for {key[0]}/{key[1]}: "
                f"{len(ordered)}/{required}"
            )
        selected.extend(ordered[:required])
    return stable_order(selected, seed, "bench-smoke")


def selected_source_ids(rows: Iterable[Mapping[str, Any]]) -> set[str]:
    """Collect non-empty source utterance IDs."""
    return {
        str(row["source_utterance_id"])
        for row in rows
        if row.get("source_utterance_id") not in (None, "")
    }


def _selection_settings(config: Mapping[str, Any]) -> tuple[int, float, float]:
    project = config_section(config, "project")
    selection = config_section(config, "selection")
    return (
        int(project["seed"]),
        float(selection["min_duration_s"]),
        float(selection["max_duration_s"]),
    )


def load_all_candidate_inputs(
    config: Mapping[str, Any], args: argparse.Namespace
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Load the four local candidate files used by build and smoke."""
    sources = config_section(config, "sources")
    robust_config = require_mapping(sources.get("robust"), "sources.robust")
    atomic_splits = [str(item) for item in robust_config.get("atomic_splits", [])]
    seed, minimum, maximum = _selection_settings(config)
    rejects: list[dict[str, Any]] = []
    robust = load_candidates(
        Path(args.robust_candidates).expanduser(),
        source_config=robust_config,
        source_kind="robust",
        atomic_splits=atomic_splits,
        seed=seed,
        min_duration_s=minimum,
        max_duration_s=maximum,
        rejects=rejects,
    )
    english = load_candidates(
        Path(args.english_clean_candidates).expanduser(),
        source_config=require_mapping(sources.get("english_clean"), "sources.english_clean"),
        source_kind="clean",
        atomic_splits=atomic_splits,
        seed=seed,
        min_duration_s=minimum,
        max_duration_s=maximum,
        rejects=rejects,
    )
    chinese = load_candidates(
        Path(args.chinese_clean_candidates).expanduser(),
        source_config=require_mapping(sources.get("chinese_clean"), "sources.chinese_clean"),
        source_kind="clean",
        atomic_splits=atomic_splits,
        seed=seed,
        min_duration_s=minimum,
        max_duration_s=maximum,
        rejects=rejects,
    )
    bench = load_candidates(
        Path(args.bench_candidates).expanduser(),
        source_config=require_mapping(sources.get("robust_test"), "sources.robust_test"),
        source_kind="bench",
        atomic_splits=atomic_splits,
        seed=seed,
        min_duration_s=minimum,
        max_duration_s=maximum,
        rejects=rejects,
    )
    return robust, english, chinese, bench, rejects


def build_full_selection(
    config: Mapping[str, Any],
    robust_rows: Sequence[dict[str, Any]],
    english_rows: Sequence[dict[str, Any]],
    chinese_rows: Sequence[dict[str, Any]],
    bench_rows: Sequence[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Build the exact 200k/10k/5k deterministic selection."""
    seed = int(config_section(config, "project")["seed"])
    sources = config_section(config, "sources")
    robust_config = require_mapping(sources.get("robust"), "sources.robust")
    selection = config_section(config, "selection")
    train_plan = require_mapping(selection.get("robust_train"), "selection.robust_train")
    val_plan = require_mapping(
        selection.get("robust_validation"), "selection.robust_validation"
    )
    atomic_splits = [str(item) for item in robust_config["atomic_splits"]]
    split_names = sorted({str(row["scenario"]) for row in robust_rows})
    expected_compounds = int(robust_config["expected_compound_splits"])
    if len(split_names) != int(robust_config["expected_splits"]):
        raise ManifestError(
            f"robust candidate split count {len(split_names)} != "
            f"expected {robust_config['expected_splits']}"
        )

    train_scenarios = plan_scenario_quotas(
        split_names,
        atomic_splits,
        int(train_plan["atomic_per_split"]),
        int(train_plan["compound_total"]),
        expected_compounds,
    )
    val_scenarios = plan_scenario_quotas(
        split_names,
        atomic_splits,
        int(val_plan["atomic_per_split"]),
        int(val_plan["compound_total"]),
        expected_compounds,
    )
    robust_train = select_stratified(
        robust_rows,
        plan_language_quotas(train_scenarios),
        seed=seed,
        namespace="robust-train",
    )
    train_sources = selected_source_ids(robust_train)
    robust_validation = select_stratified(
        robust_rows,
        plan_language_quotas(val_scenarios),
        seed=seed,
        namespace="robust-validation",
        excluded_source_ids=train_sources,
    )

    clean_train_count = int(selection["clean_train_per_language"])
    clean_val_count = int(selection["clean_validation_per_language"])
    en_train = select_clean(
        english_rows,
        language="en",
        count=clean_train_count,
        seed=seed,
        namespace="english-clean-train",
    )
    zh_train = select_clean(
        chinese_rows,
        language="zh",
        count=clean_train_count,
        seed=seed,
        namespace="chinese-clean-train",
    )
    en_validation = select_clean(
        english_rows,
        language="en",
        count=clean_val_count,
        seed=seed,
        namespace="english-clean-validation",
        excluded_source_ids=selected_source_ids(en_train),
    )
    zh_validation = select_clean(
        chinese_rows,
        language="zh",
        count=clean_val_count,
        seed=seed,
        namespace="chinese-clean-validation",
        excluded_source_ids=selected_source_ids(zh_train),
    )

    expected_bench = int(selection["expected_bench_rows"])
    if len(bench_rows) != expected_bench:
        raise ManifestError(
            f"Bench must be used in full: got {len(bench_rows)} rows, expected {expected_bench}"
        )
    train = stable_order(
        [*robust_train, *en_train, *zh_train], seed, "full-train"
    )
    validation = stable_order(
        [*robust_validation, *en_validation, *zh_validation],
        seed,
        "full-validation",
    )
    test = stable_order(bench_rows, seed, "full-bench")
    canary = build_validation_canary(config, validation)
    return {
        "train": train,
        "validation": validation,
        "canary": canary,
        "test": test,
    }


def build_validation_canary(
    config: Mapping[str, Any], validation_rows: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Derive the fixed 512-row phase canary from the 10k validation set."""
    seed = int(config_section(config, "project")["seed"])
    settings = config_section(config, "canary")
    robust_target = int(settings["robust_rows"])
    clean_target = int(settings["clean_rows"])
    if robust_target + clean_target != int(settings["expected_rows"]):
        raise ConfigError("canary robust_rows plus clean_rows must equal expected_rows")

    robust = [
        row for row in validation_rows if row.get("condition_group") in {"atomic", "compound"}
    ]
    clean = [row for row in validation_rows if row.get("condition_group") == "clean"]
    scenarios = sorted({str(row["scenario"]) for row in robust})
    robust_scenario_quotas = distribute_total(scenarios, robust_target)
    robust_selected = select_stratified(
        robust,
        plan_language_quotas(robust_scenario_quotas),
        seed=seed,
        namespace="validation-canary-robust",
    )
    clean_language_quotas = distribute_total(LANGUAGES, clean_target)
    clean_selected: list[dict[str, Any]] = []
    for language in LANGUAGES:
        clean_selected.extend(
            select_clean(
                clean,
                language=language,
                count=clean_language_quotas[language],
                seed=seed,
                namespace=f"validation-canary-clean-{language}",
            )
        )
    canary = stable_order(
        [*robust_selected, *clean_selected], seed, "validation-canary"
    )
    if len(canary) != int(settings["expected_rows"]):
        raise ManifestError(
            f"canary rows {len(canary)} != expected {settings['expected_rows']}"
        )
    return canary


def build_smoke_selection(
    config: Mapping[str, Any],
    robust_rows: Sequence[dict[str, Any]],
    english_rows: Sequence[dict[str, Any]],
    chinese_rows: Sequence[dict[str, Any]],
    bench_rows: Sequence[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Build a 128-row train fixture plus a four-row Bench probe."""
    seed = int(config_section(config, "project")["seed"])
    sources = config_section(config, "sources")
    robust_config = require_mapping(sources.get("robust"), "sources.robust")
    smoke = config_section(config, "smoke")
    split_names = sorted({str(row["scenario"]) for row in robust_rows})
    if len(split_names) != int(robust_config["expected_splits"]):
        raise ManifestError(
            f"smoke needs all {robust_config['expected_splits']} robust splits; "
            f"got {len(split_names)}"
        )
    per_language = int(smoke["robust_per_language_per_split"])
    quotas = {
        (scenario, language): per_language
        for scenario in split_names
        for language in LANGUAGES
    }
    robust = select_stratified(
        robust_rows, quotas, seed=seed, namespace="smoke-robust"
    )
    clean_count = int(smoke["clean_per_language"])
    english = select_clean(
        english_rows,
        language="en",
        count=clean_count,
        seed=seed,
        namespace="smoke-clean-en",
    )
    chinese = select_clean(
        chinese_rows,
        language="zh",
        count=clean_count,
        seed=seed,
        namespace="smoke-clean-zh",
    )
    train = stable_order([*robust, *english, *chinese], seed, "smoke-train")
    if len(train) != int(smoke["expected_train_rows"]):
        raise ManifestError(
            f"smoke train rows {len(train)} != expected {smoke['expected_train_rows']}"
        )
    test = select_bench_smoke(
        bench_rows, seed, int(smoke["bench_per_language_origin"])
    )
    if len(test) != int(smoke["expected_bench_rows"]):
        raise ManifestError(
            f"smoke Bench rows {len(test)} != expected {smoke['expected_bench_rows']}"
        )
    return {"train": train, "test": test}


def normalized_transcript(text: Any) -> str:
    """Normalize transcript text only for overlap reporting."""
    normalized = unicodedata.normalize("NFKC", str(text or "")).lower()
    return "".join(char for char in normalized if char.isalnum())


def _audio_gate(path: Path, mode: str) -> str | None:
    if mode == "ignore":
        return None
    if not path.exists() or not path.is_file():
        return f"audio does not exist: {path}"
    if mode == "decode":
        try:
            import soundfile  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ManifestError(
                "--audio-mode decode requires soundfile; install it with "
                "`pip install soundfile`"
            ) from exc
        try:
            info = soundfile.info(str(path))
        except Exception as exc:  # library-specific exception types vary
            return f"audio decode failed for {path}: {exc}"
        if info.frames <= 0 or info.samplerate <= 0:
            return f"audio metadata invalid: {path}"
    return None


def validate_rows(
    manifests: Mapping[str, Sequence[dict[str, Any]]],
    config: Mapping[str, Any],
    *,
    data_root: Path,
    audio_mode: str = "ignore",
    allow_absolute_audio: bool = False,
    check_counts: bool = True,
) -> dict[str, Any]:
    """Validate schema, paths, counts and hard cross-split leakage gates."""
    required = [str(item) for item in config_section(config, "manifest")["required_fields"]]
    selection = config_section(config, "selection")
    minimum = float(selection["min_duration_s"])
    maximum = float(selection["max_duration_s"])
    errors: list[str] = []
    counts: dict[str, int] = {}
    field_values: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    sample_owner: dict[str, str] = {}
    role_sample_ids: dict[str, set[str]] = defaultdict(set)
    transcript_sets: dict[str, set[str]] = defaultdict(set)

    for role, rows in manifests.items():
        counts[role] = len(rows)
        for index, row in enumerate(rows):
            location = f"{role}[{index}]"
            missing = [field for field in required if row.get(field) in (None, "")]
            if missing:
                errors.append(f"{location}: missing fields: {', '.join(missing)}")
                continue
            if row.get("language") not in LANGUAGES:
                errors.append(f"{location}: invalid language {row.get('language')!r}")
            if row.get("condition_group") not in CONDITION_GROUPS:
                errors.append(
                    f"{location}: invalid condition_group {row.get('condition_group')!r}"
                )
            if row.get("audio_origin") not in AUDIO_ORIGINS:
                errors.append(f"{location}: invalid audio_origin {row.get('audio_origin')!r}")
            try:
                duration = finite_float(row.get("duration_s"), "duration_s")
                if not minimum <= duration <= maximum:
                    errors.append(f"{location}: duration_s {duration} outside allowed range")
            except ManifestError as exc:
                errors.append(f"{location}: {exc}")
            revision = str(row.get("source_revision", ""))
            if len(revision) != PINNED_REVISION_LENGTH or any(
                char not in "0123456789abcdef" for char in revision.lower()
            ):
                errors.append(f"{location}: source_revision is not a pinned commit")

            audio_value = str(row.get("audio", ""))
            audio = Path(audio_value).expanduser()
            if audio.is_absolute() and not allow_absolute_audio:
                errors.append(f"{location}: audio path must be relative to data_root")
            resolved_audio = audio if audio.is_absolute() else data_root / audio
            audio_error = _audio_gate(resolved_audio, audio_mode)
            if audio_error:
                errors.append(f"{location}: {audio_error}")

            sample_id = str(row.get("sample_id", ""))
            if sample_id in role_sample_ids[role]:
                errors.append(f"duplicate sample_id within {role}: {sample_id!r}")
            role_sample_ids[role].add(sample_id)
            if role not in DERIVED_ROLES:
                previous = sample_owner.get(sample_id)
                if previous is not None:
                    errors.append(f"sample_id overlap: {sample_id!r} in {previous} and {role}")
                else:
                    sample_owner[sample_id] = role
            for field in (
                "source_utterance_id",
                "source_name",
                "benchmark_id",
                "audio",
                "audio_sha256",
            ):
                value = row.get(field)
                if value not in (None, ""):
                    field_values[field][role].add(str(value))
            transcript = normalized_transcript(row.get("answer"))
            if transcript:
                transcript_sets[role].add(transcript)

    leakage_roles = [role for role in ("train", "validation", "test") if role in manifests]
    leakage: dict[str, dict[str, int]] = {}
    for field, by_role in sorted(field_values.items()):
        for left_index, left in enumerate(leakage_roles):
            for right in leakage_roles[left_index + 1 :]:
                overlap = by_role[left] & by_role[right]
                key = f"{left}:{right}:{field}"
                leakage[key] = {
                    "count": len(overlap),
                    "examples": sorted(overlap)[:10],
                }
                if overlap:
                    errors.append(f"hard overlap {key}: {len(overlap)}")

    transcript_overlap: dict[str, int] = {}
    for left_index, left in enumerate(leakage_roles):
        for right in leakage_roles[left_index + 1 :]:
            transcript_overlap[f"{left}:{right}"] = len(
                transcript_sets[left] & transcript_sets[right]
            )

    if check_counts:
        expected = {
            "train": int(selection["expected_train_rows"]),
            "validation": int(selection["expected_validation_rows"]),
            "test": int(selection["expected_bench_rows"]),
        }
        curriculum = config_section(config, "curriculum")
        expected["curriculum"] = int(curriculum["target_rows"])
        expected["canary"] = int(config_section(config, "canary")["expected_rows"])
        for role, actual in counts.items():
            if role in expected and actual != expected[role]:
                errors.append(f"{role} rows {actual} != expected {expected[role]}")

    if "curriculum" in manifests:
        train_ids = {str(row["sample_id"]) for row in manifests.get("train", [])}
        maximum_error = float(config_section(config, "curriculum")["maximum_error_rate"])
        for index, row in enumerate(manifests["curriculum"]):
            sample_id = str(row.get("sample_id", ""))
            if train_ids and sample_id not in train_ids:
                errors.append(f"curriculum[{index}]: sample is not in train: {sample_id}")
            try:
                error_rate = finite_float(row.get("base_error_rate"), "base_error_rate")
                if error_rate < 0 or error_rate >= maximum_error:
                    errors.append(
                        f"curriculum[{index}]: base_error_rate {error_rate} outside "
                        f"[0, {maximum_error})"
                    )
            except ManifestError as exc:
                errors.append(f"curriculum[{index}]: {exc}")
            expected_metric = "wer" if row.get("language") == "en" else "cer"
            if row.get("base_metric") != expected_metric:
                errors.append(
                    f"curriculum[{index}]: base_metric must be {expected_metric}"
                )

    if "canary" in manifests:
        validation_ids = {
            str(row["sample_id"]) for row in manifests.get("validation", [])
        }
        if validation_ids:
            for index, row in enumerate(manifests["canary"]):
                sample_id = str(row.get("sample_id", ""))
                if sample_id not in validation_ids:
                    errors.append(f"canary[{index}]: sample is not in validation: {sample_id}")

    return {
        "hard_checks_pass": not errors,
        "errors": errors[:200],
        "error_count": len(errors),
        "row_counts": counts,
        "hard_overlap": leakage,
        "normalized_transcript_overlap_report_only": transcript_overlap,
        "audio_mode": audio_mode,
        "data_root": str(data_root),
    }


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return compact scenario/language/group statistics."""
    return {
        "rows": len(rows),
        "language": dict(sorted(Counter(str(row.get("language")) for row in rows).items())),
        "condition_group": dict(
            sorted(Counter(str(row.get("condition_group")) for row in rows).items())
        ),
        "scenario": dict(sorted(Counter(str(row.get("scenario")) for row in rows).items())),
        "audio_origin": dict(
            sorted(Counter(str(row.get("audio_origin")) for row in rows).items())
        ),
    }


def ensure_overwrite_allowed(paths: Iterable[Path], force: bool) -> None:
    """Avoid silently replacing a reproducible manifest run."""
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        preview = "\n".join(str(path) for path in existing)
        raise ManifestError(f"refusing to overwrite outputs; pass --force:\n{preview}")


def build_curriculum_rows(
    train_rows: Sequence[dict[str, Any]],
    scored_rows: Sequence[dict[str, Any]],
    *,
    target_rows: int,
    maximum_error_rate: float,
    seed: int,
) -> list[dict[str, Any]]:
    """Select a deterministic language/scenario round-robin A2S curriculum."""
    train_by_id: dict[str, dict[str, Any]] = {}
    for row in train_rows:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in train_by_id:
            raise ManifestError(f"train sample_id is empty or duplicated: {sample_id!r}")
        train_by_id[sample_id] = row

    scored_by_id: dict[str, dict[str, Any]] = {}
    for score in scored_rows:
        sample_id = str(score.get("sample_id", ""))
        if not sample_id or sample_id in scored_by_id:
            raise ManifestError(f"scored sample_id is empty or duplicated: {sample_id!r}")
        if sample_id not in train_by_id:
            raise ManifestError(f"scored sample is not in train: {sample_id}")
        train_row = train_by_id[sample_id]
        if train_row.get("condition_group") not in {"atomic", "compound"}:
            continue
        raw_error = score.get("base_error_rate")
        if raw_error is None:
            raw_error = score.get("error_rate")
        if raw_error is None:
            raw_error = score.get("wer") if train_row.get("language") == "en" else score.get("cer")
        error_rate = finite_float(raw_error, "base_error_rate")
        if error_rate < 0:
            raise ManifestError(f"base_error_rate must be non-negative: {sample_id}")
        if error_rate >= maximum_error_rate:
            continue
        expected_metric = "wer" if train_row.get("language") == "en" else "cer"
        metric = str(score.get("base_metric") or score.get("metric") or expected_metric).lower()
        if metric != expected_metric:
            raise ManifestError(
                f"{sample_id}: base_metric {metric!r} does not match {expected_metric!r}"
            )
        enriched = dict(train_row)
        enriched["base_prediction"] = str(
            score.get("base_prediction") or score.get("prediction") or ""
        )
        enriched["base_error_rate"] = error_rate
        enriched["base_metric"] = metric
        scored_by_id[sample_id] = enriched

    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scored_by_id.values():
        buckets[(str(row["language"]), str(row["scenario"]))].append(row)
    for key in buckets:
        buckets[key] = stable_order(
            buckets[key], seed, f"curriculum:{key[0]}:{key[1]}"
        )

    selected: list[dict[str, Any]] = []
    offsets = {key: 0 for key in buckets}
    keys = sorted(buckets)
    while len(selected) < target_rows:
        made_progress = False
        for key in keys:
            offset = offsets[key]
            if offset >= len(buckets[key]):
                continue
            selected.append(buckets[key][offset])
            offsets[key] += 1
            made_progress = True
            if len(selected) == target_rows:
                break
        if not made_progress:
            raise ManifestError(
                f"only {len(selected)} scored train rows have base_error_rate "
                f"< {maximum_error_rate}; need {target_rows}"
            )
    return selected


def threshold_slug(threshold: float) -> str:
    """Format 0.30 as the stable filename fragment 0_30."""
    return f"{threshold:.2f}".replace(".", "_")


def curriculum_views(
    rows: Sequence[dict[str, Any]], thresholds: Sequence[float]
) -> dict[float, list[dict[str, Any]]]:
    """Build cumulative threshold views while preserving selected order."""
    ordered_thresholds = sorted(float(value) for value in thresholds)
    views = {
        threshold: [
            row for row in rows if float(row["base_error_rate"]) < threshold
        ]
        for threshold in ordered_thresholds
    }
    previous: set[str] = set()
    for threshold in ordered_thresholds:
        current = {str(row["sample_id"]) for row in views[threshold]}
        if not previous <= current:
            raise ManifestError("curriculum threshold views are not cumulative")
        previous = current
    return views


def _feature_names(features: Any) -> list[str]:
    if features is None:
        return []
    if hasattr(features, "keys"):
        return sorted(str(key) for key in features.keys())
    if isinstance(features, Mapping):
        return sorted(str(key) for key in features)
    return []


def probe_dataset_source(
    source_name: str,
    source_config: Mapping[str, Any],
    *,
    loader: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Inspect one pinned Hub dataset builder without loading any examples."""
    dataset_id = str(source_config.get("dataset_id") or "")
    revision = assert_pinned_revision(source_config.get("revision"), source_name)
    if not dataset_id:
        raise ConfigError(f"sources.{source_name}.dataset_id is required")
    if loader is None:
        try:
            from datasets import load_dataset_builder  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ConfigError(
                "metadata probe requires `datasets`; install it with "
                "`pip install datasets`"
            ) from exc
        loader = load_dataset_builder
    kwargs: dict[str, Any] = {"revision": revision}
    if source_config.get("config"):
        kwargs["name"] = source_config["config"]
    builder = loader(dataset_id, **kwargs)
    info = builder.info
    split_info = info.splits or {}
    split_rows = {
        str(name): int(getattr(value, "num_examples", 0) or 0)
        for name, value in split_info.items()
    }
    features = _feature_names(info.features)
    errors: list[str] = []
    if source_config.get("expected_splits") is not None:
        expected = int(source_config["expected_splits"])
        if len(split_rows) != expected:
            errors.append(f"split count {len(split_rows)} != expected {expected}")
    required_fields = [
        str(item) for item in source_config.get("required_source_fields", [])
    ]
    missing_fields = sorted(set(required_fields) - set(features))
    if missing_fields:
        errors.append("missing source fields: " + ", ".join(missing_fields))
    if source_config.get("expected_rows") is not None:
        expected_rows = int(source_config["expected_rows"])
        actual_rows = sum(split_rows.values())
        if actual_rows != expected_rows:
            errors.append(f"rows {actual_rows} != expected {expected_rows}")
    download_size = int(getattr(info, "download_size", 0) or 0)
    dataset_size = int(getattr(info, "dataset_size", 0) or 0)
    if source_config.get("max_download_bytes") is not None and download_size:
        maximum_download = int(source_config["max_download_bytes"])
        if download_size > maximum_download:
            errors.append(
                f"download size {download_size} exceeds budget {maximum_download}"
            )
    if source_config.get("max_dataset_bytes") is not None and dataset_size:
        maximum_dataset = int(source_config["max_dataset_bytes"])
        if dataset_size > maximum_dataset:
            errors.append(f"dataset size {dataset_size} exceeds budget {maximum_dataset}")
    return {
        "source": source_name,
        "dataset_id": dataset_id,
        "requested_revision": revision,
        "audio_downloaded": False,
        "features": features,
        "split_count": len(split_rows),
        "split_rows": dict(sorted(split_rows.items())),
        "download_size_bytes": download_size,
        "dataset_size_bytes": dataset_size,
        "quota_language_check": "deferred_to_materialized_candidate_build",
        "errors": errors,
        "passed": not errors,
    }


def add_robust_quota_capacity_check(
    report: dict[str, Any], config: Mapping[str, Any]
) -> None:
    """Check total per-split capacity before any audio is materialized."""
    sources = config_section(config, "sources")
    source = require_mapping(sources.get("robust"), "sources.robust")
    selection = config_section(config, "selection")
    train_plan = require_mapping(selection.get("robust_train"), "selection.robust_train")
    val_plan = require_mapping(
        selection.get("robust_validation"), "selection.robust_validation"
    )
    split_rows = {str(key): int(value) for key, value in report["split_rows"].items()}
    split_names = sorted(split_rows)
    train = plan_scenario_quotas(
        split_names,
        [str(item) for item in source["atomic_splits"]],
        int(train_plan["atomic_per_split"]),
        int(train_plan["compound_total"]),
        int(source["expected_compound_splits"]),
    )
    validation = plan_scenario_quotas(
        split_names,
        [str(item) for item in source["atomic_splits"]],
        int(val_plan["atomic_per_split"]),
        int(val_plan["compound_total"]),
        int(source["expected_compound_splits"]),
    )
    required = {name: train[name] + validation[name] for name in split_names}
    shortages = {
        name: {"available": split_rows[name], "required": required[name]}
        for name in split_names
        if split_rows[name] < required[name]
    }
    report["quota_required_rows_by_split"] = required
    report["quota_total_capacity_passed"] = not shortages
    report["quota_shortages"] = shortages
    report["quota_language_check"] = "deferred_to_materialized_candidate_build"
    if shortages:
        report["errors"].append(
            f"{len(shortages)} splits do not have enough rows for train plus validation"
        )
        report["passed"] = False


def output_directory(config: Mapping[str, Any], override: str | None) -> Path:
    value = override or str(config_section(config, "project")["output_dir"])
    return Path(value).expanduser()


def command_probe(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(Path(args.config).expanduser())
    sources = config_section(config, "sources")
    names = sorted(sources) if args.source == "all" else [args.source]
    reports = []
    for name in names:
        report = probe_dataset_source(
            name, require_mapping(sources.get(name), f"sources.{name}")
        )
        if name == "robust" and report["passed"]:
            add_robust_quota_capacity_check(report, config)
        reports.append(report)
    result = {
        "passed": all(report["passed"] for report in reports),
        "audio_downloaded": False,
        "sources": reports,
    }
    if args.output:
        write_json(Path(args.output).expanduser(), result)
    if not result["passed"]:
        raise ManifestError("metadata probe failed; inspect the probe report")
    return result


def _candidate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--robust-candidates", required=True)
    parser.add_argument("--english-clean-candidates", required=True)
    parser.add_argument("--chinese-clean-candidates", required=True)
    parser.add_argument("--bench-candidates", required=True)


def command_smoke(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(Path(args.config).expanduser())
    robust, english, chinese, bench, rejects = load_all_candidate_inputs(config, args)
    selected = build_smoke_selection(config, robust, english, chinese, bench)
    output_dir = output_directory(config, args.output_dir)
    train_path = output_dir / "public_robust_smoke_128.jsonl"
    bench_path = output_dir / "vitw_bench_smoke_4.jsonl"
    report_path = output_dir / "public_robust_smoke_validation.json"
    rejects_path = output_dir / "public_robust_smoke_rejects.jsonl"
    ensure_overwrite_allowed(
        [train_path, bench_path, report_path, rejects_path], args.force
    )
    write_jsonl(train_path, selected["train"])
    write_jsonl(bench_path, selected["test"])
    write_jsonl(rejects_path, rejects)
    data_root = Path(args.data_root or config_section(config, "project")["data_root"])
    report = validate_rows(
        selected,
        config,
        data_root=data_root,
        audio_mode=args.audio_mode,
        allow_absolute_audio=args.allow_absolute_audio,
        check_counts=False,
    )
    report["expected_train_rows"] = int(config_section(config, "smoke")["expected_train_rows"])
    report["expected_test_rows"] = int(config_section(config, "smoke")["expected_bench_rows"])
    write_json(report_path, report)
    if not report["hard_checks_pass"]:
        raise ManifestError("smoke validation failed; inspect the validation report")
    return {
        "train": str(train_path),
        "test": str(bench_path),
        "rejects": len(rejects),
        "validation": str(report_path),
    }


def command_build(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(Path(args.config).expanduser())
    robust, english, chinese, bench, rejects = load_all_candidate_inputs(config, args)
    selected = build_full_selection(config, robust, english, chinese, bench)
    output_dir = output_directory(config, args.output_dir)
    manifest_config = config_section(config, "manifest")
    outputs = require_mapping(manifest_config.get("outputs"), "manifest.outputs")
    paths = {name: output_dir / str(filename) for name, filename in outputs.items()}
    required_paths = [
        paths["train"],
        paths["validation"],
        paths["validation_canary"],
        paths["robust_test"],
        paths["stats"],
        paths["validation_report"],
        paths["rejects"],
        paths["sources"],
    ]
    ensure_overwrite_allowed(required_paths, args.force)
    write_jsonl(paths["train"], selected["train"])
    write_jsonl(paths["validation"], selected["validation"])
    write_jsonl(paths["validation_canary"], selected["canary"])
    write_jsonl(paths["robust_test"], selected["test"])
    write_jsonl(paths["rejects"], rejects)

    data_root = Path(args.data_root or config_section(config, "project")["data_root"])
    report = validate_rows(
        selected,
        config,
        data_root=data_root,
        audio_mode=args.audio_mode,
        allow_absolute_audio=args.allow_absolute_audio,
        check_counts=True,
    )
    write_json(paths["validation_report"], report)
    stats = {
        "seed": int(config_section(config, "project")["seed"]),
        "train": summarize_rows(selected["train"]),
        "validation": summarize_rows(selected["validation"]),
        "canary": summarize_rows(selected["canary"]),
        "test": summarize_rows(selected["test"]),
        "rejects": len(rejects),
        "manifest_sha256": {
            "train": file_sha256(paths["train"]),
            "validation": file_sha256(paths["validation"]),
            "canary": file_sha256(paths["validation_canary"]),
            "test": file_sha256(paths["robust_test"]),
        },
    }
    write_json(paths["stats"], stats)
    write_json(
        paths["sources"],
        {
            "seed": int(config_section(config, "project")["seed"]),
            "sources": config_section(config, "sources"),
            "config_sha256": file_sha256(Path(args.config).expanduser()),
        },
    )
    if not report["hard_checks_pass"]:
        raise ManifestError("full manifest validation failed; inspect validation report")
    return {
        "train": str(paths["train"]),
        "validation": str(paths["validation"]),
        "canary": str(paths["validation_canary"]),
        "test": str(paths["robust_test"]),
        "stats": str(paths["stats"]),
        "rejects": len(rejects),
    }


def command_validate(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(Path(args.config).expanduser())
    manifests: dict[str, list[dict[str, Any]]] = {}
    for role in ("train", "validation", "canary", "test", "curriculum"):
        value = getattr(args, role)
        if value:
            manifests[role] = read_jsonl(Path(value).expanduser())
    if not manifests:
        raise ManifestError("validate requires at least one manifest path")
    data_root = Path(args.data_root or config_section(config, "project")["data_root"])
    report = validate_rows(
        manifests,
        config,
        data_root=data_root,
        audio_mode=args.audio_mode,
        allow_absolute_audio=args.allow_absolute_audio,
        check_counts=True,
    )
    if args.output:
        write_json(Path(args.output).expanduser(), report)
    if not report["hard_checks_pass"]:
        raise ManifestError("manifest validation failed")
    return report


def command_curriculum(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(Path(args.config).expanduser())
    settings = config_section(config, "curriculum")
    seed = int(config_section(config, "project")["seed"])
    train_rows = read_jsonl(Path(args.train).expanduser())
    scored_rows = read_jsonl(Path(args.scored).expanduser())
    target = int(settings["target_rows"])
    maximum = float(settings["maximum_error_rate"])
    rows = build_curriculum_rows(
        train_rows,
        scored_rows,
        target_rows=target,
        maximum_error_rate=maximum,
        seed=seed,
    )
    thresholds = [float(value) for value in settings["cumulative_thresholds"]]
    views = curriculum_views(rows, thresholds)
    output = Path(args.output).expanduser()
    paths = [output, *[
        output.with_name(f"{output.stem}.lt_{threshold_slug(value)}{output.suffix}")
        for value in thresholds
    ]]
    if args.report:
        paths.append(Path(args.report).expanduser())
    ensure_overwrite_allowed(paths, args.force)
    write_jsonl(output, rows)
    view_paths: dict[str, str] = {}
    previous_ids: set[str] = set()
    for threshold in sorted(views):
        view_path = output.with_name(
            f"{output.stem}.lt_{threshold_slug(threshold)}{output.suffix}"
        )
        write_jsonl(view_path, views[threshold])
        current_ids = {str(row["sample_id"]) for row in views[threshold]}
        if not previous_ids <= current_ids:
            raise ManifestError("written curriculum views are not cumulative")
        previous_ids = current_ids
        view_paths[f"lt_{threshold:.2f}"] = str(view_path)
    result = {
        "output": str(output),
        "rows": len(rows),
        "maximum_error_rate": maximum,
        "manifest_sha256": file_sha256(output),
        "views": {
            key: {"path": path, "rows": len(views[float(key[3:])])}
            for key, path in view_paths.items()
        },
        "strata": {
            f"{language}:{scenario}": count
            for (language, scenario), count in sorted(
                Counter((str(row["language"]), str(row["scenario"])) for row in rows).items()
            )
        },
    }
    if args.report:
        write_json(Path(args.report).expanduser(), result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe = subparsers.add_parser("probe", help="inspect pinned Hub metadata only")
    probe.add_argument("--config", required=True)
    probe.add_argument(
        "--source",
        default="all",
        choices=("all", "robust", "robust_test", "english_clean", "chinese_clean"),
    )
    probe.add_argument("--output", default="")
    probe.set_defaults(handler=command_probe)

    smoke = subparsers.add_parser("smoke", help="build the 128-row local fixture")
    smoke.add_argument("--config", required=True)
    _candidate_args(smoke)
    smoke.add_argument("--output-dir", default="")
    smoke.add_argument("--data-root", default="")
    smoke.add_argument("--audio-mode", choices=("ignore", "exists", "decode"), default="ignore")
    smoke.add_argument("--allow-absolute-audio", action="store_true")
    smoke.add_argument("--force", action="store_true")
    smoke.set_defaults(handler=command_smoke)

    build = subparsers.add_parser("build", help="build fixed 200k/10k/5k manifests")
    build.add_argument("--config", required=True)
    _candidate_args(build)
    build.add_argument("--output-dir", default="")
    build.add_argument("--data-root", default="")
    build.add_argument("--audio-mode", choices=("exists", "decode"), default="exists")
    build.add_argument("--allow-absolute-audio", action="store_true")
    build.add_argument("--force", action="store_true")
    build.set_defaults(handler=command_build)

    validate = subparsers.add_parser("validate", help="validate canonical manifests")
    validate.add_argument("--config", required=True)
    validate.add_argument("--train", default="")
    validate.add_argument("--validation", default="")
    validate.add_argument("--canary", default="")
    validate.add_argument("--test", default="")
    validate.add_argument("--curriculum", default="")
    validate.add_argument("--data-root", default="")
    validate.add_argument("--audio-mode", choices=("exists", "decode"), default="exists")
    validate.add_argument("--allow-absolute-audio", action="store_true")
    validate.add_argument("--output", default="")
    validate.set_defaults(handler=command_validate)

    curriculum = subparsers.add_parser(
        "curriculum", help="build deterministic 30k base-error views"
    )
    curriculum.add_argument("--config", required=True)
    curriculum.add_argument("--train", required=True)
    curriculum.add_argument("--scored", required=True)
    curriculum.add_argument("--output", required=True)
    curriculum.add_argument("--report", default="")
    curriculum.add_argument("--force", action="store_true")
    curriculum.set_defaults(handler=command_curriculum)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = args.handler(args)
    except (ConfigError, ManifestError, KeyError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
