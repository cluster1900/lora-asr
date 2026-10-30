"""zh-cer-v1 normalization, Levenshtein CER/WER, and output-health flags.

The rules are the contract in docs/qwen3-asr/05_testing_plan.md. Normalization
is for scoring only and must not replace the training target or prediction_raw.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import unicodedata
from pathlib import Path
from typing import Any, Sequence

NORMALIZATION_VERSION = "zh-cer-v1"
LOCKED_JIEBA_VERSION = "0.42.1"
LOCKED_JIEBA_DICT_SHA256 = "7197c3211ddd98962b036cdf40324d1ea2bfaa12bd028e68faa70111a88e12a8"
REPEAT_RUN_MIN = 4
REPEAT_PATTERN_MIN = 2
REPEAT_PATTERN_MAX = 8
REPEAT_PATTERN_TIMES = 3
LONG_LENGTH_RATIO = 1.5
_JIEBA_CACHE: dict[tuple[str, str, str], Any] = {}


def normalize_zh_cer_v1(text: str) -> str:
    """Apply zh-cer-v1 and return the character string used for CER."""
    value = unicodedata.normalize("NFKC", str(text or "")).casefold()
    kept: list[str] = []
    for char in value:
        category = unicodedata.category(char)
        if category.startswith("P") or category.startswith("Z") or category == "Cc":
            continue
        kept.append(char)
    return "".join(kept)


GAP_ANCHOR_LOW = 0.40
GAP_ANCHOR_HIGH = 0.55


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """Levenshtein distance with one row of memory. Each edit costs 1."""
    previous = list(range(len(hypothesis) + 1))
    for row_index, ref_token in enumerate(reference, start=1):
        current = [row_index]
        for column_index, hyp_token in enumerate(hypothesis, start=1):
            substitution = previous[column_index - 1] + int(ref_token != hyp_token)
            current.append(
                min(
                    previous[column_index] + 1,
                    current[column_index - 1] + 1,
                    substitution,
                )
            )
        previous = current
    return previous[-1]


def align_edits(reference: Sequence[str], hypothesis: Sequence[str]) -> list[dict[str, int | str]]:
    """Full-matrix backtrace. The edit count matches ``edit_distance``.

    From the end, the first predecessor that reproduces the cell cost wins:
    an equal-character diagonal, then a substitution, then a deletion, then
    an insertion. An insertion records the reference index still ahead of it.
    After the reference is consumed, that index equals the reference length.
    """
    ref = list(reference)
    hyp = list(hypothesis)
    rows = len(ref) + 1
    cols = len(hyp) + 1
    cost = [[0] * cols for _ in range(rows)]
    for column in range(1, cols):
        cost[0][column] = column
    for row in range(1, rows):
        cost[row][0] = row
        for column in range(1, cols):
            substitution = cost[row - 1][column - 1] + int(ref[row - 1] != hyp[column - 1])
            cost[row][column] = min(cost[row - 1][column] + 1, cost[row][column - 1] + 1, substitution)
    operations: list[dict[str, int | str]] = []
    row = len(ref)
    column = len(hyp)
    while row > 0 or column > 0:
        if row > 0 and column > 0 and ref[row - 1] == hyp[column - 1] and cost[row][column] == cost[row - 1][column - 1]:
            operations.append({"op": "match", "ref_index": row - 1, "hyp_index": column - 1})
            row -= 1
            column -= 1
            continue
        if (
            row > 0
            and column > 0
            and ref[row - 1] != hyp[column - 1]
            and cost[row][column] == cost[row - 1][column - 1] + 1
        ):
            operations.append({"op": "substitute", "ref_index": row - 1, "hyp_index": column - 1})
            row -= 1
            column -= 1
            continue
        if row > 0 and cost[row][column] == cost[row - 1][column] + 1:
            operations.append({"op": "delete", "ref_index": row - 1, "hyp_index": column})
            row -= 1
            continue
        if column > 0 and cost[row][column] == cost[row][column - 1] + 1:
            operations.append({"op": "insert", "ref_index": row, "hyp_index": column - 1})
            column -= 1
            continue
        raise RuntimeError("alignment backtrace lost the Levenshtein path")
    operations.reverse()
    return operations


def gap_insertion_count(reference: str, hypothesis: str) -> int:
    """Count insertions whose reference anchor sits in the dropout gap band."""
    if not reference:
        raise ValueError("gap alignment needs a non-empty reference")
    length = len(reference)
    count = 0
    for operation in align_edits(list(reference), list(hypothesis)):
        if operation["op"] != "insert":
            continue
        anchor = int(operation["ref_index"]) / length
        if GAP_ANCHOR_LOW <= anchor <= GAP_ANCHOR_HIGH:
            count += 1
    return count


def is_repeated(normalized: str) -> bool:
    """Flag a 4-char run or a 2–8 char block repeated three times in a row."""
    chars = list(normalized)
    if not chars:
        return False
    run = 1
    for index in range(1, len(chars)):
        run = run + 1 if chars[index] == chars[index - 1] else 1
        if run >= REPEAT_RUN_MIN:
            return True
    length = len(chars)
    for size in range(REPEAT_PATTERN_MIN, REPEAT_PATTERN_MAX + 1):
        span = size * REPEAT_PATTERN_TIMES
        if length < span:
            continue
        for start in range(0, length - span + 1):
            block = chars[start : start + size]
            if (
                chars[start + size : start + 2 * size] == block
                and chars[start + 2 * size : start + 3 * size] == block
            ):
                return True
    return False


def is_long_output(reference_chars: int, hypothesis_chars: int) -> bool:
    """True when the hypothesis is strictly longer than 1.5 times the reference."""
    if reference_chars <= 0:
        return False
    return hypothesis_chars > LONG_LENGTH_RATIO * reference_chars


def score_pair(
    reference_text: str,
    prediction_text: str,
    *,
    inference_error: bool,
    stop_reason: str | None,
) -> dict[str, object]:
    """Score one pair. Inference errors use an empty hypothesis for CER only."""
    reference_normalized = normalize_zh_cer_v1(reference_text)
    prediction_normalized = normalize_zh_cer_v1(prediction_text)
    if not reference_normalized:
        raise ValueError("reference text is empty after zh-cer-v1 normalization")
    scored = "" if inference_error else prediction_normalized
    edits = edit_distance(list(reference_normalized), list(scored))
    reference_chars = len(reference_normalized)
    return {
        "normalization_version": NORMALIZATION_VERSION,
        "reference_normalized": reference_normalized,
        "prediction_normalized": prediction_normalized,
        "scored_prediction_normalized": scored,
        "edit_count": edits,
        "reference_char_count": reference_chars,
        "sample_cer": edits / reference_chars,
        "empty_output": prediction_normalized == "",
        "repeated_output": is_repeated(prediction_normalized),
        "long_output": is_long_output(reference_chars, len(prediction_normalized)),
        "truncated": stop_reason == "length",
        "inference_error": inference_error,
        "valid_output": (not inference_error) and prediction_normalized != "",
    }


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jieba_tokenizer(
    *,
    expected_version: str,
    expected_dictionary_sha256: str,
    dict_path_override: Path | None = None,
) -> Any:
    """Return an isolated jieba.Tokenizer verified against version and dict SHA-256."""
    if not isinstance(expected_version, str) or not expected_version.strip():
        raise ValueError("expected_version must be a non-empty string")
    if (
        not isinstance(expected_dictionary_sha256, str)
        or len(expected_dictionary_sha256) != 64
        or not all(ch in "0123456789abcdefABCDEF" for ch in expected_dictionary_sha256)
    ):
        raise ValueError("expected_dictionary_sha256 must be a 64-char hex string")
    expected_sha = expected_dictionary_sha256.lower()

    try:
        import jieba  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ValueError("jieba is not installed in the active Python environment") from exc

    try:
        actual_version = importlib.metadata.version("jieba")
    except importlib.metadata.PackageNotFoundError:
        actual_version = str(getattr(jieba, "__version__", ""))
    if actual_version != expected_version:
        raise ValueError(f"jieba version mismatch: expected {expected_version}, got {actual_version}")

    dict_path = (
        dict_path_override.resolve()
        if dict_path_override is not None
        else Path(jieba.__file__).resolve().parent / "dict.txt"
    )
    if not dict_path.is_file():
        raise ValueError(f"jieba dictionary file not found: {dict_path}")
    actual_sha = _sha256_path(dict_path)
    if actual_sha != expected_sha:
        raise ValueError(
            f"jieba dictionary sha256 mismatch: expected {expected_sha}, got {actual_sha}"
        )

    cache_key = (actual_version, actual_sha, str(dict_path))
    cached = _JIEBA_CACHE.get(cache_key)
    if cached is not None:
        return cached

    tokenizer = jieba.Tokenizer(dictionary=str(dict_path))
    tokenizer.initialize()
    _JIEBA_CACHE[cache_key] = tokenizer
    return tokenizer


def score_wer_pair(
    reference_normalized: str,
    scored_prediction_normalized: str,
    tokenizer: Any,
    *,
    hmm: bool = False,
) -> dict[str, object]:
    """Score word-level Levenshtein WER on zh-cer-v1 normalized strings."""
    if not reference_normalized:
        raise ValueError("reference text is empty after zh-cer-v1 normalization")
    reference_words = [token for token in tokenizer.lcut(reference_normalized, HMM=hmm) if token]
    if not reference_words:
        raise ValueError("reference words are empty after tokenization")
    scored_words = (
        []
        if not scored_prediction_normalized
        else [token for token in tokenizer.lcut(scored_prediction_normalized, HMM=hmm) if token]
    )
    word_edits = edit_distance(reference_words, scored_words)
    reference_word_count = len(reference_words)
    return {
        "reference_words": reference_words,
        "scored_prediction_words": scored_words,
        "word_edit_count": word_edits,
        "reference_word_count": reference_word_count,
        "sample_wer": word_edits / reference_word_count,
    }
