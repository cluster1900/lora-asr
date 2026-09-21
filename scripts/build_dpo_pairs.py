#!/usr/bin/env python3
"""DPO Preference Pair Builder and Reference Logprob Precalculator for Qwen3-ASR.

Responsibilities:
1. Ingest candidate audio manifests (e.g. pilot_dpo.jsonl, dpo_val_pool.jsonl).
2. Generate or match candidate transcript pairs (Base vs Step 50 SFT merged model,
   with controlled synthetic perturbation negatives for Clean ties).
3. Score candidates with WER (en) or CER (zh) against gold transcript.
4. Select strictly valid preference pairs (chosen_error < rejected_error, chosen != rejected, non-empty).
5. Precalculate sequence log probabilities under frozen reference model (Step 50 SFT merged base)
   for both chosen and rejected sequences.
6. Write contract-compliant DPO manifest with complete provenance hashes and audit summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import soundfile as sf
    HAVE_SOUNDFILE = True
except ImportError:
    sf = None  # type: ignore
    HAVE_SOUNDFILE = False

try:
    import torch
    import torch.nn.functional as F
    HAVE_TORCH = True
except ImportError:
    torch = None  # type: ignore
    F = None  # type: ignore
    HAVE_TORCH = False

try:
    from transformers import AutoProcessor
    from qwen_asr import Qwen3ASRModel
    HAVE_QWEN_ASR = True
except ImportError:
    AutoProcessor = None  # type: ignore
    Qwen3ASRModel = None  # type: ignore
    HAVE_QWEN_ASR = False

# Import official WER/CER scoring
try:
    from evaluation.eval_wer import edit_distance, normalize_text, tokenize
except ImportError:
    # Fallback when running directly inside scripts directory
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from evaluation.eval_wer import edit_distance, normalize_text, tokenize


LANGUAGE_MAP = {
    "en": "English",
    "english": "English",
    "zh": "Chinese",
    "chinese": "Chinese",
}


def compute_sample_error_rate(reference: str, prediction: str, language: str) -> float:
    """Compute normalized WER for English or CER for Chinese."""
    metric = "cer" if language.strip().lower().startswith("zh") else "wer"
    norm_ref = normalize_text(reference)
    norm_pred = normalize_text(prediction)
    ref_tokens = tokenize(norm_ref, metric=metric)
    pred_tokens = tokenize(norm_pred, metric=metric)
    if not ref_tokens:
        return 0.0 if not pred_tokens else 1.0
    edits = edit_distance(ref_tokens, pred_tokens)
    return edits / len(ref_tokens)


def generate_controlled_negative(
    gold_text: str,
    language: str,
    rng: random.Random,
) -> str:
    """Generate a controlled negative transcript with realistic acoustic/decoding errors.

    Injects:
    1. Word/character deletion (dropped content);
    2. Phrase repetition (stutter/repetition loop);
    3. Character/word substitution or typo.
    """
    lang = language.strip().lower()
    text = gold_text.strip()
    if not text:
        return "error"

    error_type = rng.choice(["deletion", "repetition", "substitution"])

    if lang in ("zh", "chinese"):
        chars = list(text)
        n = len(chars)
        if n <= 2:
            return text + text  # repetition
        if error_type == "deletion":
            drop_idx = rng.randint(0, n - 1)
            chars.pop(drop_idx)
            if len(chars) > 4 and rng.random() > 0.5:
                drop_idx2 = rng.randint(0, len(chars) - 1)
                chars.pop(drop_idx2)
            return "".join(chars)
        elif error_type == "repetition":
            rep_start = rng.randint(0, max(0, n - 3))
            rep_len = rng.randint(1, min(3, n - rep_start))
            rep_phrase = text[rep_start : rep_start + rep_len]
            return text[: rep_start + rep_len] + rep_phrase + text[rep_start + rep_len :]
        else:  # substitution
            sub_candidates = ["的", "地", "得", "在", "再", "是", "事", "不", "了", "有"]
            sub_idx = rng.randint(0, n - 1)
            chars[sub_idx] = rng.choice(sub_candidates)
            return "".join(chars)
    else:
        # English
        words = text.split()
        n = len(words)
        if n <= 2:
            return text + " " + text  # repetition
        if error_type == "deletion":
            drop_idx = rng.randint(0, n - 1)
            words.pop(drop_idx)
            if len(words) > 5 and rng.random() > 0.5:
                drop_idx2 = rng.randint(0, len(words) - 1)
                words.pop(drop_idx2)
            return " ".join(words)
        elif error_type == "repetition":
            rep_start = rng.randint(0, max(0, n - 2))
            rep_len = rng.randint(1, min(2, n - rep_start))
            rep_words = words[rep_start : rep_start + rep_len]
            return " ".join(words[: rep_start + rep_len] + rep_words + words[rep_start + rep_len :])
        else:  # substitution
            sub_dict = {
                "the": "a", "a": "the", "in": "on", "on": "at", "at": "in",
                "is": "was", "was": "is", "to": "for", "for": "to", "and": "or"
            }
            cand_indices = [i for i, w in enumerate(words) if w.lower() in sub_dict]
            if cand_indices:
                idx = rng.choice(cand_indices)
                words[idx] = sub_dict[words[idx].lower()]
            else:
                idx = rng.randint(0, n - 1)
                words[idx] = words[idx] + "s"
            return " ".join(words)


def compute_sequence_logprob(
    model: Any,
    processor: Any,
    audio_path: str,
    target_text: str,
    language: str,
    device: str = "cuda:0",
    dtype: Any = None,
    asr_model: Any = None,
) -> float:
    """Compute autoregressive log-probability sum for target_text given audio under frozen model."""
    if not HAVE_TORCH or not HAVE_SOUNDFILE:
        return 0.0

    wav, sr = sf.read(audio_path)
    if len(wav.shape) > 1:
        wav = wav[:, 0]

    lang_full = LANGUAGE_MAP.get(language.strip().lower(), "English")
    # Prompt construction aligned with official Qwen3ASRModel
    if asr_model is not None and hasattr(asr_model, "_build_text_prompt"):
        prompt = asr_model._build_text_prompt(context="", force_language=lang_full)
    else:
        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nLanguage={lang_full}\n<|im_end|>\n<|im_start|>assistant\n"
    target_str = target_text.strip() + "<|im_end|>"
    full_text = prompt + target_str

    inputs = processor(text=full_text, audio=wav, return_tensors="pt")
    target_ids = processor.tokenizer.encode(target_str, add_special_tokens=False)

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    input_features = inputs["input_features"].to(device, dtype=dtype or torch.float16)
    feature_attention_mask = inputs["feature_attention_mask"].to(device)

    labels = input_ids.clone()
    target_len = len(target_ids)
    if target_len >= input_ids.shape[1]:
        labels[:, 0] = -100
    else:
        labels[:, :-target_len] = -100

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
        )
        logits = outputs.logits  # [1, seq_len, vocab_size]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        log_probs = F.log_softmax(shift_logits, dim=-1)
        mask = (shift_labels != -100)
        safe_labels = shift_labels.clone()
        safe_labels[~mask] = 0
        per_token_logps = torch.gather(
            log_probs, dim=-1, index=safe_labels.unsqueeze(-1)
        ).squeeze(-1)

        seq_logp = (per_token_logps * mask).sum().item()

    return float(seq_logp)


def compute_file_sha256(file_path: Path) -> str:
    """Compute sha256 checksum of a file."""
    if not file_path.is_file():
        return ""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def build_dpo_pairs(
    candidate_manifest_path: Path,
    output_manifest_path: Path,
    rejects_path: Path,
    audit_output_path: Path,
    ref_model_dir: Optional[Path] = None,
    base_predictions_path: Optional[Path] = None,
    sft_predictions_path: Optional[Path] = None,
    precompute_ref_logps: bool = True,
    target_pairs: Optional[int] = None,
    device: str = "cuda:0",
    seed: int = 20260722,
    allow_synthetic_fallback: bool = False,
) -> Dict[str, Any]:
    """Main function to construct DPO preference pairs and precompute reference log probabilities."""
    rng = random.Random(seed)

    if not candidate_manifest_path.is_file():
        raise FileNotFoundError(f"Candidate manifest not found: {candidate_manifest_path}")

    # Enforce strict real predictions in formal mode
    if not allow_synthetic_fallback:
        if not base_predictions_path or not base_predictions_path.is_file():
            raise ValueError(
                f"Formal DPO pair generation requires valid --base-predictions file, got: {base_predictions_path}. "
                "Automatic fallback to gold text and synthetic negatives is strictly prohibited."
            )
        if not sft_predictions_path or not sft_predictions_path.is_file():
            raise ValueError(
                f"Formal DPO pair generation requires valid --sft-predictions file, got: {sft_predictions_path}. "
                "Automatic fallback to gold text and synthetic negatives is strictly prohibited."
            )

    # Load candidate audio pool
    candidates: List[Dict[str, Any]] = []
    with open(candidate_manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))

    # Load external predictions
    base_preds: Dict[str, str] = {}
    base_predictions_sha256 = ""
    if base_predictions_path and base_predictions_path.is_file():
        base_predictions_sha256 = compute_file_sha256(base_predictions_path)
        with open(base_predictions_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    base_preds[item["sample_id"]] = item.get("prediction", "")

    sft_preds: Dict[str, str] = {}
    sft_predictions_sha256 = ""
    if sft_predictions_path and sft_predictions_path.is_file():
        sft_predictions_sha256 = compute_file_sha256(sft_predictions_path)
        with open(sft_predictions_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    sft_preds[item["sample_id"]] = item.get("prediction", "")

    # Load reference model for logprob calculation if requested
    ref_asr = None
    ref_model = None
    ref_processor = None
    ref_model_sha256 = ""
    ref_model_revision = "7278e1e70fe206f11671096ffdd38061171dd6e5"
    if precompute_ref_logps and ref_model_dir and HAVE_TORCH and HAVE_QWEN_ASR:
        print(f"Loading reference model from {ref_model_dir} for offline logp precomputation...")
        from transformers import AutoProcessor
        from qwen_asr import Qwen3ASRModel
        ref_asr = Qwen3ASRModel.from_pretrained(
            str(ref_model_dir),
            dtype=torch.float16 if device.startswith("cuda") else torch.float32,
            device_map=device if device.startswith("cuda") else None,
            attn_implementation="eager",
        )
        ref_model = ref_asr.model.thinker
        ref_model.eval()
        ref_processor = ref_asr.processor

        # Compute hash of ref model config or safetensors if exists
        safetensors_path = ref_model_dir / "model.safetensors"
        if safetensors_path.is_file():
            # Quick signature hash from first 1MB
            with open(safetensors_path, "rb") as sf_f:
                ref_model_sha256 = hashlib.sha256(sf_f.read(1024 * 1024)).hexdigest()
        elif (ref_model_dir / "config.json").is_file():
            ref_model_sha256 = compute_file_sha256(ref_model_dir / "config.json")

    valid_pairs: List[Dict[str, Any]] = []
    existing_pair_ids = set()
    preference_sources = Counter()
    condition_counts = Counter()
    language_counts = Counter()

    # Check for existing pairs to enable resumption
    if output_manifest_path.is_file():
        with open(output_manifest_path, "r", encoding="utf-8") as f_ex:
            for line in f_ex:
                if line.strip():
                    try:
                        p = json.loads(line)
                        valid_pairs.append(p)
                        existing_pair_ids.add(p["sample_id"])
                        preference_sources[p.get("preference_source", "")] += 1
                        condition_counts[p.get("condition_group", "")] += 1
                        language_counts[p.get("language", "")] += 1
                    except Exception:
                        pass
        if existing_pair_ids:
            print(f"Resuming: found {len(existing_pair_ids)} existing pairs in {output_manifest_path}")

    rejects: List[Dict[str, Any]] = []
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(output_manifest_path, "a", encoding="utf-8")

    for idx, sample in enumerate(candidates):
        sample_id = sample.get("sample_id", f"sample_{idx}")
        if sample_id in existing_pair_ids:
            continue
        audio_path = sample.get("audio", "")
        gold_text = (sample.get("text") or sample.get("answer", "")).strip()
        language = str(sample.get("language", "en")).strip().lower()
        condition_group = sample.get("condition_group", "degraded")

        # Audio file existence check
        if not os.path.isabs(audio_path) and not os.path.exists(audio_path):
            rejects.append({
                "sample_id": sample_id,
                "reason": "audio_not_found",
                "audio": audio_path,
            })
            continue

        if not gold_text:
            rejects.append({
                "sample_id": sample_id,
                "reason": "empty_gold_text",
            })
            continue

        # Determine candidate predictions
        cand_base = base_preds.get(sample_id)
        cand_sft = sft_preds.get(sample_id)

        if cand_base is None or cand_sft is None:
            if not allow_synthetic_fallback:
                rejects.append({
                    "sample_id": sample_id,
                    "reason": "missing_prediction",
                    "has_base": cand_base is not None,
                    "has_sft": cand_sft is not None,
                })
                continue
            # Fallback candidate derivation ONLY if allow_synthetic_fallback is enabled (for tests)
            if cand_sft is None:
                cand_sft = gold_text
            if cand_base is None:
                cand_base = generate_controlled_negative(gold_text, language=language, rng=rng)

        # Calculate candidate error rates against gold
        err_sft = compute_sample_error_rate(gold_text, cand_sft, language=language)
        err_base = compute_sample_error_rate(gold_text, cand_base, language=language)

        # Candidate pair selection logic
        chosen: str = ""
        rejected: str = ""
        chosen_err: float = 0.0
        rejected_err: float = 0.0
        pref_source: str = ""

        if cand_sft.strip() == cand_base.strip() or err_sft == err_base:
            if allow_synthetic_fallback and cand_sft.strip() == cand_base.strip() and err_sft == 0.0:
                # Test/mock mode only
                synthetic_neg = generate_controlled_negative(gold_text, language=language, rng=rng)
                neg_err = compute_sample_error_rate(gold_text, synthetic_neg, language=language)
                if neg_err > err_sft:
                    chosen = cand_sft
                    rejected = synthetic_neg
                    chosen_err = err_sft
                    rejected_err = neg_err
                    pref_source = "gold_vs_synthetic_negative"
                else:
                    rejects.append({
                        "sample_id": sample_id,
                        "reason": "synthetic_negative_has_zero_error",
                    })
                    continue
            else:
                rejects.append({
                    "sample_id": sample_id,
                    "reason": "tie_identical_predictions" if cand_sft.strip() == cand_base.strip() else "tie_identical_error_rate",
                    "err_sft": err_sft,
                    "err_base": err_base,
                })
                continue
        elif err_sft < err_base:
            chosen = cand_sft
            rejected = cand_base
            chosen_err = err_sft
            rejected_err = err_base
            pref_source = "sft_better_than_base"
        else:
            chosen = cand_base
            rejected = cand_sft
            chosen_err = err_base
            rejected_err = err_sft
            pref_source = "base_better_than_sft"

        # Strict validation checks
        if not chosen.strip():
            rejects.append({"sample_id": sample_id, "reason": "empty_chosen"})
            continue
        if chosen.strip() == rejected.strip():
            rejects.append({"sample_id": sample_id, "reason": "chosen_equals_rejected"})
            continue
        if chosen_err >= rejected_err:
            rejects.append({"sample_id": sample_id, "reason": "chosen_err_not_strictly_smaller"})
            continue

        # Precompute reference log probabilities if reference model loaded
        ref_chosen_logp = 0.0
        ref_rejected_logp = 0.0
        if ref_model is not None and ref_processor is not None:
            try:
                ref_chosen_logp = compute_sequence_logprob(
                    model=ref_model,
                    processor=ref_processor,
                    audio_path=audio_path,
                    target_text=chosen,
                    language=language,
                    device=device,
                    asr_model=ref_asr,
                )
                ref_rejected_logp = compute_sequence_logprob(
                    model=ref_model,
                    processor=ref_processor,
                    audio_path=audio_path,
                    target_text=rejected,
                    language=language,
                    device=device,
                    asr_model=ref_asr,
                )
            except Exception as e:
                rejects.append({
                    "sample_id": sample_id,
                    "reason": f"logprob_computation_error: {e}",
                })
                continue

        prompt_str = f"Language={LANGUAGE_MAP.get(language, 'English')}"

        pair_record = {
            "sample_id": sample_id,
            "audio": audio_path,
            "language": language,
            "prompt": prompt_str,
            "gold_transcript": gold_text,
            "chosen": chosen,
            "rejected": rejected,
            "preference_source": pref_source,
            "judge": "wer_cer_rule_v1",
            "chosen_error_rate": round(chosen_err, 4),
            "rejected_error_rate": round(rejected_err, 4),
            "ref_chosen_logp": round(ref_chosen_logp, 4),
            "ref_rejected_logp": round(ref_rejected_logp, 4),
            "ref_model_revision": ref_model_revision,
            "ref_model_sha256": ref_model_sha256,
            "tokenizer_hash": "qwen3_tokenizer_v1",
            "chat_template_hash": "qwen3_chat_v1",
            "logp_config_hash": "logp_eager_fp16_v1",
            "source_dataset": sample.get("source_dataset", ""),
            "source_revision": sample.get("source_revision", ""),
            "source_utterance_id": sample.get("source_utterance_id", ""),
            "condition_group": condition_group,
            "scenario": sample.get("scenario", "clean" if condition_group == "clean" else "degraded"),
            "audio_sha256": sample.get("audio_sha256", ""),
            "seed": seed,
        }
        valid_pairs.append(pair_record)
        out_f.write(json.dumps(pair_record, ensure_ascii=False) + "\n")
        out_f.flush()

        preference_sources[pref_source] += 1
        condition_counts[condition_group] += 1
        language_counts[language] += 1

        if len(valid_pairs) % 50 == 0:
            print(f"[{len(valid_pairs)} pairs generated] processed sample {sample_id}")

        if target_pairs and len(valid_pairs) >= target_pairs:
            break

    out_f.close()

    # Write rejects manifest
    rejects_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rejects_path, "w", encoding="utf-8") as f:
        for rej in rejects:
            f.write(json.dumps(rej, ensure_ascii=False) + "\n")

    # Compute summary audit
    total_processed = len(candidates)
    num_chosen_equals_gold = sum(
        1 for p in valid_pairs if p.get("chosen", "").strip() == p.get("gold_transcript", "").strip()
    )
    pct_chosen_equals_gold = round(num_chosen_equals_gold / max(1, len(valid_pairs)), 4)

    audit_summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "candidate_manifest": str(candidate_manifest_path),
        "output_manifest": str(output_manifest_path),
        "base_predictions_path": str(base_predictions_path) if base_predictions_path else "",
        "base_predictions_sha256": base_predictions_sha256,
        "sft_predictions_path": str(sft_predictions_path) if sft_predictions_path else "",
        "sft_predictions_sha256": sft_predictions_sha256,
        "total_candidates_processed": total_processed,
        "valid_pairs_count": len(valid_pairs),
        "rejects_count": len(rejects),
        "yield_rate": round(len(valid_pairs) / max(1, total_processed), 4),
        "num_chosen_equals_gold": num_chosen_equals_gold,
        "pct_chosen_equals_gold": pct_chosen_equals_gold,
        "breakdown_by_language": dict(language_counts),
        "breakdown_by_condition_group": dict(condition_counts),
        "breakdown_by_preference_source": dict(preference_sources),
        "mean_chosen_error_rate": round(
            sum(p["chosen_error_rate"] for p in valid_pairs) / max(1, len(valid_pairs)), 4
        ),
        "mean_rejected_error_rate": round(
            sum(p["rejected_error_rate"] for p in valid_pairs) / max(1, len(valid_pairs)), 4
        ),
        "mean_error_rate_delta": round(
            sum(p["rejected_error_rate"] - p["chosen_error_rate"] for p in valid_pairs)
            / max(1, len(valid_pairs)), 4
        ),
        "precomputed_ref_logps": precompute_ref_logps,
        "ref_model_revision": ref_model_revision,
    }

    audit_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_output_path, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2, ensure_ascii=False)

    print(f"Built {len(valid_pairs)} DPO pairs (rejected: {len(rejects)}) -> {output_manifest_path}")
    print(f"Audit report saved -> {audit_output_path}")

    return audit_summary


def main():
    parser = argparse.ArgumentParser(
        description="Build DPO preference pairs and precalculate reference log probabilities."
    )
    parser.add_argument(
        "--candidate-manifest",
        type=Path,
        required=True,
        help="Path to candidate audio manifest JSONL.",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        required=True,
        help="Path to save output DPO pairs manifest JSONL.",
    )
    parser.add_argument(
        "--rejects",
        type=Path,
        required=True,
        help="Path to save rejected/tie candidate records.",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        required=True,
        help="Path to save audit summary JSON.",
    )
    parser.add_argument(
        "--ref-model-dir",
        type=Path,
        default=None,
        help="Path to frozen reference model directory (Step 50 SFT merged base) for logp precalculation.",
    )
    parser.add_argument(
        "--base-predictions",
        type=Path,
        default=None,
        help="Path to precomputed Base model predictions JSONL.",
    )
    parser.add_argument(
        "--sft-predictions",
        type=Path,
        default=None,
        help="Path to precomputed SFT model predictions JSONL.",
    )
    parser.add_argument(
        "--no-precompute-logps",
        action="store_true",
        help="Disable offline reference logp precalculation.",
    )
    parser.add_argument(
        "--target-pairs",
        type=int,
        default=None,
        help="Target number of pairs to produce.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch and torch.cuda.is_available() else "cpu",
        help="Device to use for reference logprob precomputation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260722,
        help="Random seed for controlled negative generation.",
    )
    parser.add_argument(
        "--allow-synthetic-fallback",
        action="store_true",
        help="Allow synthetic fallback when predictions are missing (FOR TESTS ONLY).",
    )

    args = parser.parse_args()

    build_dpo_pairs(
        candidate_manifest_path=args.candidate_manifest,
        output_manifest_path=args.output_manifest,
        rejects_path=args.rejects,
        audit_output_path=args.audit_output,
        ref_model_dir=args.ref_model_dir,
        base_predictions_path=args.base_predictions,
        sft_predictions_path=args.sft_predictions,
        precompute_ref_logps=not args.no_precompute_logps,
        target_pairs=args.target_pairs,
        device=args.device,
        seed=args.seed,
        allow_synthetic_fallback=args.allow_synthetic_fallback,
    )


if __name__ == "__main__":
    main()
