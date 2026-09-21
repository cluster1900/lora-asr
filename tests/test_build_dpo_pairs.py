from __future__ import annotations

import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_dpo_pairs


class BuildDPOPairsTest(unittest.TestCase):
    def test_compute_sample_error_rate(self) -> None:
        # Identical
        self.assertAlmostEqual(build_dpo_pairs.compute_sample_error_rate("hello world", "hello world", "en"), 0.0)
        self.assertAlmostEqual(build_dpo_pairs.compute_sample_error_rate("今天天气很好", "今天天气很好", "zh"), 0.0)

        # Different
        err_en = build_dpo_pairs.compute_sample_error_rate("hello world", "hello there world", "en")
        self.assertGreater(err_en, 0.0)

        err_zh = build_dpo_pairs.compute_sample_error_rate("今天天气很好", "今天天气不好", "zh")
        self.assertGreater(err_zh, 0.0)

    def test_generate_controlled_negative(self) -> None:
        rng = random.Random(42)
        # English negative
        gold_en = "the quick brown fox jumps over the lazy dog"
        neg_en = build_dpo_pairs.generate_controlled_negative(gold_en, "en", rng)
        self.assertNotEqual(gold_en, neg_en)
        err_en = build_dpo_pairs.compute_sample_error_rate(gold_en, neg_en, "en")
        self.assertGreater(err_en, 0.0)

        # Chinese negative
        gold_zh = "北京天安门广场上红旗飘扬"
        neg_zh = build_dpo_pairs.generate_controlled_negative(gold_zh, "zh", rng)
        self.assertNotEqual(gold_zh, neg_zh)
        err_zh = build_dpo_pairs.compute_sample_error_rate(gold_zh, neg_zh, "zh")
        self.assertGreater(err_zh, 0.0)

    def test_build_dpo_pairs_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_p = Path(tmpdir)

            # Create mock audio file
            audio_path = tmp_p / "sample1.wav"
            audio_path.write_bytes(b"RIFF" + b"\x00" * 100)

            candidate_manifest = tmp_p / "candidates.jsonl"
            output_manifest = tmp_p / "dpo_pairs.jsonl"
            rejects_manifest = tmp_p / "rejects.jsonl"
            audit_path = tmp_p / "audit.json"

            samples = [
                {
                    "sample_id": "s1",
                    "audio": str(audio_path),
                    "text": "THIS IS A TEST RECORDING",
                    "language": "en",
                    "condition_group": "degraded",
                    "scenario": "noise",
                },
                {
                    "sample_id": "s2",
                    "audio": str(audio_path),
                    "text": "测试数据样本内容",
                    "language": "zh",
                    "condition_group": "clean",
                    "scenario": "clean",
                },
            ]

            with open(candidate_manifest, "w", encoding="utf-8") as f:
                for s in samples:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")

            # Mock predictions:
            # s1: SFT is perfect, Base has error -> sft_better_than_base
            # s2: Identical prediction (tie) -> must be rejected in formal mode!
            # s3: Base has lower error than SFT -> base_better_than_sft
            samples.append({
                "sample_id": "s3",
                "audio": str(audio_path),
                "text": "SPEECH RECOGNITION PIPELINE",
                "language": "en",
                "condition_group": "degraded",
                "scenario": "distortion",
            })
            with open(candidate_manifest, "w", encoding="utf-8") as f:
                for s in samples:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")

            sft_preds = tmp_p / "sft_preds.jsonl"
            base_preds = tmp_p / "base_preds.jsonl"

            with open(sft_preds, "w", encoding="utf-8") as f:
                f.write(json.dumps({"sample_id": "s1", "prediction": "THIS IS A TEST RECORDING"}) + "\n")
                f.write(json.dumps({"sample_id": "s2", "prediction": "测试数据样本内容"}) + "\n")
                f.write(json.dumps({"sample_id": "s3", "prediction": "SPEECH PIPELINE ERROR"}) + "\n")

            with open(base_preds, "w", encoding="utf-8") as f:
                f.write(json.dumps({"sample_id": "s1", "prediction": "THIS IS TEST RECORD"}) + "\n")
                f.write(json.dumps({"sample_id": "s2", "prediction": "测试数据样本内容"}) + "\n")
                f.write(json.dumps({"sample_id": "s3", "prediction": "SPEECH RECOGNITION PIPELINE"}) + "\n")

            # 1. Test missing prediction files raises ValueError in formal mode
            with self.assertRaises(ValueError):
                build_dpo_pairs.build_dpo_pairs(
                    candidate_manifest_path=candidate_manifest,
                    output_manifest_path=output_manifest,
                    rejects_path=rejects_manifest,
                    audit_output_path=audit_path,
                    base_predictions_path=None,
                    sft_predictions_path=None,
                    allow_synthetic_fallback=False,
                )

            # 2. Test formal mode: strict predictions, ties rejected
            summary = build_dpo_pairs.build_dpo_pairs(
                candidate_manifest_path=candidate_manifest,
                output_manifest_path=output_manifest,
                rejects_path=rejects_manifest,
                audit_output_path=audit_path,
                base_predictions_path=base_preds,
                sft_predictions_path=sft_preds,
                precompute_ref_logps=False,
                seed=20260722,
                allow_synthetic_fallback=False,
            )

            self.assertEqual(summary["total_candidates_processed"], 3)
            # s1 accepted, s2 rejected (tie), s3 accepted
            self.assertEqual(summary["valid_pairs_count"], 2)
            self.assertEqual(summary["rejects_count"], 1)
            self.assertTrue(output_manifest.is_file())
            self.assertTrue(audit_path.is_file())
            self.assertIn("num_chosen_equals_gold", summary)

            with open(output_manifest, "r", encoding="utf-8") as f:
                pairs = [json.loads(line) for line in f]

            self.assertEqual(len(pairs), 2)
            # s1 should have chosen = sft, rejected = base
            p1 = pairs[0]
            self.assertEqual(p1["sample_id"], "s1")
            self.assertLess(p1["chosen_error_rate"], p1["rejected_error_rate"])
            self.assertEqual(p1["preference_source"], "sft_better_than_base")

            # s3 should have chosen = base, rejected = sft
            p3 = pairs[1]
            self.assertEqual(p3["sample_id"], "s3")
            self.assertLess(p3["chosen_error_rate"], p3["rejected_error_rate"])
            self.assertEqual(p3["preference_source"], "base_better_than_sft")

            # 3. Test fallback mode (allow_synthetic_fallback=True)
            output_manifest_fb = tmp_p / "dpo_pairs_fb.jsonl"
            rejects_manifest_fb = tmp_p / "rejects_fb.jsonl"
            audit_path_fb = tmp_p / "audit_fb.json"
            summary_fb = build_dpo_pairs.build_dpo_pairs(
                candidate_manifest_path=candidate_manifest,
                output_manifest_path=output_manifest_fb,
                rejects_path=rejects_manifest_fb,
                audit_output_path=audit_path_fb,
                base_predictions_path=base_preds,
                sft_predictions_path=sft_preds,
                precompute_ref_logps=False,
                seed=20260722,
                allow_synthetic_fallback=True,
            )
            # In fallback mode, s2 generates synthetic negative -> 3 pairs
            self.assertEqual(summary_fb["valid_pairs_count"], 3)


if __name__ == "__main__":
    unittest.main()
