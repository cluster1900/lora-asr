from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from train.train_qwen3_asr_a2s import (
    EXPECTED_GROUPS,
    active_target_names,
    classify_target,
    configure_phase_trainability,
    evaluate_canary_gate,
    expected_target_specs,
    find_latest_checkpoint,
    materialize_curriculum,
    prepare_rows,
    resume_value_for_phase,
    target_group_counts,
    target_map_hash,
    target_specs_from_records,
    validate_config,
    validate_target_map,
)


def full_module_records() -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for layer in range(24):
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            records.append(
                {
                    "module_name": f"model.thinker.audio_tower.layers.{layer}.self_attn.{name}",
                    "class_name": "Linear",
                }
            )
        for name in ("fc1", "fc2"):
            records.append(
                {
                    "module_name": f"model.thinker.audio_tower.layers.{layer}.{name}",
                    "class_name": "Linear",
                }
            )
    for name in ("conv_out", "proj1", "proj2"):
        records.append(
            {"module_name": f"model.thinker.audio_tower.{name}", "class_name": "Linear"}
        )
    for layer in range(28):
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            records.append(
                {
                    "module_name": f"model.thinker.model.layers.{layer}.self_attn.{name}",
                    "class_name": "Linear",
                }
            )
        for name in ("gate_proj", "up_proj", "down_proj"):
            records.append(
                {
                    "module_name": f"model.thinker.model.layers.{layer}.mlp.{name}",
                    "class_name": "Linear",
                }
            )
    return records


class FakeParameter:
    def __init__(self) -> None:
        self.requires_grad = True

    @staticmethod
    def numel() -> int:
        return 1


class FakePeftModel:
    def __init__(self, canonical_names: list[str]) -> None:
        self.parameters = [
            (
                f"base_model.model.{name}.lora_A.default.weight",
                FakeParameter(),
            )
            for name in canonical_names
        ]
        self.parameters.append(("base_model.model.thinker.lm_head.weight", FakeParameter()))

    def named_parameters(self):
        return iter(self.parameters)


class TargetContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.targets = target_specs_from_records(full_module_records())

    def test_exact_group_contract_and_hash(self) -> None:
        validate_target_map(self.targets)
        self.assertEqual(target_group_counts(self.targets), EXPECTED_GROUPS)
        self.assertEqual(len(self.targets), 343)
        self.assertEqual(target_map_hash(self.targets), target_map_hash(list(reversed(self.targets))))

    def test_generated_contract_matches_runtime_shape_fixture(self) -> None:
        self.assertEqual(
            target_map_hash(expected_target_specs()),
            target_map_hash(self.targets),
        )

    def test_phase_scopes_are_27_196_343(self) -> None:
        self.assertEqual(len(active_target_names(self.targets, "upper_audio_projection")), 27)
        self.assertEqual(len(active_target_names(self.targets, "decoder")), 196)
        self.assertEqual(len(active_target_names(self.targets, "all")), 343)

    def test_runtime_type_and_forbidden_modules_are_excluded(self) -> None:
        self.assertIsNone(classify_target("model.thinker.audio_tower.conv2d1", "Conv2d"))
        self.assertIsNone(classify_target("model.thinker.audio_tower.conv_out", "Conv2d"))
        self.assertIsNone(classify_target("model.thinker.lm_head", "Linear"))
        self.assertEqual(
            classify_target("model.thinker.audio_tower.conv_out", "Linear"),
            ("projection", None),
        )

    def test_trainability_switch_freezes_everything_else(self) -> None:
        model = FakePeftModel([target.canonical_name for target in self.targets])
        summary = configure_phase_trainability(
            model,
            self.targets,
            {"active_scope": "upper_audio_projection"},
            upper_audio_layers=4,
        )
        self.assertEqual(summary["active_target_count"], 27)
        enabled = [parameter for _, parameter in model.named_parameters() if parameter.requires_grad]
        self.assertEqual(len(enabled), 27)


class CurriculumTest(unittest.TestCase):
    def test_equal_ordered_segments_keep_total_two_epoch_exposure(self) -> None:
        rows = [
            {
                "sample_id": f"sample-{index:03d}",
                "audio": f"{index}.wav",
                "answer": "text",
                "base_error_rate": (index % 6) / 10,
            }
            for index in range(30)
        ]
        first = materialize_curriculum(rows, 2, [0.3, 0.5, 0.7], seed=7)
        second = materialize_curriculum(rows, 2, [0.3, 0.5, 0.7], seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 60)
        self.assertEqual([row["curriculum_segment"] for row in first[:20]], [0] * 20)
        self.assertEqual([row["curriculum_segment"] for row in first[20:40]], [1] * 20)
        self.assertEqual([row["curriculum_segment"] for row in first[40:]], [2] * 20)
        self.assertTrue(all(row["base_error_rate"] < row["curriculum_threshold"] for row in first))


class PipelineContractTest(unittest.TestCase):
    def test_training_manifest_requires_current_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "sample.wav"
            audio.touch()
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps({"audio": str(audio), "answer": "text", "duration_s": 1.0}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "no sample_id"):
                prepare_rows(manifest, root, {"name": "phase_2"}, seed=1)

    def test_canary_gate_uses_base_relative_metrics(self) -> None:
        base = {
            "overall": {
                "language_macro_error_rate": 0.4,
                "inference_error_rate": 0.0,
                "empty_output_rate": 0.01,
                "repeat_like_output_rate": 0.01,
                "too_long_output_rate": 0.0,
                "hallucination_like_output_rate": 0.0,
            }
        }
        adapter = {
            "overall": {
                "language_macro_error_rate": 0.42,
                "inference_error_rate": 0.01,
                "empty_output_rate": 0.02,
                "repeat_like_output_rate": 0.01,
                "too_long_output_rate": 0.0,
                "hallucination_like_output_rate": 0.0,
            }
        }
        result = evaluate_canary_gate(
            base,
            adapter,
            {
                "max_relative_robust_regression": 0.15,
                "min_valid_output_rate": 0.95,
                "max_failure_rate_increase": 0.05,
            },
        )
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["relative_robust_regression"], 0.05)

    def test_config_rejects_effective_batch_drift(self) -> None:
        config = {
            "schema_version": 1,
            "model": {"dtype": "bfloat16"},
            "data": {},
            "lora": {},
            "training": {
                "per_device_batch_size": 4,
                "gradient_accumulation_steps": 16,
                "effective_batch_size": 64,
            },
            "phases": [
                {"name": "phase_1"},
                {"name": "phase_2"},
                {"name": "phase_3"},
            ],
            "evaluation": {},
        }
        with self.assertRaisesRegex(ValueError, "Effective batch"):
            validate_config(config)

    def test_latest_checkpoint_is_numeric(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "checkpoint-9").mkdir()
            (root / "checkpoint-100").mkdir()
            (root / "checkpoint-bad").mkdir()
            self.assertEqual(find_latest_checkpoint(root), root / "checkpoint-100")

    def test_explicit_resume_applies_to_only_its_phase(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "phase_2" / "checkpoint-10"
            checkpoint.mkdir(parents=True)
            self.assertEqual(
                resume_value_for_phase(str(checkpoint), root / "phase_2"),
                str(checkpoint.resolve()),
            )
            self.assertEqual(resume_value_for_phase(str(checkpoint), root / "phase_3"), "")


if __name__ == "__main__":
    unittest.main()
