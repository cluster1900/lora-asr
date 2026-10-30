"""Unit and contract tests for Phase 5 sealed test.jsonl benchmark & notebook."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import soundfile as sf

from evaluation.jsonl_io import write_jsonl
from scripts.manifest_audio import render_scenarios
from scripts.manifest_contract import SCENARIOS, load_data_config
from scripts.run_phase5_test_benchmark import (
    PHASE5_BENCHMARK_SCHEMA,
    STAGES,
    ensure_test_audio_verified,
    run_phase5_benchmark,
    validate_published_releases,
    zero_lora_parameters,
)
from train.dpo_release import DPO_PAYLOAD_FILES, DPO_RELEASE_SCHEMA, DPO_RELEASE_SHA_SCHEMA
from train.rl_release import RL_PAYLOAD_FILES, RL_RELEASE_SCHEMA, RL_RELEASE_SHA_SCHEMA
from train.rl_schema import load_rl_config
from train.sft_checkpoint import atomic_write_json, directory_sha256, file_map, sha256_file
from train.sft_release import PAYLOAD_FILES as SFT_PAYLOAD_FILES
from train.sft_release import RELEASE_SCHEMA as SFT_RELEASE_SCHEMA
from train.sft_release import RELEASE_SHA_SCHEMA as SFT_RELEASE_SHA_SCHEMA

ROOT = Path(__file__).resolve().parents[1]


def _make_dummy_release(
    root: Path,
    *,
    stage: str,
    release_id: str,
    status: str,
    publishable: bool,
    config: dict[str, Any],
    parent_release: str | None,
    parent_sha: str | None,
    adapter_bytes: bytes,
) -> dict[str, Any]:
    (root / "adapter").mkdir(parents=True, exist_ok=True)
    (root / "source").mkdir(parents=True, exist_ok=True)
    (root / "adapter" / "adapter.safetensors").write_bytes(adapter_bytes)
    atomic_write_json(root / "adapter" / "adapter_config.json", {"rank": 16})
    atomic_write_json(root / "source" / "gate.json", {"status": "passed"})
    (root / "source" / "scored.jsonl").write_text('{"sample_id": "s1"}\n', encoding="utf-8")
    atomic_write_json(root / "source" / "training_state.json", {"stage": stage})

    if stage == "sft":
        payload_files = list(SFT_PAYLOAD_FILES)
        rel_schema = SFT_RELEASE_SCHEMA
        sha_schema = SFT_RELEASE_SHA_SCHEMA
    elif stage == "dpo":
        atomic_write_json(root / "source" / "preference.json", {"accuracy": 0.65})
        payload_files = list(DPO_PAYLOAD_FILES)
        rel_schema = DPO_RELEASE_SCHEMA
        sha_schema = DPO_RELEASE_SHA_SCHEMA
    else:
        atomic_write_json(root / "source" / "decision.json", {"passed": True})
        payload_files = list(RL_PAYLOAD_FILES)
        rel_schema = RL_RELEASE_SCHEMA
        sha_schema = RL_RELEASE_SHA_SCHEMA

    mapping = file_map(root, payload_files)
    payload_sha = directory_sha256(mapping)
    atomic_write_json(
        root / "sha256.json",
        {
            "schema_version": sha_schema,
            "files": mapping,
            "directory_sha256": payload_sha,
        },
    )
    release = {
        "schema_version": rel_schema,
        "release_id": release_id,
        "status": status,
        "publishable": publishable,
        "stage": stage,
        "source_run_id": f"20261023T000000_{stage}_06b_zh_seed42",
        "source_step": 10,
        "base_model": config["model"]["name"],
        "base_revision": config["model"]["revision"],
        "processor_sha256": config["model"]["processor_sha256"],
        "eval_decode_sha256": config["decode_eval_sha256"],
        "parent_release": parent_release,
        "parent_release_sha256": parent_sha,
        "adapter_sha256": mapping["adapter/adapter.safetensors"],
        "payload_sha256": payload_sha,
        "audit_disposition": "unlistened",
    }
    atomic_write_json(root / "release.json", release)
    return release


def _build_synthetic_test_manifest(td: Path, data_cfg: dict[str, Any], *, keep_only_clean: bool = False) -> tuple[Path, Path]:
    from scripts.manifest_audio import load_mono_16k

    manifest_dir = td / "manifest"
    (manifest_dir / "wavs").mkdir(parents=True, exist_ok=True)
    t = np.linspace(0.0, 0.25, 4000, endpoint=False, dtype=np.float32)
    raw_wave = (0.2 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    raw_src = td / "raw_src.wav"
    sf.write(raw_src, raw_wave, 16000, subtype="PCM_16", format="WAV")
    clean_wave = load_mono_16k(str(raw_src), sf.read)
    source_id = "aishell:BAC009S0764W0130"
    waveforms = render_scenarios(clean_wave, int(data_cfg["seed"]), source_id, data_cfg["augmentation"])

    rows: list[dict[str, Any]] = []
    sha_files: list[dict[str, Any]] = []
    for sc in SCENARIOS:
        rel_audio = f"wavs/BAC009S0764W0130__{sc}.wav"
        wav_path = manifest_dir / rel_audio
        sf.write(wav_path, waveforms[sc], 16000, subtype="PCM_16", format="WAV")
        wav_sha = sha256_file(wav_path)
        sha_files.append({"name": rel_audio, "bytes": wav_path.stat().st_size, "sha256": wav_sha})
        if keep_only_clean and sc != "clean":
            wav_path.unlink()
        rows.append(
            {
                "schema_version": "qwen3-asr-manifest-v1",
                "sample_id": f"BAC009S0764W0130__{sc}",
                "source_utterance_id": source_id,
                "speaker_id": "S0764",
                "official_split": "test",
                "split": "test",
                "scenario": sc,
                "language": "zh",
                "audio": rel_audio,
                "text": "你好世界",
            }
        )
    test_manifest = manifest_dir / "test.jsonl"
    write_jsonl(test_manifest, rows)
    sha_files.append({"name": "test.jsonl", "bytes": test_manifest.stat().st_size, "sha256": sha256_file(test_manifest)})
    sha_index = manifest_dir / "sha256.json"
    atomic_write_json(
        sha_index,
        {
            "schema_version": "qwen3-asr-manifest-sha-v1",
            "generated_at": "2026-10-23T00:00:00Z",
            "files": sha_files,
        },
    )
    return test_manifest, sha_index


class FakeTensor:
    def __init__(self, value: float = 1.0) -> None:
        self.value = value

    @property
    def data(self) -> "FakeTensor":
        return self

    def zero_(self) -> None:
        self.value = 0.0

    def detach(self) -> "FakeTensor":
        return self

    def abs(self) -> "FakeTensor":
        return FakeTensor(abs(self.value))

    def sum(self) -> "FakeTensor":
        return self

    def item(self) -> float:
        return float(self.value)


class FakeModel:
    def __init__(self) -> None:
        self._params = {
            "thinker.model.layers.0.self_attn.q_proj.lora_A.default.weight": FakeTensor(1.0),
            "thinker.model.layers.0.self_attn.q_proj.lora_B.default.weight": FakeTensor(1.0),
        }
        self.loaded_adapter: str = "init"

    def named_parameters(self) -> list[tuple[str, FakeTensor]]:
        return list(self._params.items())


class Phase5BenchmarkTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config_path = ROOT / "configs" / "train" / "rl_pilot_aligned.yaml"
        self.data_config_path = ROOT / "configs" / "data" / "manifest_mvp.yaml"
        self.config = load_rl_config(self.config_path)
        self.data_cfg = load_data_config(self.data_config_path)

    def test_ensure_test_audio_renders_missing_degraded_scenarios_and_verifies_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            test_manifest, sha_index = _build_synthetic_test_manifest(td, self.data_cfg, keep_only_clean=True)
            info = ensure_test_audio_verified(
                test_manifest,
                sha_index,
                self.data_config_path,
                expected_test_sha256=sha256_file(test_manifest),
            )
            self.assertEqual(info["rendered_files"], 4)
            self.assertEqual(info["verified_files"], 5)

            corrupt_wav = test_manifest.parent / "wavs" / "BAC009S0764W0130__noise.wav"
            corrupt_wav.write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "sha256 mismatch"):
                ensure_test_audio_verified(
                    test_manifest,
                    sha_index,
                    self.data_config_path,
                    expected_test_sha256=sha256_file(test_manifest),
                )

    def test_ensure_test_audio_reconciles_cross_arch_noise_ulp_drift(self) -> None:
        from scripts.manifest_audio import load_mono_16k
        from scripts.manifest_contract import scenario_seed

        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            test_manifest, sha_index = _build_synthetic_test_manifest(td, self.data_cfg, keep_only_clean=True)
            manifest_dir = test_manifest.parent
            clean_path = manifest_dir / "wavs" / "BAC009S0764W0130__clean.wav"
            clean_wave = load_mono_16k(str(clean_path), sf.read)
            source_id = "aishell:BAC009S0764W0130"
            seed = scenario_seed(int(self.data_cfg["seed"]), source_id, "noise")
            gen = np.random.default_rng(seed)
            noise = gen.normal(0.0, 1.0, size=clean_wave.shape).astype(np.float32)
            sp = float(np.mean(np.square(clean_wave)))
            npow = float(np.mean(np.square(noise)))
            snr_db = float(self.data_cfg["augmentation"]["noise"]["snr_db"])
            s0 = np.float32(np.sqrt(sp / (npow * (10.0 ** (snr_db / 10.0)))))
            s_plus3 = s0
            for _ in range(3):
                s_plus3 = np.nextafter(s_plus3, np.float32(np.inf))
            tmp_noise = td / "ulp3_noise.wav"
            sf.write(tmp_noise, np.clip(clean_wave + noise * s_plus3, -1.0, 1.0), 16000, subtype="PCM_16", format="WAV")
            ulp3_sha = sha256_file(tmp_noise)

            sha_payload = json.loads(sha_index.read_text(encoding="utf-8"))
            for item in sha_payload["files"]:
                if item["name"] == "wavs/BAC009S0764W0130__noise.wav":
                    item["sha256"] = ulp3_sha
            atomic_write_json(sha_index, sha_payload)

            info = ensure_test_audio_verified(
                test_manifest,
                sha_index,
                self.data_config_path,
                expected_test_sha256=sha256_file(test_manifest),
            )
            self.assertEqual(info["verified_files"], 5)
            self.assertEqual(
                sha256_file(manifest_dir / "wavs" / "BAC009S0764W0130__noise.wav"),
                ulp3_sha,
            )

    def test_validate_published_releases_enforces_status_and_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            sft_rel = _make_dummy_release(
                td / "sft",
                stage="sft",
                release_id="sft_06b_zh_20261015T234939_step000030",
                status="published",
                publishable=True,
                config=self.config,
                parent_release=None,
                parent_sha=None,
                adapter_bytes=b"sft-adapter",
            )
            dpo_rel = _make_dummy_release(
                td / "dpo",
                stage="dpo",
                release_id="dpo_06b_zh_20261022T133007_step000040",
                status="published",
                publishable=True,
                config=self.config,
                parent_release=sft_rel["release_id"],
                parent_sha=sft_rel["adapter_sha256"],
                adapter_bytes=b"dpo-adapter",
            )
            _make_dummy_release(
                td / "rl",
                stage="rl",
                release_id="rl_06b_zh_20261023T042034_step000005",
                status="published",
                publishable=True,
                config=self.config,
                parent_release=dpo_rel["release_id"],
                parent_sha=dpo_rel["adapter_sha256"],
                adapter_bytes=b"rl-adapter",
            )
            releases = validate_published_releases(
                td / "sft",
                td / "dpo",
                td / "rl",
                self.config,
                enforce_pinned_ids=False,
            )
            self.assertEqual(set(releases), {"sft", "dpo", "rl"})

            # Staged release must be rejected
            _make_dummy_release(
                td / "rl_staged",
                stage="rl",
                release_id="rl_06b_zh_20261023T042034_step000005",
                status="staged",
                publishable=False,
                config=self.config,
                parent_release=dpo_rel["release_id"],
                parent_sha=dpo_rel["adapter_sha256"],
                adapter_bytes=b"rl-adapter",
            )
            with self.assertRaisesRegex(ValueError, "rl release is not published"):
                validate_published_releases(
                    td / "sft",
                    td / "dpo",
                    td / "rl_staged",
                    self.config,
                    enforce_pinned_ids=False,
                )

    def test_run_phase5_benchmark_evaluates_four_stages_resumes_and_syncs_drive(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            test_manifest, sha_index = _build_synthetic_test_manifest(td, self.data_cfg, keep_only_clean=True)
            sft_rel = _make_dummy_release(
                td / "sft",
                stage="sft",
                release_id="sft_06b_zh_20261015T234939_step000030",
                status="published",
                publishable=True,
                config=self.config,
                parent_release=None,
                parent_sha=None,
                adapter_bytes=b"sft-adapter",
            )
            dpo_rel = _make_dummy_release(
                td / "dpo",
                stage="dpo",
                release_id="dpo_06b_zh_20261022T133007_step000040",
                status="published",
                publishable=True,
                config=self.config,
                parent_release=sft_rel["release_id"],
                parent_sha=sft_rel["adapter_sha256"],
                adapter_bytes=b"dpo-adapter",
            )
            rl_rel = _make_dummy_release(
                td / "rl",
                stage="rl",
                release_id="rl_06b_zh_20261023T042034_step000005",
                status="published",
                publishable=True,
                config=self.config,
                parent_release=dpo_rel["release_id"],
                parent_sha=dpo_rel["adapter_sha256"],
                adapter_bytes=b"rl-adapter",
            )

            stage_transcripts = {
                "base_zero_lora": "你号世界",  # 1 edit per row -> 5 edits total
                sft_rel["adapter_sha256"]: "你好世介",  # 1 edit per row -> 5 edits total
                dpo_rel["adapter_sha256"]: "你好世界",  # 0 edits
                rl_rel["adapter_sha256"]: "你好世界",  # 0 edits
            }

            def fake_model_loader(_cfg: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
                model = FakeModel()
                count = zero_lora_parameters(model)
                self.assertEqual(count, 1)
                return model, object(), {}

            def fake_adapter_loader(model: FakeModel, path: Path) -> None:
                model.loaded_adapter = sha256_file(path)

            def fake_decode(
                _model: Any,
                _processor: Any,
                rows: list[dict[str, Any]],
                _manifest: Path,
                _cfg: dict[str, Any],
                _step: int,
                adapter_sha: str,
                on_progress: Any = None,
                on_row: Any = None,
            ) -> list[dict[str, Any]]:
                preds = []
                text = stage_transcripts[adapter_sha]
                for idx, row in enumerate(rows, start=1):
                    pred = {
                        "sample_id": row["sample_id"],
                        "text": row["text"],
                        "prediction_raw": text,
                        "generation_raw": text,
                        "token_ids": [1, 2],
                        "stop_reason": "eos",
                        "error": None,
                        "adapter": adapter_sha,
                    }
                    preds.append(pred)
                    if on_row is not None:
                        on_row(pred)
                    if on_progress is not None:
                        on_progress(idx, len(rows))
                return preds

            out_dir = td / "run_out"
            drive_root = td / "drive"
            drive_root.mkdir(parents=True, exist_ok=True)
            # Pre-seed 2 prefix rows for SFT to verify prefix resume
            manifest_rows = [json.loads(line) for line in test_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
            (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
            write_jsonl(
                out_dir / "predictions" / "sft.jsonl",
                [
                    {
                        "sample_id": row["sample_id"],
                        "text": row["text"],
                        "prediction_raw": "你好世介",
                        "generation_raw": "你好世介",
                        "token_ids": [1, 2],
                        "stop_reason": "eos",
                        "error": None,
                        "adapter": sft_rel["adapter_sha256"],
                    }
                    for row in manifest_rows[:2]
                ],
            )

            with patch("scripts.run_phase5_test_benchmark._match_training_runtime"):
                summary = run_phase5_benchmark(
                    config_path=self.config_path,
                    test_manifest=test_manifest,
                    manifest_sha_index=sha_index,
                    data_config_path=self.data_config_path,
                    sft_release_dir=td / "sft",
                    dpo_release_dir=td / "dpo",
                    rl_release_dir=td / "rl",
                    output_dir=out_dir,
                    run_id="20261023T120000_phase5_test_06b_zh_seed42",
                    seed=42,
                    drive_root=drive_root,
                    enforce_pinned_hashes=False,
                    decode_fn=fake_decode,
                    model_loader=fake_model_loader,
                    adapter_loader=fake_adapter_loader,
                )

            self.assertEqual(summary["schema_version"], PHASE5_BENCHMARK_SCHEMA)
            self.assertEqual(tuple(summary["stages"].keys()), STAGES)
            self.assertEqual(summary["stages"]["base"]["edit_count"], 5)
            self.assertEqual(summary["stages"]["sft"]["edit_count"], 5)
            self.assertEqual(summary["stages"]["dpo"]["edit_count"], 0)
            self.assertEqual(summary["stages"]["rl"]["edit_count"], 0)
            self.assertEqual(summary["comparisons"]["rl"]["total_edit_delta_vs_base"], -5)

            drive_run = drive_root / "runs" / "20261023T120000_phase5_test_06b_zh_seed42"
            for rel in ("benchmark_summary.json", "run_report.md", "status.json"):
                self.assertEqual(sha256_file(out_dir / rel), sha256_file(drive_run / rel))

    def test_notebook_and_val_rl_reference_contracts(self) -> None:
        nb_path = ROOT / "notebooks" / "01_qwen3_asr_post_training_pipeline.ipynb"
        self.assertTrue(nb_path.is_file())
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        self.assertEqual(nb.get("nbformat"), 4)
        code_cells = [cell for cell in nb.get("cells", []) if cell.get("cell_type") == "code"]
        self.assertGreaterEqual(len(code_cells), 5)
        full_code = "\n".join("".join(cell.get("source", [])) for cell in code_cells)
        self.assertIn("scripts/run_phase5_test_benchmark.py", full_code)
        self.assertIn("train/train_sft.py", full_code)
        self.assertIn("train/train_dpo.py", full_code)
        self.assertIn("train/train_rl.py", full_code)
        self.assertNotIn("import torch", full_code)
        self.assertNotIn("from qwen_asr", full_code)

        rl_val_ref = json.loads((ROOT / "configs" / "train" / "aishell_val_rl_step5_20261023.json").read_text(encoding="utf-8"))
        self.assertEqual(rl_val_ref["schema_version"], "qwen3-asr-parent-metrics-v1")
        self.assertEqual(rl_val_ref["release_id"], "rl_06b_zh_20261023T042034_step000005")
        self.assertEqual(rl_val_ref["source_step"], 5)
        self.assertAlmostEqual(rl_val_ref["micro_cer"], 2298 / 36620, places=12)


if __name__ == "__main__":
    unittest.main()
