"""Unit and contract tests for Phase 6 Track 2 AliMeeting Eval OOD benchmark."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

from scripts.prepare_alimeeting_ood_eval import (
    OOD_MANIFEST_ROW_SCHEMA,
    OOD_MANIFEST_SET_SCHEMA,
    clean_alimeeting_transcript,
    load_alimeeting_ood_config,
    main as prepare_main,
    parse_praat_textgrid,
    prepare_alimeeting_ood,
)
from scripts.run_alimeeting_ood_benchmark import (
    OOD_BENCHMARK_SCHEMA,
    run_alimeeting_ood_benchmark,
    verify_ood_manifest_and_audio,
)
from test_phase5_benchmark import FakeModel, _make_dummy_release
from train.rl_schema import load_rl_config
from train.sft_checkpoint import atomic_write_json, sha256_file


def _write_long_textgrid(path: Path, tiers: list[tuple[str, float, float, list[tuple[float, float, str]]]], *, encoding: str = "utf-8") -> None:
    max_t = max((t[2] for t in tiers), default=10.0)
    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        "xmin = 0",
        f"xmax = {max_t}",
        "tiers? <exists>",
        f"size = {len(tiers)}",
        "item []:",
    ]
    for t_idx, (name, t_min, t_max, ivs) in enumerate(tiers, start=1):
        esc_name = name.replace('"', '""')
        lines.extend(
            [
                f"    item [{t_idx}]:",
                '        class = "IntervalTier"',
                f'        name = "{esc_name}"',
                f"        xmin = {t_min}",
                f"        xmax = {t_max}",
                f"        intervals: size = {len(ivs)}",
            ]
        )
        for i_idx, (imin, imax, mark) in enumerate(ivs, start=1):
            esc_mark = mark.replace('"', '""')
            lines.extend(
                [
                    f"        intervals [{i_idx}]:",
                    f"            xmin = {imin}",
                    f"            xmax = {imax}",
                    f'            text = "{esc_mark}"',
                ]
            )
    raw = "\n".join(lines).encode(encoding)
    if encoding == "utf-16-le":
        raw = b"\xff\xfe" + raw
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def _build_synthetic_eval_ali(root: Path) -> Path:
    eval_ali = root / "Eval_Ali"
    near_audio = eval_ali / "Eval_Ali_near" / "audio_dir"
    near_tg = eval_ali / "Eval_Ali_near" / "textgrid_dir"
    far_audio = eval_ali / "Eval_Ali_far" / "audio_dir"
    far_tg = eval_ali / "Eval_Ali_far" / "textgrid_dir"
    for d in (near_audio, near_tg, far_audio, far_tg):
        d.mkdir(parents=True, exist_ok=True)

    sr = 16000
    dur_s = 14.0
    n_samples = int(sr * dur_s)
    t = np.arange(n_samples, dtype=np.float32) / float(sr)

    for m_idx, (meeting_id, spk1, spk2) in enumerate(
        [
            ("R8001_M8004", "SPK8001", "SPK8002"),
            ("R8002_M8005", "SPK8003", "SPK8004"),
        ],
        start=1,
    ):
        w1 = (0.25 * np.sin(2.0 * np.pi * (220.0 + 30.0 * m_idx) * t)).astype(np.float32)
        w2 = (0.20 * np.sin(2.0 * np.pi * (330.0 + 30.0 * m_idx) * t)).astype(np.float32)
        sf.write(near_audio / f"{meeting_id}_F_{spk1}.wav", w1, sr, subtype="PCM_16", format="WAV")
        sf.write(near_audio / f"{meeting_id}_M_{spk2}.wav", w2, sr, subtype="PCM_16", format="WAV")

        # 8-channel far-field array: channel 0 carries attenuated speech, ch1..7 carry distinct tones
        far_multi = np.zeros((n_samples, 8), dtype=np.float32)
        far_multi[:, 0] = 0.15 * np.sin(2.0 * np.pi * (180.0 + 20.0 * m_idx) * t)
        for ch in range(1, 8):
            far_multi[:, ch] = 0.05 * float(ch)
        sf.write(far_audio / f"{meeting_id}_MS80{m_idx}.wav", far_multi, sr, subtype="PCM_16", format="WAV")

        # Timeline design:
        # [0.5, 2.5]: spk1 valid Mandarin with <%> tag; spk2 has "<sil>" (silence tag must NOT count as overlap!)
        # [3.0, 5.0]: spk1 & spk2 overlap ([3.0, 5.0] vs [4.2, 5.8]) -> both rejected as speaker_overlap
        # [6.2, 8.0]: spk2 valid Mandarin; spk1 empty ""
        # [8.2, 9.8]: spk1 Latin code-switch "讨论API接口" -> rejected contains_latin_code_switch
        # [10.0, 11.6]: spk1 explicit "(/OVERLAP)多人同时说话" -> rejected overlap_tag
        # [11.8, 13.4]: spk2 valid Mandarin #2
        spk1_ivs = [
            (0.0, 0.5, ""),
            (0.5, 2.5, f"<%> 今天第{m_idx}场会议讨论语音识别 <->"),
            (2.5, 3.0, "<sil>"),
            (3.0, 5.0, "这段话和别人重叠了"),
            (5.0, 8.2, ""),
            (8.2, 9.8, "今天讨论API接口"),
            (9.8, 10.0, "嗯"),
            (10.0, 11.6, "(/OVERLAP)多人同时说话"),
            (11.6, 14.0, ""),
        ]
        spk2_ivs = [
            (0.0, 2.5, "<sil>"),
            (2.5, 4.2, ""),
            (4.2, 5.8, "我也同时插话进来"),
            (5.8, 6.2, "(SIL)"),
            (6.2, 8.0, f"远场麦克风阵列第{m_idx}组测试正常"),
            (8.0, 11.8, ""),
            (11.8, 13.4, f"近场耳麦与远场通道零第{m_idx}次配对"),
            (13.4, 14.0, ""),
        ]
        _write_long_textgrid(
            near_tg / f"{meeting_id}_F_{spk1}.TextGrid",
            [(f"{meeting_id}_F_{spk1}", 0.0, dur_s, spk1_ivs)],
            encoding="utf-16-le" if m_idx == 2 else "utf-8",
        )
        _write_long_textgrid(
            near_tg / f"{meeting_id}_M_{spk2}.TextGrid",
            [(f"{meeting_id}_M_{spk2}", 0.0, dur_s, spk2_ivs)],
        )
        _write_long_textgrid(
            far_tg / f"{meeting_id}.TextGrid",
            [
                (spk1, 0.0, dur_s, spk1_ivs),
                (spk2, 0.0, dur_s, spk2_ivs),
            ],
        )
    return eval_ali


class AliMeetingOodBenchmarkTest(unittest.TestCase):
    def test_parse_praat_textgrid_long_and_short_formats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            long_p = td / "long.TextGrid"
            _write_long_textgrid(
                long_p,
                [("SPK8001", 0.0, 4.0, [(0.0, 1.5, ""), (1.5, 3.5, '你好"世界"')])],
                encoding="utf-16-le",
            )
            tiers_long = parse_praat_textgrid(long_p)
            self.assertEqual(len(tiers_long), 1)
            self.assertEqual(tiers_long[0].name, "SPK8001")
            self.assertEqual(tiers_long[0].intervals[1].text, '你好"世界"')

            short_p = td / "short.TextGrid"
            short_p.write_text(
                "\n".join(
                    [
                        'File type = "ooTextFile"',
                        'Object class = "TextGrid"',
                        "0",
                        "4.0",
                        '"<exists>"',
                        "1",
                        '"IntervalTier"',
                        '"SPK8001"',
                        "0",
                        "4.0",
                        "2",
                        "0",
                        "1.5",
                        '""',
                        "1.5",
                        "3.5",
                        '"纯短格式转写"',
                    ]
                ),
                encoding="utf-8",
            )
            tiers_short = parse_praat_textgrid(short_p)
            self.assertEqual(len(tiers_short), 1)
            self.assertEqual(tiers_short[0].intervals[1].text, "纯短格式转写")

        cleaned, reason = clean_alimeeting_transcript("<%> 你好 (~) 世界 <sil>")
        self.assertIsNone(reason)
        self.assertEqual(cleaned, "你好世界")

    def test_prepare_alimeeting_ood_filters_stratifies_and_caches(self) -> None:
        base_cfg = load_alimeeting_ood_config(ROOT / "configs" / "data" / "alimeeting_ood_eval.yaml")
        self.assertEqual(base_cfg["license"], "CC-BY-SA-4.0")
        self.assertEqual(base_cfg["usage_policy"], "read_only_ood_evaluation")

        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            eval_ali = _build_synthetic_eval_ali(td)

            # Default config requires min_paired_sources=50 -> 6 valid sources must return exit code 3
            rc_short = prepare_main(
                [
                    "--config",
                    str(ROOT / "configs" / "data" / "alimeeting_ood_eval.yaml"),
                    "--extracted-root",
                    str(eval_ali),
                    "--output-dir",
                    str(td / "out_short"),
                    "--seed",
                    "42",
                ]
            )
            self.assertEqual(rc_short, 3)

            # Fixture config with min_paired_sources=4, max_paired_sources=4 tests stratified round-robin (2 per meeting)
            custom_cfg = yaml.safe_load((ROOT / "configs" / "data" / "alimeeting_ood_eval.yaml").read_text(encoding="utf-8"))
            custom_cfg["selection"]["min_paired_sources"] = 4
            custom_cfg["selection"]["max_paired_sources"] = 4
            cfg_path = td / "ood_cfg.yaml"
            cfg_path.write_text(yaml.safe_dump(custom_cfg, sort_keys=False), encoding="utf-8")

            drive_cache = td / "drive_cache"
            code1, summary1 = prepare_alimeeting_ood(
                config_path=cfg_path,
                output_dir=td / "out1",
                seed=42,
                extracted_root=eval_ali,
                drive_cache_dir=drive_cache,
            )
            self.assertEqual(code1, 0)
            self.assertEqual(summary1["schema_version"], OOD_MANIFEST_SET_SCHEMA)
            self.assertFalse(summary1["trainable"])
            self.assertEqual(summary1["available_paired_candidates_pre_audio"], 6)
            self.assertEqual(summary1["selected_paired_sources"], 4)
            self.assertEqual(summary1["total_ood_rows"], 8)
            self.assertEqual(summary1["per_meeting_selected_sources"], {"R8001_M8004": 2, "R8002_M8005": 2})

            reasons = summary1["rejected_by_reason"]
            self.assertGreaterEqual(reasons.get("speaker_overlap", 0), 4)
            self.assertEqual(reasons.get("contains_latin_code_switch", 0), 2)
            self.assertEqual(reasons.get("overlap_tag", 0), 2)
            self.assertEqual(reasons.get("empty_or_too_short_text", 0), 2)

            # Second fresh build (without cache) produces bit-identical ood_eval.jsonl & manifest.json
            code2, _ = prepare_alimeeting_ood(
                config_path=cfg_path,
                output_dir=td / "out2",
                seed=42,
                extracted_root=eval_ali,
            )
            self.assertEqual(code2, 0)
            self.assertEqual(sha256_file(td / "out1" / "ood_eval.jsonl"), sha256_file(td / "out2" / "ood_eval.jsonl"))
            self.assertEqual(sha256_file(td / "out1" / "manifest.json"), sha256_file(td / "out2" / "manifest.json"))

            # Third run restores directly from drive_cache without needing extracted_root!
            code3, _ = prepare_alimeeting_ood(
                config_path=cfg_path,
                output_dir=td / "out_restored",
                seed=42,
                drive_cache_dir=drive_cache,
            )
            self.assertEqual(code3, 0)
            self.assertEqual(
                sha256_file(td / "out1" / "ood_eval.jsonl"),
                sha256_file(td / "out_restored" / "ood_eval.jsonl"),
            )
            rows = [json.loads(line) for line in (td / "out_restored" / "ood_eval.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["schema_version"], OOD_MANIFEST_ROW_SCHEMA)

    def test_run_alimeeting_ood_benchmark_cer_wer_bootstrap_resume_and_guards(self) -> None:
        train_cfg_path = ROOT / "configs" / "train" / "rl_pilot_aligned.yaml"
        infer_wer_path = ROOT / "configs" / "inference" / "base_eval_wer.yaml"
        infer_no_wer_path = ROOT / "configs" / "inference" / "base_eval.yaml"
        train_cfg = load_rl_config(train_cfg_path)

        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            eval_ali = _build_synthetic_eval_ali(td)
            custom_cfg = yaml.safe_load((ROOT / "configs" / "data" / "alimeeting_ood_eval.yaml").read_text(encoding="utf-8"))
            custom_cfg["selection"]["min_paired_sources"] = 4
            custom_cfg["selection"]["max_paired_sources"] = 6
            cfg_path = td / "ood_cfg.yaml"
            cfg_path.write_text(yaml.safe_dump(custom_cfg, sort_keys=False), encoding="utf-8")

            ood_dir = td / "ood_manifest"
            code, _ = prepare_alimeeting_ood(
                config_path=cfg_path,
                output_dir=ood_dir,
                seed=42,
                extracted_root=eval_ali,
            )
            self.assertEqual(code, 0)
            ood_jsonl = ood_dir / "ood_eval.jsonl"
            ood_sha = ood_dir / "sha256.json"

            # Guard 1: trainable=True must be rejected
            orig_meta_bytes = (ood_dir / "manifest.json").read_bytes()
            tampered_meta = json.loads(orig_meta_bytes.decode("utf-8"))
            tampered_meta["trainable"] = True
            atomic_write_json(ood_dir / "manifest.json", tampered_meta)
            with self.assertRaisesRegex(ValueError, "trainable=false"):
                verify_ood_manifest_and_audio(ood_jsonl, ood_sha)
            (ood_dir / "manifest.json").write_bytes(orig_meta_bytes)

            # Build dummy published SFT -> DPO -> RL releases
            sft_dir = td / "releases" / "sft"
            dpo_dir = td / "releases" / "dpo"
            rl_dir = td / "releases" / "rl"
            sft_rel = _make_dummy_release(
                sft_dir,
                stage="sft",
                release_id="sft_test_rel",
                status="published",
                publishable=True,
                config=train_cfg,
                parent_release=None,
                parent_sha=None,
                adapter_bytes=b"sft_ood_weights",
            )
            dpo_rel = _make_dummy_release(
                dpo_dir,
                stage="dpo",
                release_id="dpo_test_rel",
                status="published",
                publishable=True,
                config=train_cfg,
                parent_release=sft_rel["release_id"],
                parent_sha=sft_rel["adapter_sha256"],
                adapter_bytes=b"dpo_ood_weights",
            )
            rl_rel = _make_dummy_release(
                rl_dir,
                stage="rl",
                release_id="rl_test_rel",
                status="published",
                publishable=True,
                config=train_cfg,
                parent_release=dpo_rel["release_id"],
                parent_sha=dpo_rel["adapter_sha256"],
                adapter_bytes=b"rl_ood_weights",
            )

            # Guard 2: infer_config without locked jieba WER must be rejected
            with self.assertRaisesRegex(ValueError, "locked jieba WER"):
                run_alimeeting_ood_benchmark(
                    train_config_path=train_cfg_path,
                    infer_config_path=infer_no_wer_path,
                    ood_manifest=ood_jsonl,
                    manifest_sha_index=ood_sha,
                    sft_release_dir=sft_dir,
                    dpo_release_dir=dpo_dir,
                    rl_release_dir=rl_dir,
                    output_dir=td / "bad_out",
                    run_id="bad_no_wer",
                    seed=42,
                    enforce_pinned_hashes=False,
                )

            fake_model = FakeModel()
            stage_by_sha = {
                "base_zero_lora": "base",
                sft_rel["adapter_sha256"]: "sft",
                dpo_rel["adapter_sha256"]: "dpo",
                rl_rel["adapter_sha256"]: "rl",
            }

            def fake_loader(_cfg: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
                return fake_model, object(), {}

            def fake_adapter_loader(m: FakeModel, path: Path) -> None:
                val = float(len(path.read_bytes()))
                for _name, p in m.named_parameters():
                    p.value = val

            def fake_decode(
                _model: Any,
                _processor: Any,
                subset_rows: list[dict[str, Any]],
                _manifest: Path,
                cfg: dict[str, Any],
                _step: int,
                adapter_sha: str,
                on_progress: Any = None,
                on_row: Any = None,
            ) -> list[dict[str, Any]]:
                stage_name = stage_by_sha[adapter_sha]
                out: list[dict[str, Any]] = []
                for idx, r in enumerate(subset_rows, start=1):
                    gold = str(r["text"])
                    if r["scenario"] == "near_field":
                        pred_text = gold
                    elif stage_name == "base":
                        pred_text = "错字" + gold[2:]
                    elif stage_name == "sft":
                        pred_text = "错" + gold[1:]
                    else:
                        pred_text = gold
                    rec = {
                        "schema_version": "qwen3-asr-predictions-v1",
                        "sample_id": r["sample_id"],
                        "audio": r["audio"],
                        "scenario": r["scenario"],
                        "split": r["split"],
                        "model_name": cfg["model"]["name"],
                        "model_revision": cfg["model"]["revision"],
                        "adapter_sha256": adapter_sha,
                        "decode_eval_sha256": cfg["decode_eval_sha256"],
                        "prediction_raw": pred_text,
                        "prediction_normalized": pred_text,
                        "stop_reason": "eos",
                        "error": None,
                    }
                    out.append(rec)
                    if on_row is not None:
                        on_row(rec)
                    if on_progress is not None:
                        on_progress(idx, len(subset_rows))
                return out

            # Pre-seed 2 prefix rows in predictions/dpo.jsonl to test resume
            rows = [json.loads(line) for line in ood_jsonl.read_text(encoding="utf-8").splitlines()]
            out_dir = td / "ood_run_out"
            drive_root = td / "drive_root"
            drive_root.mkdir(parents=True, exist_ok=True)
            (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
            with (out_dir / "predictions" / "dpo.jsonl").open("w", encoding="utf-8") as h:
                for r in rows[:2]:
                    h.write(
                        json.dumps(
                            {
                                "schema_version": "qwen3-asr-predictions-v1",
                                "sample_id": r["sample_id"],
                                "audio": r["audio"],
                                "scenario": r["scenario"],
                                "split": r["split"],
                                "model_name": train_cfg["model"]["name"],
                                "model_revision": train_cfg["model"]["revision"],
                                "adapter_sha256": dpo_rel["adapter_sha256"],
                                "decode_eval_sha256": train_cfg["decode_eval_sha256"],
                                "prediction_raw": r["text"],
                                "prediction_normalized": r["text"],
                                "stop_reason": "eos",
                                "error": None,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            with patch("scripts.run_alimeeting_ood_benchmark._match_training_runtime", return_value=None):
                summary = run_alimeeting_ood_benchmark(
                    train_config_path=train_cfg_path,
                    infer_config_path=infer_wer_path,
                    ood_manifest=ood_jsonl,
                    manifest_sha_index=ood_sha,
                    sft_release_dir=sft_dir,
                    dpo_release_dir=dpo_dir,
                    rl_release_dir=rl_dir,
                    output_dir=out_dir,
                    run_id="20261030T130000_phase6_alimeeting_ood_06b_zh_seed42",
                    seed=42,
                    drive_root=drive_root,
                    enforce_pinned_hashes=False,
                    bootstrap_resamples=200,
                    decode_fn=fake_decode,
                    model_loader=fake_loader,
                    adapter_loader=fake_adapter_loader,
                )

            self.assertEqual(summary["schema_version"], OOD_BENCHMARK_SCHEMA)
            self.assertEqual(summary["stages"]["base"]["edit_count"], 12)
            self.assertEqual(summary["stages"]["sft"]["edit_count"], 6)
            self.assertEqual(summary["stages"]["dpo"]["edit_count"], 0)
            self.assertEqual(summary["stages"]["rl"]["edit_count"], 0)
            self.assertGreater(summary["stages"]["base"]["micro_wer"], 0.0)
            self.assertEqual(summary["stages"]["dpo"]["micro_wer"], 0.0)
            self.assertGreater(summary["stages"]["base"]["far_minus_near_micro_cer"], 0.0)
            self.assertLess(
                summary["paired_cluster_bootstrap_95ci"]["dpo_vs_base"]["far_field"]["micro_cer_ci95"][1],
                0.0,
            )

            # Check WER columns exist in by_scenario.csv and Drive sync readback matches
            with (out_dir / "eval" / "base" / "by_scenario.csv").open("r", encoding="utf-8") as f:
                header = next(csv.reader(f))
            self.assertIn("micro_wer", header)
            drive_summary = drive_root / "runs" / "20261030T130000_phase6_alimeeting_ood_06b_zh_seed42" / "benchmark_summary.json"
            self.assertTrue(drive_summary.is_file())
            self.assertEqual(sha256_file(drive_summary), sha256_file(out_dir / "benchmark_summary.json"))


if __name__ == "__main__":
    unittest.main()
