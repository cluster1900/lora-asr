"""Every behaviour-bearing key in the v30A RL config must have a consumer.

Background (docs/qwen3-asr/28_rl_v30_trainer_sync_and_stats.md): the
repository trainer once lacked most of the keys the v29/v30 configs set, so a
config could silently run a different experiment than it described.

Two layers:

* a static check that each leaf key under ``train``, ``grpo`` and ``reward``
  appears as a quoted literal in code that could read it (name based, so it
  cannot prove a generic name such as ``penalty`` is read from *this* section;
  the reward section is therefore pinned to ``config_path`` only);
* ``check_config_contract``: the stop/probe thresholds are hard-coded in
  ``STOP_CONTRACT``; a YAML copy that disagrees must be rejected, not ignored.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path
from typing import Iterator

import yaml

from train.train_rl import (
    STOP_CONTRACT,
    STOP_PROFILES,
    TRAIN_SEQUENCES,
    check_config_contract,
    decide_rl_stop,
    driver_action_from_records,
    resolve_stop_profile,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "train" / "qwen3_asr_rl_v30a.yaml"
SECTIONS = ("train", "grpo", "reward")
CONSUMER_GLOBS = ("train/*.py", "scripts/*", "evaluation/*.py")

# Declared in YAML but read by nothing. Must stay empty for v30A: unread keys
# were removed, and stop thresholds are now enforced by check_config_contract.
DECLARED_ONLY: frozenset[str] = frozenset()


def _leaves(node: dict, prefix: str) -> Iterator[str]:
    for key, value in node.items():
        name = f"{prefix}.{key}"
        if isinstance(value, dict):
            yield from _leaves(value, name)
        else:
            yield name


def _consumer_text() -> str:
    parts = []
    for pattern in CONSUMER_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            if path.is_file():
                parts.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(parts)


def _is_consumed(key: str, text: str) -> bool:
    return re.search(r"""["']%s["']""" % re.escape(key), text) is not None


class RlConfigContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        cls.text = _consumer_text()

    def test_unconsumed_keys_match_the_pinned_set(self) -> None:
        unconsumed = set()
        for section in SECTIONS:
            for leaf in _leaves(self.config[section], section):
                if not _is_consumed(leaf.split(".")[-1], self.text):
                    unconsumed.add(leaf)
        self.assertEqual(
            unconsumed,
            set(DECLARED_ONLY),
            "config keys without a consumer changed; wire the key or remove it, "
            "and update docs/qwen3-asr/28_rl_v30_trainer_sync_and_stats.md",
        )

    def test_reward_section_only_points_at_reward_config(self) -> None:
        # train_rl.py loads penalties/clip from reward.config_path when the
        # inline section has no ``components``; inline copies would be ignored.
        self.assertEqual(set(self.config["reward"]), {"config_path"})
        self.assertTrue((ROOT / self.config["reward"]["config_path"]).is_file())

    def test_v30a_behaviour_keys_are_consumed(self) -> None:
        for key in (
            "sample_strategy",
            "sample_strategy_options",
            "policy_token_mask",
            "include_reference_candidate",
            "local_max_relative",
            "min_improvement",
            "apply_learning_rate_on_resume",
            "train_audio_projections",
            "second_round",
        ):
            self.assertTrue(_is_consumed(key, self.text), f"{key} has no consumer")

    def test_trainer_wires_degraded_balanced(self) -> None:
        trainer = (ROOT / "train" / "train_rl.py").read_text(encoding="utf-8")
        self.assertIn('"degraded_balanced"', trainer)
        self.assertIn("degraded_balanced_summary", trainer)
        self.assertIn("strategy_options=sample_strategy_options", trainer)
        self.assertIn('train_cfg.get("sample_strategy_options")', trainer)
        self.assertIn("check_config_contract(train_cfg, grpo_cfg)", trainer)

    def test_v30a_weights_are_equal_and_cover_seven_scenarios(self) -> None:
        options = self.config["train"]["sample_strategy_options"]
        weights = options["scenario_weights"]
        self.assertEqual(len(weights), 7)
        self.assertEqual(len(set(weights.values())), 1)
        self.assertEqual(self.config["train"]["sample_strategy"], "degraded_balanced")


class StopContractTest(unittest.TestCase):
    def test_every_rl_config_matches_the_code(self) -> None:
        configs = sorted((ROOT / "configs" / "train").glob("qwen3_asr_rl*.yaml"))
        self.assertTrue(configs)
        for path in configs:
            cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            with self.subTest(config=path.name):
                check_config_contract(cfg.get("train") or {}, cfg.get("grpo") or {})

    def test_changed_threshold_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_raw_kl"):
            check_config_contract({"max_raw_kl": 1.0e-3}, {})
        with self.assertRaisesRegex(ValueError, "train_sequences"):
            check_config_contract({}, {"train_sequences": TRAIN_SEQUENCES + 1})

    def test_missing_copies_are_allowed(self) -> None:
        check_config_contract({}, {})

    def test_kl_stop_uses_the_contract_value(self) -> None:
        limit = STOP_CONTRACT["max_raw_kl"]
        self.assertEqual(decide_rl_stop(4, 20.0, raw_kl=limit * 1.01), "STOPPED_KL")
        self.assertNotEqual(decide_rl_stop(4, 20.0, raw_kl=limit), "STOPPED_KL")

    def test_reward_drop_uses_the_contract_value(self) -> None:
        drop = STOP_CONTRACT["early_held_out_reward_drop"]
        self.assertEqual(
            decide_rl_stop(4, 20.0, greedy_gain=-drop - 1e-4), "STOPPED_REWARD_DROP"
        )


V31 = ROOT / "configs" / "train" / "qwen3_asr_rl_v31_scale.yaml"


class ScaleStopProfileTest(unittest.TestCase):
    """v31 ``scale`` profile (docs/qwen3-asr/29_rl_v31_scale_design.md)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = yaml.safe_load(V31.read_text(encoding="utf-8"))
        cls.scale = STOP_PROFILES["scale"]

    def test_pilot_alias_is_unchanged(self) -> None:
        self.assertIs(STOP_CONTRACT, STOP_PROFILES["pilot"])
        self.assertEqual(STOP_CONTRACT["max_raw_kl"], 5.0e-4)
        self.assertEqual(STOP_CONTRACT["step12_mass_floor"], 17.0)
        self.assertEqual(self.scale["futility_step"], 640)
        self.assertEqual(self.scale["futility_patience"], 2)

    def test_probe_threshold_is_profile_independent(self) -> None:
        # The probe helpers read STOP_CONTRACT; this is only correct while
        # every profile uses the same probe_mass_min.
        values = {table["probe_mass_min"] for table in STOP_PROFILES.values()}
        self.assertEqual(values, {17.0})

    def test_v31_config_selects_scale_and_passes(self) -> None:
        train = self.config["train"]
        self.assertEqual(train["stop_profile"], "scale")
        self.assertEqual(train["sample_strategy"], "degraded")
        self.assertEqual((train["max_steps"], train["save_steps"], train["eval_steps"]), (2560, 320, 320))
        check_config_contract(train, self.config["grpo"])
        self.assertEqual(set(self.config["reward"]), {"config_path"})

    def test_v31_keys_are_consumed(self) -> None:
        text = _consumer_text()
        unconsumed = {
            leaf
            for section in SECTIONS
            for leaf in _leaves(self.config[section], section)
            if not _is_consumed(leaf.split(".")[-1], text)
        }
        self.assertEqual(unconsumed, set())

    def test_v31_keeps_non_scale_contract_and_applies_stability_patch(self) -> None:
        v30a = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        for section in ("model", "runtime", "grpo", "reward", "lora", "lifecycle", "gates"):
            if section == "grpo":
                # Only the KL tether is intentionally strengthened in v31.
                actual = dict(self.config[section])
                expected = dict(v30a[section])
                actual_kl = dict(actual["kl_regularization"])
                expected_kl = dict(expected["kl_regularization"])
                actual["kl_regularization"] = dict(actual_kl)
                expected["kl_regularization"] = dict(expected_kl)
                actual["kl_regularization"].pop("beta", None)
                expected["kl_regularization"].pop("beta", None)
                self.assertEqual(actual, expected, section)
            else:
                self.assertEqual(self.config[section], v30a[section], section)
        self.assertEqual(self.config["train"]["learning_rate"], 2.0e-6)
        self.assertEqual(self.config["train"]["warmup_steps"], 64)
        self.assertEqual(self.config["grpo"]["kl_regularization"]["beta"], 0.08)

    def test_v31_launcher_fails_closed_and_excludes_all_roles(self) -> None:
        launcher = (ROOT / "scripts" / "run_rl_scale_v31.sh").read_text(encoding="utf-8")
        self.assertIn("--query-compute-apps=pid,process_name", launcher)
        self.assertIn("nvidia-smi is unavailable", launcher)
        self.assertIn("validate_degraded_manifest", launcher)
        self.assertIn("POOL_MIN_ROWS=160000", launcher)
        self.assertIn("CELL_MIN_ROWS=640", launcher)
        self.assertIn("MAX_CELL_FRACTION=0.20", launcher)
        self.assertIn("require_all_cells=require_all_cells", launcher)
        self.assertIn("ISOLATION_CHECK", launcher)
        self.assertIn("MIN_FREE_GB=100", launcher)
        self.assertIn("RL_RESUME=1", launcher)
        self.assertIn("RL_RUN_DIR must be under /data/mega-asr/runs/rl_scale_v31*", launcher)
        self.assertIn("TRAIN_CHUNK_DONE", launcher)
        self.assertIn("--resume-from-checkpoint", launcher)
        self.assertIn("--max-steps \"$chunk_end\"", launcher)
        self.assertIn("select_resume_step", launcher)
        self.assertIn("materialize_terminal_pipeline_state", launcher)
        self.assertIn("pipeline.with_suffix(\".json.tmp\")", launcher)
        self.assertIn('materialize_terminal_pipeline_state "$CURRENT_STEP" "COMPLETED"', launcher)
        self.assertIn("scheduler.pt", launcher)
        self.assertIn("CURRENT_TERMINAL", launcher)
        self.assertIn('cd "$REPO" || exit 1', launcher)
        for role in ("sft_train", "dpo_train_pool", "rl_train_pool", "rl_val_pool", "validation", "bench_test"):
            self.assertIn(f"{role}.jsonl", launcher)

    def test_unknown_profile_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "stop_profile"):
            check_config_contract({"stop_profile": "turbo"}, {})
        with self.assertRaisesRegex(ValueError, "stop_profile"):
            decide_rl_stop(4, 20.0, profile="turbo")
        self.assertEqual(resolve_stop_profile(None), "pilot")
        self.assertEqual(resolve_stop_profile("  "), "pilot")
        self.assertEqual(resolve_stop_profile("Scale"), "scale")

    def test_cross_profile_keys_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "step12_mass_floor is not used by stop_profile=scale"):
            check_config_contract({"stop_profile": "scale", "step12_mass_floor": 17.0}, {})
        with self.assertRaisesRegex(ValueError, "futility_step is not used by stop_profile=pilot"):
            check_config_contract({"futility_step": 32}, {})

    def test_changed_scale_threshold_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_raw_kl"):
            check_config_contract({"stop_profile": "scale", "max_raw_kl": 5.0e-4}, {})
        with self.assertRaisesRegex(ValueError, "futility_step"):
            check_config_contract({"stop_profile": "scale", "futility_step": 16}, {})

    def test_scale_kl_ceiling(self) -> None:
        limit = self.scale["max_raw_kl"]
        self.assertEqual(limit, 2.0e-2)
        # A raw KL that stops a pilot does not stop a scale run.
        self.assertEqual(decide_rl_stop(7, 20.0, raw_kl=6.2e-4), "STOPPED_KL")
        self.assertEqual(decide_rl_stop(7, 20.0, raw_kl=6.2e-4, profile="scale"), "")
        self.assertEqual(decide_rl_stop(7, 20.0, raw_kl=limit * 1.01, profile="scale"), "STOPPED_KL")

    def test_scale_has_no_pilot_search_or_transfer_stops(self) -> None:
        kwargs = dict(probe_update_rate=0.3, probe_median_gap=0.075)
        # Pilot: step 12, low gain, enough mass -> BLOCKED_TRANSFER; low mass -> BLOCKED_SEARCH.
        self.assertEqual(decide_rl_stop(12, 20.0, greedy_gain=0.0004, **kwargs), "BLOCKED_TRANSFER")
        self.assertEqual(decide_rl_stop(12, 5.0, greedy_gain=0.0004, **kwargs), "BLOCKED_SEARCH")
        self.assertEqual(decide_rl_stop(4, 0.5, greedy_gain=0.0, **kwargs), "BLOCKED_SEARCH")
        for step, mass in ((4, 0.5), (8, 5.0), (12, 5.0), (12, 20.0), (16, 20.0)):
            with self.subTest(step=step, mass=mass):
                self.assertEqual(
                    decide_rl_stop(step, mass, greedy_gain=0.0, profile="scale", **kwargs), ""
                )

    def test_scale_futility_stop(self) -> None:
        self.assertEqual(decide_rl_stop(16, 50.0, greedy_gain=-0.001, profile="scale"), "")
        self.assertEqual(decide_rl_stop(640, 50.0, greedy_gain=-0.0001, profile="scale"), "")
        self.assertEqual(
            decide_rl_stop(640, 50.0, greedy_gain=-0.0001, consecutive_futile_evals=1, profile="scale"),
            "",
        )
        self.assertEqual(
            decide_rl_stop(640, 50.0, greedy_gain=-0.0001, consecutive_futile_evals=2, profile="scale"),
            "BLOCKED_TRANSFER",
        )
        self.assertEqual(decide_rl_stop(960, 50.0, greedy_gain=0.0012, profile="scale"), "")
        # No fresh held-out measurement -> futility cannot fire.
        self.assertEqual(decide_rl_stop(40, 50.0, greedy_gain=None, profile="scale"), "")

    def test_scale_keeps_safety_stops(self) -> None:
        self.assertEqual(
            decide_rl_stop(16, 50.0, greedy_gain=-0.0051, profile="scale"), "STOPPED_REWARD_DROP"
        )
        self.assertEqual(decide_rl_stop(4, 0.0, cumulative_winners=0, profile="scale"), "BLOCKED_NO_WINNERS")
        self.assertEqual(
            decide_rl_stop(
                5, 10.0, identical_ratio=0.9, mean_reward=0.5, consecutive_collapsed=2,
                collapse_mean_reward_floor=0.75, profile="scale",
            ),
            "FAILED_ZERO_VARIANCE",
        )
        self.assertEqual(decide_rl_stop(16, 50.0, robust_increase=0.0006, profile="scale"), "STOPPED_ROBUST")

    def test_driver_threads_the_profile(self) -> None:
        records = [
            {"global_step": 0, "val_mean_reward": 0.8773, "val_eval_scope": "Full Held-out", "val_decode": "greedy"},
            {"global_step": 12, "reward_mass_in_step": 20.0, "winners_in_step": 100, "raw_kl": 6.2e-4,
             "zero_variance_ratio": 0.4, "mean_reward": 0.8},
            {"global_step": 12, "val_mean_reward": 0.8775, "val_eval_scope": "Full Held-out", "val_decode": "greedy"},
        ]
        probe = {"update_rate_k11": 0.3, "median_winning_gap_k11": 0.075}
        common = dict(global_step=12, horizon=64, probe_payload=probe, gate_status=None, robust_increase=None)
        pilot = driver_action_from_records(records, **common)
        scale = driver_action_from_records(records, stop_profile="scale", **common)
        self.assertEqual(pilot["stop"], "STOPPED_KL")
        self.assertEqual(scale["stop"], "")
        self.assertTrue(scale["resume"])

    def test_driver_recomputes_scale_futility_from_the_log(self) -> None:
        probe = {"update_rate_k11": 0.3, "median_winning_gap_k11": 0.075}

        def train_row(step: int) -> dict:
            return {
                "global_step": step,
                "reward_mass_in_step": 1.0,
                "winners_in_step": 10,
                "raw_kl": 1.0e-4,
                "zero_variance_ratio": 0.3,
                "mean_reward": 0.8,
            }

        def held_out(step: int, reward: float) -> dict:
            return {
                "global_step": step,
                "val_mean_reward": reward,
                "val_eval_scope": "Full Held-out",
                "val_decode": "greedy",
            }

        first = [held_out(0, 0.8773), train_row(640), held_out(640, 0.8771)]
        continued = driver_action_from_records(
            first,
            global_step=640,
            horizon=2560,
            probe_payload=probe,
            gate_status=None,
            robust_increase=None,
            stop_profile="scale",
        )
        self.assertEqual(continued["stop"], "")
        self.assertEqual(continued["status"], "CHUNK_DONE")
        self.assertTrue(continued["resume"])

        second = first + [train_row(960), held_out(960, 0.8770)]
        stopped = driver_action_from_records(
            second,
            global_step=960,
            horizon=2560,
            probe_payload=probe,
            gate_status=None,
            robust_increase=None,
            stop_profile="scale",
        )
        self.assertEqual(stopped["stop"], "BLOCKED_TRANSFER")
        self.assertFalse(stopped["resume"])

    def test_trainer_passes_profile_in_loop(self) -> None:
        trainer = (ROOT / "train" / "train_rl.py").read_text(encoding="utf-8")
        self.assertIn('stop_profile = resolve_stop_profile(train_cfg.get("stop_profile"))', trainer)
        self.assertIn("profile=stop_profile,", trainer)


if __name__ == "__main__":
    unittest.main()
