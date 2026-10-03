"""v19 caps the raw reward gap and reads the latest prior gate."""

import json
import tempfile
import unittest
from pathlib import Path

from train.rl_v17_decision import followup_action
from train.rl_v19_decision import latest_prior_robust

CONFIG = Path("configs/train/qwen3_asr_rl_v19.yaml")


def _gate(robust: float) -> str:
    return json.dumps({"metrics": {"robust_error_rate_increase": robust}})


class V19ContractTest(unittest.TestCase):
    def test_shipped_config_caps_the_gap(self) -> None:
        text = CONFIG.read_text(encoding="utf-8")
        self.assertIn("learning_rate: 1.0e-5", text)
        self.assertIn("mode: capped_gap", text)
        self.assertIn("cap: 0.05", text)
        self.assertNotIn("mode: unit", text)
        self.assertNotIn("mode: raw_gap", text)
        self.assertIn("local_max_relative: 0.35", text)
        self.assertIn("beta: 0.04", text)
        self.assertIn("max_raw_kl: 5.0e-4", text)
        self.assertIn("max_steps: 48", text)

    def test_step10_uses_the_copied_step8_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "gate_step_8.json").write_text(_gate(0.000164), encoding="utf-8")
            self.assertAlmostEqual(latest_prior_robust(run, 10), 0.000164)

    def test_later_step_uses_the_newest_earlier_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "gate_step_8.json").write_text(_gate(0.000164), encoding="utf-8")
            (run / "gate_step_10.json").write_text(_gate(0.000014), encoding="utf-8")
            self.assertAlmostEqual(latest_prior_robust(run, 12), 0.000014)

    def test_kl_stop_is_terminal(self) -> None:
        action, reason = followup_action(
            "STOPPED_KL",
            10,
            48,
            0.0001,
            0.000014,
            "FAILED",
            0.000164,
            16.354781,
        )
        self.assertEqual(action, "stop")
        self.assertIn("STOPPED_KL", reason)


if __name__ == "__main__":
    unittest.main()
