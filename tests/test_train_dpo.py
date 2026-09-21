from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
    import torch.nn.functional as F
    HAVE_TORCH = True
except ImportError:
    torch = None
    F = None
    HAVE_TORCH = False

from train import train_dpo


class TrainDPOTest(unittest.TestCase):
    def test_dpo_preference_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_p = Path(tmpdir) / "dpo_test.jsonl"
            samples = [
                {
                    "sample_id": "pair_001",
                    "audio": "/dummy/audio1.wav",
                    "language": "en",
                    "chosen": "good transcript",
                    "rejected": "bad transcript",
                    "ref_chosen_logp": -5.0,
                    "ref_rejected_logp": -15.0,
                },
                {
                    "sample_id": "pair_002",
                    "audio": "/dummy/audio2.wav",
                    "language": "zh",
                    "chosen": "优质转写",
                    "rejected": "错误转写",
                    "ref_chosen_logp": -3.5,
                    "ref_rejected_logp": -12.0,
                },
            ]
            with open(manifest_p, "w", encoding="utf-8") as f:
                for s in samples:
                    f.write(json.dumps(s) + "\n")

            ds = train_dpo.DPOPreferenceDataset(manifest_p)
            self.assertEqual(len(ds), 2)
            self.assertEqual(ds[0]["sample_id"], "pair_001")
            self.assertEqual(ds[1]["sample_id"], "pair_002")

    @unittest.skipUnless(HAVE_TORCH, "PyTorch not available")
    def test_compute_logps(self) -> None:
        batch_size = 2
        seq_len = 5
        vocab_size = 10

        # Create deterministic logits
        logits = torch.zeros(batch_size, seq_len, vocab_size)
        # Sequence 0 predicts token 2 at each step with logit 10.0
        logits[0, :, 2] = 10.0
        # Sequence 1 predicts token 4 at each step with logit 10.0
        logits[1, :, 4] = 10.0

        labels = torch.tensor([
            [-100, 2, 2, 2, 2],  # 4 target tokens
            [-100, -100, 4, 4, 4],  # 3 target tokens
        ])

        logps = train_dpo.compute_logps(logits, labels)
        self.assertEqual(logps.shape, (batch_size,))
        # Since prediction matches token 2/4 with high logit, logp should be near 0 (sum of ~0)
        self.assertGreater(logps[0].item(), -0.1)
        self.assertGreater(logps[1].item(), -0.1)

    @unittest.skipUnless(HAVE_TORCH, "PyTorch not available")
    def test_dpo_loss_mathematical_properties(self) -> None:
        beta = 0.1

        # Case 1: Policy strongly prefers chosen over rejected
        pi_c_logp = torch.tensor([-5.0])
        pi_r_logp = torch.tensor([-20.0])
        ref_c_logp = torch.tensor([-10.0])
        ref_r_logp = torch.tensor([-10.0])

        pi_ratio = pi_c_logp - pi_r_logp  # +15.0
        ref_ratio = ref_c_logp - ref_r_logp  # 0.0
        logits = pi_ratio - ref_ratio  # +15.0

        loss_good = -F.logsigmoid(beta * logits).item()
        acc_good = (logits.item() > 0)
        self.assertTrue(acc_good)
        self.assertLess(loss_good, 0.3)

        # Case 2: Policy erroneously prefers rejected over chosen
        pi_c_logp_bad = torch.tensor([-20.0])
        pi_r_logp_bad = torch.tensor([-5.0])
        pi_ratio_bad = pi_c_logp_bad - pi_r_logp_bad  # -15.0
        logits_bad = pi_ratio_bad - ref_ratio  # -15.0

        loss_bad = -F.logsigmoid(beta * logits_bad).item()
        acc_bad = (logits_bad.item() > 0)
        self.assertFalse(acc_bad)
        self.assertGreater(loss_bad, 1.0)
        self.assertGreater(loss_bad, loss_good)

    def test_lora_target_regex(self) -> None:
        # Audio tower targets (3)
        self.assertTrue(train_dpo.LORA_TARGET_REGEX.match("audio_tower.conv_out"))
        self.assertTrue(train_dpo.LORA_TARGET_REGEX.match("audio_tower.proj1"))
        self.assertTrue(train_dpo.LORA_TARGET_REGEX.match("audio_tower.proj2"))

        # Decoder targets (28 layers * 7 = 196)
        for layer in range(28):
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                self.assertTrue(train_dpo.LORA_TARGET_REGEX.match(f"model.layers.{layer}.self_attn.{name}"))
            for name in ("gate_proj", "up_proj", "down_proj"):
                self.assertTrue(train_dpo.LORA_TARGET_REGEX.match(f"model.layers.{layer}.mlp.{name}"))

        # Non-targets should be rejected
        self.assertFalse(train_dpo.LORA_TARGET_REGEX.match("model.embed_tokens"))
        self.assertFalse(train_dpo.LORA_TARGET_REGEX.match("lm_head"))
        self.assertFalse(train_dpo.LORA_TARGET_REGEX.match("model.norm"))
        self.assertFalse(train_dpo.LORA_TARGET_REGEX.match("audio_tower.layers.0.conv1"))


if __name__ == "__main__":
    unittest.main()
