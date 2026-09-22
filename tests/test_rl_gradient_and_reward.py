#!/usr/bin/env python3
"""Unit tests verifying Schulman K3 KL gradient dependence on reference,
excess repetition penalty distinguishing natural repetitions, dynamic reward config,
and deterministic validation evaluation seeds.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from train.train_rl import (
    compute_sequence_reward,
    detect_excess_repetition,
    max_token_pair_count,
    max_token_run,
)


class TestKLGradientAndReferenceDependence(unittest.TestCase):
    """Verify that Schulman K3 KL loss gradient depends on reference log-probs,
    fixing the naive difference gradient degeneracy.
    """

    def test_naive_kl_gradient_is_degenerate_and_reference_invariant(self) -> None:
        """Prove that naive KL loss grad w.r.t. theta is invariant to reference logps."""
        # Policy parameter
        w1 = nn.Parameter(torch.tensor([1.0, 2.0], requires_grad=True))
        w2 = nn.Parameter(torch.tensor([1.0, 2.0], requires_grad=True))

        policy_logp1 = F.log_softmax(w1, dim=-1)[0]
        policy_logp2 = F.log_softmax(w2, dim=-1)[0]

        ref_logp_A = torch.tensor(-0.5)  # Reference model A
        ref_logp_B = torch.tensor(-2.5)  # Reference model B != A

        # Naive loss: beta * (policy_logp - ref_logp)
        beta = 0.04
        naive_loss_A = beta * (policy_logp1 - ref_logp_A)
        naive_loss_B = beta * (policy_logp2 - ref_logp_B)

        naive_loss_A.backward()
        naive_loss_B.backward()

        # Both gradients are identical because d/d_theta (ref_logp) = 0
        self.assertTrue(torch.allclose(w1.grad, w2.grad, atol=1e-7))

    def test_schulman_k3_gradient_depends_on_reference(self) -> None:
        """Verify that Schulman K3 KL estimator produces distinct gradients for distinct reference logps."""
        w1 = nn.Parameter(torch.tensor([1.0, 2.0], requires_grad=True))
        w2 = nn.Parameter(torch.tensor([1.0, 2.0], requires_grad=True))

        policy_logp1 = F.log_softmax(w1, dim=-1)[0]
        policy_logp2 = F.log_softmax(w2, dim=-1)[0]

        ref_logp_A = torch.tensor(-0.2)  # High reference prob
        ref_logp_B = torch.tensor(-2.5)  # Low reference prob

        beta = 0.04
        # K3: beta * (exp(ref - pol) - (ref - pol) - 1.0)
        diff1 = ref_logp_A - policy_logp1
        k3_loss_A = beta * (torch.exp(diff1) - diff1 - 1.0)

        diff2 = ref_logp_B - policy_logp2
        k3_loss_B = beta * (torch.exp(diff2) - diff2 - 1.0)

        k3_loss_A.backward()
        k3_loss_B.backward()

        # Gradients MUST differ because d(K3)/d(policy_logp) = beta * (1 - exp(ref - policy))
        self.assertFalse(torch.allclose(w1.grad, w2.grad, atol=1e-5))
        self.assertNotEqual(k3_loss_A.item(), k3_loss_B.item())

    def test_schulman_k3_zero_gradient_when_policy_equals_reference(self) -> None:
        """Verify that when policy equals reference, K3 loss is 0 and gradient w.r.t policy is 0."""
        w = nn.Parameter(torch.tensor([1.0, 2.0], requires_grad=True))
        policy_logp = F.log_softmax(w, dim=-1)[0]
        ref_logp = policy_logp.detach().clone()

        beta = 0.04
        diff = ref_logp - policy_logp
        k3_loss = beta * (torch.exp(diff) - diff - 1.0)

        self.assertAlmostEqual(k3_loss.item(), 0.0, places=6)
        k3_loss.backward()
        self.assertTrue(torch.allclose(w.grad, torch.zeros_like(w.grad), atol=1e-6))


class TestExcessRepetitionDetection(unittest.TestCase):
    """Verify that natural repetitions in reference are preserved while excess loops are penalized."""

    def test_natural_repetition_in_reference_not_penalized(self) -> None:
        """Exact match on natural speech containing repeats ('可能可能从全球意义上来讲')
        must receive reward 1.0 and repeat penalty 0.0, higher than deletion (reward ~0.8333).
        """
        gold = "可能可能从全球意义上来讲"
        exact_pred = "可能可能从全球意义上来讲"
        reward_exact, comps_exact, err_exact = compute_sequence_reward(exact_pred, gold, "zh")

        self.assertEqual(comps_exact["repeat"], 0.0)
        self.assertEqual(comps_exact["empty"], 0.0)
        self.assertEqual(comps_exact["too_long"], 0.0)
        self.assertAlmostEqual(err_exact, 0.0)
        self.assertAlmostEqual(reward_exact, 1.0)

        # Deletion candidate (removing the stutter: "很可能从全球意义上来讲")
        deletion_pred = "很可能从全球意义上来讲"
        reward_del, comps_del, err_del = compute_sequence_reward(deletion_pred, gold, "zh")

        # Exact match reward must strictly exceed deletion reward
        self.assertGreater(reward_exact, reward_del)
        self.assertAlmostEqual(reward_del, 1.0 - err_del, places=4)

    def test_excess_repetition_penalized(self) -> None:
        """Hypothesis that loops far beyond reference repetition must be penalized."""
        gold = "可能可能从全球意义上来讲"
        looping_pred = "可能可能可能可能可能从全球意义上来讲"

        reward, comps, err = compute_sequence_reward(looping_pred, gold, "zh")
        self.assertEqual(comps["repeat"], -0.25)
        self.assertLess(reward, 1.0)

    def test_english_single_word_and_phrase_repetition(self) -> None:
        """Test English excess repetition detection."""
        gold = "thank you thank you very much"
        exact_pred = "thank you thank you very much"
        rew_exact, comps_exact, _ = compute_sequence_reward(exact_pred, gold, "en")
        self.assertEqual(comps_exact["repeat"], 0.0)

        loop_pred = "thank you thank you thank you thank you very much"
        rew_loop, comps_loop, _ = compute_sequence_reward(loop_pred, gold, "en")
        self.assertEqual(comps_loop["repeat"], -0.25)


class TestDynamicRewardConfig(unittest.TestCase):
    """Verify that compute_sequence_reward dynamically uses values from reward_config."""

    def test_custom_penalties_applied(self) -> None:
        custom_config: Dict[str, Any] = {
            "components": {
                "empty": {"penalty": -0.6},
                "repeat": {"penalty": -0.45},
                "too_long": {"penalty": -0.35, "length_ratio_threshold": 1.2},
                "hallucination": {"penalty": -0.55, "error_rate_threshold": 0.5},
            },
            "clip": {"min": -2.0, "max": 1.5},
        }

        # Test empty penalty
        rew_empty, comps_empty, _ = compute_sequence_reward("", "hello world", "en", reward_config=custom_config)
        self.assertEqual(comps_empty["empty"], -0.6)
        self.assertEqual(rew_empty, -0.6)

        # Test custom repeat penalty
        rew_rep, comps_rep, _ = compute_sequence_reward(
            "hello hello hello hello", "hello", "en", reward_config=custom_config
        )
        self.assertEqual(comps_rep["repeat"], -0.45)

        # Test custom too_long threshold (ratio > 1.2)
        rew_long, comps_long, _ = compute_sequence_reward(
            "one two three four five six", "one two three four", "en", reward_config=custom_config
        )
        self.assertEqual(comps_long["too_long"], -0.35)


class TestRNGPreservation(unittest.TestCase):
    """Verify RNG state save and restore behavior used by validation."""

    def test_rng_state_save_and_restore(self) -> None:
        torch.manual_seed(12345)
        state_before = torch.get_rng_state()
        val_sample1 = torch.rand(5)

        # Reset seed back to before
        torch.set_rng_state(state_before)

        # Simulate eval_seed change with save/restore
        saved = torch.get_rng_state()
        try:
            torch.manual_seed(99999)
            _ = torch.rand(10)  # Validation sampling
        finally:
            torch.set_rng_state(saved)

        val_sample2 = torch.rand(5)
        self.assertTrue(torch.allclose(val_sample1, val_sample2))


if __name__ == "__main__":
    unittest.main()
