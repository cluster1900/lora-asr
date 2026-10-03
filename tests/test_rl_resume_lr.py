"""The resume learning-rate helper overwrites a restored scheduler rate."""

import unittest

from train.rl_resume_lr import apply_configured_learning_rate


class _Optimizer:
    def __init__(self) -> None:
        self.param_groups = [{"lr": 1.0e-5, "initial_lr": 1.0e-5}]


class _Scheduler:
    def __init__(self) -> None:
        self.base_lrs = [1.0e-5]
        self.last_epoch = 8


class ResumeLearningRateTest(unittest.TestCase):
    def test_overwrites_optimizer_and_scheduler_base(self) -> None:
        optimizer = _Optimizer()
        scheduler = _Scheduler()
        returned = apply_configured_learning_rate(optimizer, scheduler, 5.0e-6)
        self.assertAlmostEqual(returned, 5.0e-6)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 5.0e-6)
        self.assertAlmostEqual(optimizer.param_groups[0]["initial_lr"], 5.0e-6)
        self.assertEqual(scheduler.base_lrs, [5.0e-6])
        self.assertEqual(scheduler.last_epoch, 8)

    def test_updates_optimizer_when_scheduler_is_absent(self) -> None:
        optimizer = _Optimizer()
        apply_configured_learning_rate(optimizer, None, 5.0e-6)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 5.0e-6)

    def test_rejects_a_non_positive_rate(self) -> None:
        optimizer = _Optimizer()
        with self.assertRaises(ValueError):
            apply_configured_learning_rate(optimizer, None, 0.0)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1.0e-5)
