from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.score_rl_greedy_held_out import (
    FROZEN_ROWS,
    FROZEN_SHA256,
    append_decision,
    build_greedy_row,
    checkpoint_complete,
    greedy_row_ok,
    resolve_score_step,
)


def _row(**overrides: object) -> dict:
    row = build_greedy_row(
        7,
        0.8793,
        0.1088,
        FROZEN_ROWS,
        FROZEN_SHA256,
        FROZEN_ROWS,
        FROZEN_ROWS,
        "2026-09-29T12:00:00+00:00",
    )
    row.update(overrides)
    return row


class GreedyHeldOutRowTest(unittest.TestCase):
    def test_built_row_matches_provenance(self) -> None:
        row = _row()
        self.assertTrue(greedy_row_ok(row))
        self.assertEqual(row["val_mean_reward"], 0.8793)
        self.assertIs(type(row["val_assigned_rows"]), int)
        self.assertEqual(row["val_decode"], "greedy")

    def test_float_counts_fail_provenance(self) -> None:
        row = _row(val_assigned_rows=1698.0)
        self.assertFalse(greedy_row_ok(row))

    def test_train_row_does_not_block_append(self) -> None:
        records = [{"global_step": 7, "raw_kl": 0.000568}]
        self.assertEqual(append_decision(records, 7), "append")

    def test_valid_row_skips_and_broken_row_refuses(self) -> None:
        self.assertEqual(append_decision([_row()], 7), "skip")
        broken = _row(val_rollouts=1697)
        self.assertEqual(append_decision([broken], 7), "refuse")

    def test_resolve_uses_saved_global_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            step_dir = run / "checkpoints" / "step_7" / "adapter"
            step_dir.mkdir(parents=True)
            (step_dir / "adapter_model.safetensors").write_bytes(b"x")
            (run / "checkpoints" / "step_7" / "optimizer.pt").write_bytes(b"x")
            (run / "checkpoints" / "step_4" / "adapter").mkdir(parents=True)
            state = {"global_step": 7, "status": "STOPPED_KL"}
            self.assertEqual(resolve_score_step(state, run), 7)
            self.assertTrue(checkpoint_complete(run, 7))
            self.assertFalse(checkpoint_complete(run, 4))

    def test_resolve_rejects_missing_checkpoint_and_bool_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                resolve_score_step({"global_step": 7}, run)
            with self.assertRaises(ValueError):
                resolve_score_step({"global_step": True}, run)

    def test_decision_cli_reads_loss_log(self) -> None:
        import io
        from contextlib import redirect_stdout

        from scripts.score_rl_greedy_held_out import main

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "loss_log.jsonl"
            path.write_text(json.dumps(_row()) + "\n", encoding="utf-8")
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(["--decision-only", "--step", "7", "--loss-log", str(path)])
            self.assertEqual(code, 0)
            self.assertEqual(stdout.getvalue().strip(), "skip")


if __name__ == "__main__":
    unittest.main()
