from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "12_fast_finetune_colab.ipynb"


class FastFineTuneNotebookTest(unittest.TestCase):
    def test_notebook_is_the_only_colab_entry_and_has_the_full_gate_sequence(self) -> None:
        notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
        self.assertEqual(notebooks, [NOTEBOOK])
        payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(payload["nbformat"], 4)
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in payload.get("cells", [])
        )
        required = (
            "'probe'",
            "'stage'",
            "'smoke'",
            "'--mode', 'smoke'",
            "'--mode', 'full'",
            "'--smoke-steps', '10'",
            "'--smoke-steps', '12'",
            "'--resume', 'auto'",
            "for limit in (60000, 100000, 160000, 200000)",
            "validation_canary_512",
            "validation_10k",
            "bench_5k",
            "release/adapter",
        )
        for marker in required:
            self.assertIn(marker, source)
        lowered = source.lower()
        for forbidden in ("api_key", "base_url", "gpt-5.5", "teacher"):
            self.assertNotIn(forbidden, lowered)


if __name__ == "__main__":
    unittest.main()
