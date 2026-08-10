from __future__ import annotations

import ast
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
        for explanation in ("运行前提", "产物", "通过标准", "恢复说明", "最终验收"):
            self.assertIn(explanation, source)
        lowered = source.lower()
        for forbidden in ("api_key", "base_url", "gpt-5.5", "teacher"):
            self.assertNotIn(forbidden, lowered)

    def test_production_python_definitions_have_docstrings(self) -> None:
        production_files = (
            ROOT / "scripts" / "prepare_public_robust_manifests.py",
            ROOT / "train" / "train_qwen3_asr_a2s.py",
            ROOT / "inference" / "qwen3_asr_infer.py",
            ROOT / "evaluation" / "eval_wer.py",
        )
        missing: list[str] = []
        for path in production_files:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    if ast.get_docstring(node) is None:
                        missing.append(f"{path.relative_to(ROOT)}:{node.lineno}:{node.name}")
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
