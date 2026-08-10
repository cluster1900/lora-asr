from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCUMENTED_DIRECTORIES = ("evaluation", "inference", "notebooks", "scripts")


class DirectoryReadmeTest(unittest.TestCase):
    """Keep human-readable directory inventories synchronized with the tree."""

    def test_every_maintained_file_is_listed_in_its_directory_readme(self) -> None:
        missing: list[str] = []
        for directory_name in DOCUMENTED_DIRECTORIES:
            directory = ROOT / directory_name
            readme = directory / "README.md"
            self.assertTrue(readme.is_file(), f"missing {readme.relative_to(ROOT)}")
            content = readme.read_text(encoding="utf-8")
            for path in sorted(directory.iterdir()):
                if not path.is_file() or path.name == "README.md":
                    continue
                if path.name not in content:
                    missing.append(f"{directory_name}/{path.name}")
        self.assertEqual(missing, [], "files missing from directory README inventory")

    def test_readmes_state_the_maintenance_contract(self) -> None:
        for directory_name in DOCUMENTED_DIRECTORIES:
            content = (ROOT / directory_name / "README.md").read_text(encoding="utf-8")
            self.assertIn("目录职责", content)
            self.assertIn("文件清单", content)
            self.assertIn("维护要求", content)
            self.assertIn("必须同步更新", content)


if __name__ == "__main__":
    unittest.main()
