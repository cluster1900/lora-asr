#!/usr/bin/env python3
"""Select a deterministic smoke subset from an existing train/val manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.sft_contract import build_smoke_subset  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="抽出 SFT smoke 用的 train/val 子集")
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--val-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--train-sources", type=int, default=10)
    parser.add_argument("--val-sources", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        metadata = build_smoke_subset(
            args.train_manifest,
            args.val_manifest,
            args.output_dir,
            seed=args.seed,
            train_sources=args.train_sources,
            val_sources=args.val_sources,
            copy_audio=True,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"smoke subset failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "publishable": metadata["publishable"],
        "train_sources": len(metadata["train_sources"]),
        "val_sources": len(metadata["val_sources"]),
        "output_dir": str(args.output_dir),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
