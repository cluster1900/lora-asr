# scripts

当前正式数据入口是待实现的 `prepare_public_robust_manifests.py`。它负责：

- 对 pinned `Voices-in-the-Wild-2M` revision 做 metadata-only probe。
- 按固定 seed 和配额生成 200k train、10k validation 与 Bench 5k test manifest。
- 检查音频、schema、时长、source 泄漏和 manifest hash。
- 调用统一 BF16 base 推理结果，生成 30k `<0.70` curriculum 及三个累计视图。

执行顺序：

```bash
python scripts/prepare_public_robust_manifests.py \
  probe \
  --config configs/data/public_robust_200k.yaml

python scripts/prepare_public_robust_manifests.py \
  smoke \
  --config configs/data/public_robust_200k.yaml \
  --robust-candidates /content/mega-asr-runtime/candidates/robust.jsonl \
  --english-clean-candidates /content/mega-asr-runtime/candidates/english_clean.jsonl \
  --chinese-clean-candidates /content/mega-asr-runtime/candidates/chinese_clean.jsonl \
  --bench-candidates /content/mega-asr-runtime/candidates/bench.jsonl

python scripts/prepare_public_robust_manifests.py \
  build \
  --config configs/data/public_robust_200k.yaml \
  --robust-candidates /content/mega-asr-runtime/candidates/robust.jsonl \
  --english-clean-candidates /content/mega-asr-runtime/candidates/english_clean.jsonl \
  --chinese-clean-candidates /content/mega-asr-runtime/candidates/chinese_clean.jsonl \
  --bench-candidates /content/mega-asr-runtime/candidates/bench.jsonl
```

真实音频和大缓存不提交到仓库。Google Drive 保存 shard 和结果，训练前将所需音频物化到
`/content` 本地 SSD。

其余 `create_*`、`build_difficulty_manifest.py` 和 `run_qwen3_asr_base_recheck.py` 均为
历史 smoke/MVP/v6A 工具，仅用于旧实验复现，不进入公开 200k A2S 依赖。项目代码不得调用
`references/mega-asr-upstream/`。
