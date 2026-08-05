# evaluation

正式 evaluator 是 `eval_wer.py`。它必须保存 raw/normalized reference 与 prediction，并按
语言使用不同主指标：English WER、Chinese CER。

```bash
python evaluation/eval_wer.py \
  --predictions-jsonl outputs/eval/lora.predictions.jsonl \
  --scored-jsonl outputs/eval/lora.scored.jsonl \
  --metrics-json outputs/eval/lora.metrics.json \
  --metrics-by-scenario-csv outputs/eval/lora.by_scenario.csv \
  --metrics-by-cell-csv outputs/eval/lora.by_cell.csv
```

验收输出包括 overall、language、scenario 和 32-cell（language x real/synthetic x 8 scenario）
聚合，以及空输出、重复输出、幻觉式输出和 inference error 比例。空 reference 必须硬失败；
失败 prediction 进入失败率，不能按零误差处理。

正式比较只使用同一 BF16 base、同一 evaluator 和同一 manifest。`analyze_errors.py` 用于生成
worst cases 和失败标签。禁止把 English word edit 与 Chinese character edit 混成一个未标注
的 overall WER，也不依赖 `references/mega-asr-upstream/`。
