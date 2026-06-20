# Qwen3-ASR 模块探测摘要

## 元信息

- model_id: `Qwen/Qwen3-ASR-1.7B`
- created_at: `2026-06-20T05:38:28.427703+00:00`
- dtype: `float16`
- device_map: `cuda:0`
- total_modules: `703`
- elapsed_seconds: `8.38`

## Torch Module Root

- `model`: `Qwen3ASRForConditionalGeneration`

## 候选分组

- `attention_projection`: 208 个模块；leaf=['k_proj', 'o_proj', 'out_proj', 'q_proj', 'v_proj']；默认训练=True
- `mlp_projection`: 132 个模块；leaf=['down_proj', 'fc1', 'fc2', 'gate_proj', 'up_proj']；默认训练=True
- `review_only`: 1 个模块；leaf=['lm_head']；默认训练=False
- `speech_conv`: 3 个模块；leaf=['conv2d1', 'conv2d2', 'conv2d3']；默认训练=False
- `speech_projection`: 3 个模块；leaf=['conv_out', 'proj1', 'proj2']；默认训练=True

## 模块类 Top 20

- `Linear`: 344
- `Qwen3ASRTextRMSNorm`: 113
- `LayerNorm`: 49
- `Qwen3ASRThinkerTextDecoderLayer`: 28
- `Qwen3ASRTextAttention`: 28
- `Qwen3ASRTextMLP`: 28
- `SiLUActivation`: 28
- `GELUActivation`: 25
- `Qwen3ASRAudioEncoderLayer`: 24
- `Qwen3ASRAudioAttention`: 24
- `Conv2d`: 3
- `ModuleList`: 2
- `Qwen3ASRForConditionalGeneration`: 1
- `Qwen3ASRThinkerForConditionalGeneration`: 1
- `Qwen3ASRAudioEncoder`: 1
- `SinusoidsPositionEmbedding`: 1
- `Qwen3ASRThinkerTextModel`: 1
- `Embedding`: 1
- `Qwen3ASRThinkerTextRotaryEmbedding`: 1

## 分组计数

- `attention_projection`: 208
- `mlp_projection`: 132
- `not_candidate`: 356
- `review_only`: 1
- `speech_conv`: 3
- `speech_projection`: 3

## 下一步复核标准

1. 确认 `speech_projection` 是否真实位于音频/语音路径，而不是误匹配普通文本模块。
2. 第一版优先尝试 `attention_projection`，必要时再加入 `mlp_projection`。
3. `review_only` 和 `speech_conv` 默认不进入第一版 LoRA 训练。
4. 任何最终 target 都必须经过 5-20 step smoke training 验证。
