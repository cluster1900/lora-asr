"""Qwen3-ASR LoRA target 候选规则。

本文件只做“候选模块分类”，不直接决定最终训练 target。最终 LoRA target
必须以 `train/inspect_qwen3_asr_modules.py` 在当前 Qwen3-ASR 版本上导出的
真实模块快照为准，并经过 smoke training 验证后再写入训练配置。
"""

from __future__ import annotations


ATTENTION_SUFFIXES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "query",
    "key",
    "value",
    "out_proj",
    "c_attn",
    "c_proj",
}

MLP_SUFFIXES = {
    "gate_proj",
    "up_proj",
    "down_proj",
    "fc1",
    "fc2",
    "w1",
    "w2",
    "w3",
}

SPEECH_PATH_TOKENS = {
    "audio",
    "speech",
    "encoder",
    "frontend",
    "projector",
    "adapter",
    "conformer",
    "whisper",
}

REVIEW_ONLY_SUFFIXES = {
    "lm_head",
}


def module_leaf_name(module_name: str) -> str:
    """返回模块路径最后一段名称。"""
    return module_name.rsplit(".", 1)[-1].lower()


def is_linear_like(class_name: str) -> bool:
    """判断模块类名是否类似线性层。

    量化库可能把线性层命名为 Linear4bit、QuantLinear 等，因此这里不只匹配
    精确的 `Linear`。
    """
    return "linear" in class_name.lower()


def is_conv_like(class_name: str) -> bool:
    """判断模块类名是否类似卷积层。"""
    return "conv" in class_name.lower()


def classify_lora_candidate(module_name: str, class_name: str) -> str:
    """给单个模块打上 LoRA 候选分组标签。

    返回空字符串表示暂不推荐作为 LoRA target。分组含义：

    - `attention_projection`：语言模型注意力投影层，通常是第一批 LoRA 候选。
    - `mlp_projection`：语言模型 MLP 投影层，可作为第二批或联合候选。
    - `speech_projection`：音频/语音路径中的线性投影层，需要重点人工复核。
    - `speech_conv`：音频/语音路径中的卷积层，通常只做观察，不直接挂 LoRA。
    - `review_only`：必须人工复核，默认不进入训练。
    """
    leaf = module_leaf_name(module_name)
    lower_name = module_name.lower()

    if leaf in REVIEW_ONLY_SUFFIXES:
        return "review_only"

    if is_linear_like(class_name):
        if leaf in ATTENTION_SUFFIXES:
            return "attention_projection"
        if leaf in MLP_SUFFIXES:
            return "mlp_projection"
        if any(token in lower_name for token in SPEECH_PATH_TOKENS):
            return "speech_projection"

    if is_conv_like(class_name) and any(token in lower_name for token in SPEECH_PATH_TOKENS):
        return "speech_conv"

    return ""


def candidate_is_trainable_by_default(group: str) -> bool:
    """判断候选分组是否默认允许进入第一版训练配置。"""
    return group in {"attention_projection", "mlp_projection", "speech_projection"}
