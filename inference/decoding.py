"""Shared, explicit greedy decoding contract for official Qwen3-ASR models."""

from typing import Any, Mapping


def positive_token_budget(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("max_new_tokens must be a positive integer")
    if isinstance(value, str) and value.isdecimal():
        value = int(value)
    if not isinstance(value, int) or value <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    return value


def decoding_contract(max_new_tokens: int) -> dict:
    return {
        "max_new_tokens": positive_token_budget(max_new_tokens),
        "do_sample": False,
        "num_beams": 1,
        "num_return_sequences": 1,
    }


def training_token_budget(config: Mapping[str, Any]) -> int:
    decoding = config.get("decoding", {})
    if not isinstance(decoding, Mapping) or set(decoding) - {"max_new_tokens"}:
        raise ValueError("decoding only supports max_new_tokens")
    return positive_token_budget(decoding.get("max_new_tokens", 128))


def configure_greedy_model(asr_model: Any, max_new_tokens: int) -> None:
    """Use the public wrapper budget and Transformers generation config.

    Explicit sampling kwargs in training override the greedy defaults.
    """
    contract = decoding_contract(max_new_tokens)
    asr_model.max_new_tokens = contract["max_new_tokens"]
    for model in (asr_model.model, getattr(asr_model.model, "thinker", None)):
        generation = getattr(model, "generation_config", None)
        if generation is not None:
            for key, value in contract.items():
                setattr(generation, key, value)


def check_resume_decoding(state: Mapping[str, Any], max_new_tokens: int) -> None:
    # Legacy RL checkpoints used a hard-coded 128-token rollout budget.
    saved = state.get("decoding", decoding_contract(128))
    if saved != decoding_contract(max_new_tokens):
        raise ValueError("checkpoint decoding contract differs; start a fresh run")
