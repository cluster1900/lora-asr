from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .lora_switch import LoRADeltaSwitch


MARKER_NAME = "mega_asr_materialized_lora.json"


def _path_mtime(path: Path) -> float | None:
    return path.stat().st_mtime if path.exists() else None


def _marker_payload(
    *,
    base_model_path: str,
    lora_dir: str,
) -> dict[str, Any]:
    base_path = Path(base_model_path).expanduser()
    adapter_path = Path(lora_dir).expanduser()
    return {
        "base_model_path": str(base_path),
        "lora_dir": str(adapter_path),
        "base_config_mtime": _path_mtime(base_path / "config.json"),
        "adapter_config_mtime": _path_mtime(adapter_path / "adapter_config.json"),
        "adapter_safetensors_mtime": _path_mtime(adapter_path / "adapter_model.safetensors"),
        "adapter_bin_mtime": _path_mtime(adapter_path / "adapter_model.bin"),
        "mega_lora_blocks_mtime": _path_mtime(adapter_path / "mega_lora_blocks.json"),
    }


def is_materialized_lora_fresh(
    output_dir: str | os.PathLike[str],
    *,
    base_model_path: str | os.PathLike[str],
    lora_dir: str | os.PathLike[str],
) -> bool:
    output_path = Path(output_dir).expanduser()
    marker_path = output_path / MARKER_NAME
    if not (output_path / "config.json").is_file() or not marker_path.is_file():
        return False

    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    expected = _marker_payload(
        base_model_path=str(Path(base_model_path).expanduser()),
        lora_dir=str(Path(lora_dir).expanduser()),
    )
    return marker == expected


def materialize_lora_checkpoint(
    *,
    base_model_path: str | os.PathLike[str],
    lora_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    force: bool = False,
    device_map: str | None = None,
    dtype: Any | None = None,
    max_shard_size: str = "2GB",
    keep_delta_on_gpu: bool = False,
    **from_pretrained_kwargs: Any,
) -> str:
    base_model_path = str(Path(base_model_path).expanduser())
    lora_dir = str(Path(lora_dir).expanduser())
    output_path = Path(output_dir).expanduser()

    if not force and is_materialized_lora_fresh(
        output_path,
        base_model_path=base_model_path,
        lora_dir=lora_dir,
    ):
        return str(output_path)

    import torch
    from qwen_asr import Qwen3ASRModel

    if device_map is None:
        device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        dtype = torch.bfloat16 if device_map != "cpu" else torch.float32

    output_path.mkdir(parents=True, exist_ok=True)

    asr = Qwen3ASRModel.from_pretrained(
        base_model_path,
        dtype=dtype,
        device_map=device_map,
        max_inference_batch_size=1,
        max_new_tokens=1,
        **from_pretrained_kwargs,
    )

    target_module = getattr(asr.model, "model", asr.model)
    switch = LoRADeltaSwitch(keep_delta_on_gpu=keep_delta_on_gpu)
    switch.add_adapter(
        parent_module=target_module,
        adapter_dir=lora_dir,
        name="mega_asr_materialized_adapter",
    )
    switch.set_active(True)

    asr.model.save_pretrained(
        str(output_path),
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )
    asr.processor.save_pretrained(str(output_path))

    marker = _marker_payload(base_model_path=base_model_path, lora_dir=lora_dir)
    (output_path / MARKER_NAME).write_text(
        json.dumps(marker, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return str(output_path)
