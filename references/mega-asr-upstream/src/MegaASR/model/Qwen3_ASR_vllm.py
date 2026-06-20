from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Any

from .utils.materialize_lora import materialize_lora_checkpoint


class Qwen3ASRVLLM:
    NAME = "Qwen3-ASR-1.7B-vLLM"
    HF_REPO_ID = "Qwen/Qwen3-ASR-1.7B"
    DEFAULT_MODEL_DIR = "ckpt/Mega-ASR/Qwen3-ASR-1.7B"

    def __init__(
        self,
        model_path: str | os.PathLike[str] | None = None,
        *,
        repo_id: str | None = None,
        max_inference_batch_size: int = 32,
        max_new_tokens: int = 2048,
        download_kwargs: dict[str, Any] | None = None,
        apply_lora_on_load: bool = False,
        lora_dir: str | os.PathLike[str] | None = None,
        materialized_lora_dir: str | os.PathLike[str] | None = None,
        materialize_lora_force: bool = False,
        materialize_lora_device_map: str | None = None,
        materialize_lora_kwargs: dict[str, Any] | None = None,
        **llm_kwargs: Any,
    ) -> None:
        try:
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:
            raise ImportError(
                'vLLM backend requires qwen-asr with vLLM extras. '
                'Install it with: pip install -U "qwen-asr[vllm]"'
            ) from exc

        repo_id = repo_id or self.HF_REPO_ID
        self.model_path = str(Path(model_path or self.DEFAULT_MODEL_DIR).expanduser())
        if not self._has_local_model(self.model_path):
            self.model_path = self.download_model(
                self.model_path,
                repo_id=repo_id,
                **(download_kwargs or {}),
            )

        if apply_lora_on_load:
            if lora_dir is None:
                raise ValueError("`lora_dir` is required when `apply_lora_on_load=True`.")
            materialized_lora_dir = materialized_lora_dir or (
                Path(lora_dir).expanduser().parent / "mega-asr-vllm-materialized"
            )
            self.model_path = materialize_lora_checkpoint(
                base_model_path=self.model_path,
                lora_dir=lora_dir,
                output_dir=materialized_lora_dir,
                force=materialize_lora_force,
                device_map=materialize_lora_device_map,
                **(materialize_lora_kwargs or {}),
            )

        try:
            self.model = Qwen3ASRModel.LLM(
                model=self.model_path,
                max_inference_batch_size=max_inference_batch_size,
                max_new_tokens=max_new_tokens,
                **llm_kwargs,
            )
        except ImportError as exc:
            message = str(exc)
            if "vllm._C" in message and platform.system() == "Windows":
                raise ImportError(
                    "vLLM does not support native Windows GPU inference because "
                    "the compiled extension `vllm._C` is unavailable. Run the "
                    "vLLM backend in Linux, WSL2, or a Linux Docker container. "
                    "The Transformers backend works on this Windows environment."
                ) from exc
            raise

    @staticmethod
    def _has_local_model(model_path: str | os.PathLike[str]) -> bool:
        path = Path(model_path).expanduser()
        return path.is_dir() and (path / "config.json").is_file()

    @staticmethod
    def download_model(
        model_path: str | os.PathLike[str],
        *,
        repo_id: str,
        **snapshot_kwargs: Any,
    ) -> str:
        from huggingface_hub import snapshot_download

        local_dir = Path(model_path).expanduser()
        local_dir.mkdir(parents=True, exist_ok=True)

        return snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            **snapshot_kwargs,
        )

    @staticmethod
    def _normalize_audio(audio: Any) -> Any:
        if isinstance(audio, os.PathLike):
            return str(audio)
        if isinstance(audio, (list, tuple)):
            return [str(item) if isinstance(item, os.PathLike) else item for item in audio]
        return audio

    def infer(
        self,
        audio: Any,
        *,
        language: str | list[str] | None = None,
        return_objects: bool = False,
        **transcribe_kwargs: Any,
    ) -> str | list[str] | Any:
        audio = self._normalize_audio(audio)

        results = self.model.transcribe(
            audio=audio,
            language=language,
            **transcribe_kwargs,
        )

        if return_objects:
            return results

        if isinstance(results, list):
            return [str(getattr(result, "text", result)).strip() for result in results]

        return str(getattr(results, "text", results)).strip()

    def init_streaming_state(self, **kwargs: Any) -> Any:
        return self.model.init_streaming_state(**kwargs)

    def streaming_transcribe(self, audio: Any, state: Any) -> Any:
        return self.model.streaming_transcribe(audio, state)

    def finish_streaming_transcribe(self, state: Any) -> Any:
        return self.model.finish_streaming_transcribe(state)


def get_mega_asr(*args: Any, **kwargs: Any) -> Qwen3ASRVLLM:
    return Qwen3ASRVLLM(*args, **kwargs)
