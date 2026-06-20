from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file as safe_load_file

from .Qwen3_ASR import Qwen3ASR
from .Qwen3_ASR_vllm import Qwen3ASRVLLM
from .router import AudioQualityRouter
from .utils.lora_switch import LoRADeltaSwitch

class MegaASR:
    NAME = "Mega-ASR"
    DEFAULT_MODEL_DIR = Qwen3ASR.DEFAULT_MODEL_DIR
    DEFAULT_LORA_DIR = "ckpt/Mega-ASR/mega-asr-merged"
    DEFAULT_ROUTER_CHECKPOINT = AudioQualityRouter.DEFAULT_CHECKPOINT
    DOWNLOAD_URLS = {
        "lora": None,
        "router": None,
    }

    def __init__(
        self,
        model_path: str | os.PathLike[str] | None = None,
        *,
        lora_dir: str | os.PathLike[str] | None = None,
        router_checkpoint: str | os.PathLike[str] | None = None,
        routing_enabled: bool = True,
        quality_threshold: float = 0.5,
        device_map: str | None = None,
        quality_device: str | None = None,
        max_inference_batch_size: int = 32,
        max_new_tokens: int = 256,
        keep_delta_on_gpu: bool = True,
        backend: str = "transformers",
        vllm_lora_model_path: str | os.PathLike[str] | None = None,
        vllm_apply_lora_on_load: bool = False,
        vllm_materialized_lora_dir: str | os.PathLike[str] | None = None,
        vllm_materialize_lora_force: bool = False,
        vllm_materialize_lora_device_map: str | None = None,
        **model_kwargs: Any,
    ) -> None:
        if backend not in {"transformers", "vllm"}:
            raise ValueError("backend must be either 'transformers' or 'vllm'.")

        self.backend = backend
        self.model_path = str(Path(model_path or self.DEFAULT_MODEL_DIR).expanduser())
        self.lora_dir = str(Path(lora_dir or self.DEFAULT_LORA_DIR).expanduser())
        self.vllm_lora_model_path = (
            str(Path(vllm_lora_model_path).expanduser())
            if vllm_lora_model_path is not None
            else None
        )
        self.vllm_apply_lora_on_load = vllm_apply_lora_on_load
        self.vllm_materialized_lora_dir = (
            str(Path(vllm_materialized_lora_dir).expanduser())
            if vllm_materialized_lora_dir is not None
            else None
        )
        self.router_checkpoint = str(
            Path(router_checkpoint or self.DEFAULT_ROUTER_CHECKPOINT).expanduser()
        )
        self.routing_enabled = routing_enabled

        self.stats = {"total": 0, "use_base": 0, "use_lora": 0}
        self.switch_times: list[dict[str, float | str]] = []

        if self.backend == "vllm" and self.vllm_apply_lora_on_load:
            self.routing_enabled = False

        self.router = None
        if self.routing_enabled:
            self.router = AudioQualityRouter(
                checkpoint_path=self.router_checkpoint,
                device=quality_device,
                threshold=quality_threshold,
            )

        self.lora_switch = None
        self.asr_lora = None

        if self.backend == "vllm":
            self.asr = Qwen3ASRVLLM(
                model_path=self.model_path,
                max_inference_batch_size=max_inference_batch_size,
                max_new_tokens=max_new_tokens,
                apply_lora_on_load=self.vllm_apply_lora_on_load,
                lora_dir=self.lora_dir,
                materialized_lora_dir=self.vllm_materialized_lora_dir,
                materialize_lora_force=vllm_materialize_lora_force,
                materialize_lora_device_map=vllm_materialize_lora_device_map,
                **model_kwargs,
            )
            if self.vllm_lora_model_path is not None:
                self.asr_lora = Qwen3ASRVLLM(
                    model_path=self.vllm_lora_model_path,
                    max_inference_batch_size=max_inference_batch_size,
                    max_new_tokens=max_new_tokens,
                    **model_kwargs,
                )
        else:
            self.asr = Qwen3ASR(
                model_path=self.model_path,
                device_map=device_map,
                max_inference_batch_size=max_inference_batch_size,
                max_new_tokens=max_new_tokens,
                **model_kwargs,
            )

            self.lora_switch = LoRADeltaSwitch(keep_delta_on_gpu=keep_delta_on_gpu)
            self._load_loras()
            self._set_lora(True)

    @classmethod
    def download(cls, name: str, target_dir: str | os.PathLike[str]) -> str:
        url = cls.DOWNLOAD_URLS.get(name)
        if not url:
            raise NotImplementedError(f"Download URL for {name} is not set yet.")

        from huggingface_hub import snapshot_download

        return snapshot_download(
            repo_id=url,
            local_dir=str(Path(target_dir).expanduser()),
            local_dir_use_symlinks=False,
        )

    def _load_loras(self) -> None:
        if self.lora_switch is None:
            return
        self.lora_switch.add_adapter(
            parent_module=self.asr.model.model,
            adapter_dir=self.lora_dir,
            name="mega_asr_merged_adapter",
        )

    def _set_lora(self, active: bool) -> None:
        if self.lora_switch is None:
            return
        elapsed = self.lora_switch.set_active(active)
        if elapsed > 0:
            direction = "base_to_lora" if active else "lora_to_base"
            self.switch_times.append({"direction": direction, "time": elapsed})

    @staticmethod
    def _unwrap_audio(audio: Any) -> Any:
        if isinstance(audio, (list, tuple)) and len(audio) == 1:
            return audio[0]
        return audio

    def _route(self, audio: Any) -> tuple[bool, float | None, str]:
        if self.routing_enabled and self.router is not None:
            is_degraded, degraded_prob = self.router.predict(audio)
            return is_degraded, degraded_prob, "router"

        if self.backend == "vllm" and self.vllm_apply_lora_on_load:
            return True, None, "materialized_lora"

        if self.backend == "vllm" and self.asr_lora is None:
            return False, None, "default_base"

        return True, None, "default"

    def infer(
        self,
        audio: Any,
        *,
        language: str | None = None,
        return_objects: bool = False,
        return_route: bool = False,
        **transcribe_kwargs: Any,
    ) -> Any:
        audio = self._unwrap_audio(audio)
        use_lora, degraded_prob, route_source = self._route(audio)

        asr = self._select_asr(use_lora)
        self._set_lora(use_lora)
        result = asr.infer(
            audio,
            language=language,
            return_objects=return_objects,
            **transcribe_kwargs,
        )

        self.stats["total"] += 1
        if use_lora:
            self.stats["use_lora"] += 1
        else:
            self.stats["use_base"] += 1

        if return_route:
            return {
                "text": result,
                "use_lora": use_lora,
                "degraded_prob": degraded_prob,
                "route_source": route_source,
            }

        return result

    def infer_with_lora(self, audio: Any, **kwargs: Any) -> Any:
        if self.backend == "vllm":
            return self._select_asr(True).infer(self._unwrap_audio(audio), **kwargs)
        self._set_lora(True)
        return self.asr.infer(self._unwrap_audio(audio), **kwargs)

    def infer_without_lora(self, audio: Any, **kwargs: Any) -> Any:
        if self.backend == "vllm":
            return self._select_asr(False).infer(self._unwrap_audio(audio), **kwargs)
        self._set_lora(False)
        return self.asr.infer(self._unwrap_audio(audio), **kwargs)

    def _select_asr(self, use_lora: bool) -> Any:
        if self.backend != "vllm":
            return self.asr

        if use_lora:
            if self.asr_lora is None:
                if self.routing_enabled:
                    raise RuntimeError(
                        "vLLM routing needs `vllm_lora_model_path` pointing to a "
                        "merged Mega-ASR model directory. Disable routing or pass "
                        "`vllm_lora_model_path`."
                    )
                return self.asr
            return self.asr_lora

        return self.asr

    def init_streaming_state(self, **kwargs: Any) -> Any:
        if self.backend != "vllm":
            raise RuntimeError("Streaming inference is only supported by the vLLM backend.")
        return self._select_asr(True).init_streaming_state(**kwargs)

    def streaming_transcribe(self, audio: Any, state: Any) -> Any:
        if self.backend != "vllm":
            raise RuntimeError("Streaming inference is only supported by the vLLM backend.")
        return self._select_asr(True).streaming_transcribe(audio, state)

    def finish_streaming_transcribe(self, state: Any) -> Any:
        if self.backend != "vllm":
            raise RuntimeError("Streaming inference is only supported by the vLLM backend.")
        return self._select_asr(True).finish_streaming_transcribe(state)

    @torch.no_grad()
    def batch_infer(self, audios: list[Any], **kwargs: Any) -> list[Any]:
        audio_paths = [self._unwrap_audio(audio) for audio in audios]
        routes = [self._route(audio) for audio in audio_paths]

        if self.backend == "vllm" and self.asr_lora is None:
            results = [self.asr.infer(audio_path, **kwargs) for audio_path in audio_paths]
            self.stats["total"] += len(audio_paths)
            if self.vllm_apply_lora_on_load:
                self.stats["use_lora"] += len(audio_paths)
            else:
                self.stats["use_base"] += len(audio_paths)
            return results

        base_indices = [idx for idx, route in enumerate(routes) if not route[0]]
        lora_indices = [idx for idx, route in enumerate(routes) if route[0]]

        results: list[Any] = [None] * len(audio_paths)
        groups = [("lora", lora_indices), ("base", base_indices)]
        if self.lora_switch is not None and not self.lora_switch.active:
            groups = [("base", base_indices), ("lora", lora_indices)]

        for mode, indices in groups:
            if not indices:
                continue

            use_lora = mode == "lora"
            asr = self._select_asr(use_lora)
            self._set_lora(use_lora)

            for idx in indices:
                results[idx] = asr.infer(audio_paths[idx], **kwargs)
                if use_lora:
                    self.stats["use_lora"] += 1
                else:
                    self.stats["use_base"] += 1

        self.stats["total"] += len(audio_paths)
        return results


def get_mega_asr(*args: Any, **kwargs: Any) -> MegaASR:
    return MegaASR(*args, **kwargs)


def get_Mega_ASR(*args: Any, **kwargs: Any) -> MegaASR:
    return get_mega_asr(*args, **kwargs)
