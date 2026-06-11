#!/usr/bin/env python3
"""生成 baseline MVP 评测用的 150 条本地合成音频。

目标场景固定为：

- clean：30 条
- noise：30 条
- reverb：30 条
- far_field：30 条
- dropout：30 条

脚本先使用 macOS `say` 合成 30 条 clean 语音，再从同一批 clean
音频派生四类 degraded 音频。输出 JSONL manifest 可直接交给
`inference/qwen3_asr_base_infer.py` 做 Qwen3-ASR baseline 推理。

注意：该数据集只用于工程闭环、Colab 批处理和评测输出验证。它由 TTS
和规则退化生成，不能替代真实鲁棒 ASR benchmark。默认 `--profile hard`
会主动降低 degraded 场景质量，用于尽快压出 baseline 错误。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import subprocess
import sys
import tempfile
import wave
from array import array
from collections import Counter
from pathlib import Path
from typing import Callable


SCENARIOS = ("clean", "noise", "reverb", "far_field", "dropout")

PROFILE_SETTINGS = {
    "mild": {
        "noise_voice_gain": 1.0,
        "noise_level": (1100, 2200),
        "noise_hum": 0,
        "reverb_dry_gain": 0.82,
        "reverb_delays_ms": [45, 95, 155],
        "reverb_jitter_ms": 16,
        "reverb_decays": [0.34, 0.22, 0.13],
        "reverb_noise": 0,
        "far_lowpass_alpha": 0.18,
        "far_voice_gain": 0.58,
        "far_echo_gain": 0.18,
        "far_delay_range_ms": (120, 220),
        "far_room_noise": 260,
        "far_quantize_step": 1,
        "dropout_gap_rate": 0.75,
        "dropout_gap_ms": (40, 140),
        "dropout_gains": [0.0, 0.02, 0.05],
        "dropout_residual_noise": 0,
    },
    "medium": {
        "noise_voice_gain": 0.88,
        "noise_level": (2600, 4300),
        "noise_hum": 280,
        "reverb_dry_gain": 0.68,
        "reverb_delays_ms": [60, 125, 235, 360],
        "reverb_jitter_ms": 26,
        "reverb_decays": [0.48, 0.35, 0.24, 0.16],
        "reverb_noise": 80,
        "far_lowpass_alpha": 0.11,
        "far_voice_gain": 0.43,
        "far_echo_gain": 0.27,
        "far_delay_range_ms": (160, 300),
        "far_room_noise": 480,
        "far_quantize_step": 64,
        "dropout_gap_rate": 1.45,
        "dropout_gap_ms": (80, 220),
        "dropout_gains": [0.0, 0.0, 0.02],
        "dropout_residual_noise": 260,
    },
    "hard": {
        "noise_voice_gain": 0.72,
        "noise_level": (5200, 8200),
        "noise_hum": 780,
        "reverb_dry_gain": 0.52,
        "reverb_delays_ms": [85, 175, 320, 520],
        "reverb_jitter_ms": 45,
        "reverb_decays": [0.66, 0.52, 0.38, 0.27],
        "reverb_noise": 180,
        "far_lowpass_alpha": 0.055,
        "far_voice_gain": 0.28,
        "far_echo_gain": 0.42,
        "far_delay_range_ms": (220, 420),
        "far_room_noise": 920,
        "far_quantize_step": 192,
        "dropout_gap_rate": 3.55,
        "dropout_gap_ms": (180, 460),
        "dropout_gains": [0.0, 0.0, 0.0, 0.0, 0.01],
        "dropout_residual_noise": 80,
    },
}

DEFAULT_TEXTS = [
    "Please call me when the meeting starts.",
    "The train will arrive at platform seven.",
    "I need a quiet room for the interview.",
    "Could you send the report before noon?",
    "The package was delivered this morning.",
    "We should review the budget again tomorrow.",
    "Turn left after the second traffic light.",
    "The customer asked for a faster response.",
    "My flight leaves at six thirty tonight.",
    "Please save the file in the shared folder.",
    "The doctor will call you after lunch.",
    "I forgot to charge my phone last night.",
    "The meeting notes are ready for review.",
    "She ordered coffee and a small sandwich.",
    "The office network is slow this afternoon.",
    "During the weekly project meeting, the product manager asked everyone to compare the clean audio results with the noisy recordings before making a training decision.",
    "The customer support team noticed that several callers sounded distant because their phones were lying on the table instead of being held close to the microphone.",
    "When the conference room door stayed open, background conversations from the hallway mixed with the speaker's voice and made the final sentence difficult to understand.",
    "Please review the evaluation report carefully, because the dropout samples may cause the model to skip important words near the middle of each utterance.",
    "After the software update finished, the engineer recorded another test sentence to check whether the new audio pipeline still saved every file with the correct sample rate.",
    "The researcher explained that clean speech regression should be measured separately, since a model can improve noisy transcription while becoming worse on simple recordings.",
    "If the router sends a clearly degraded clip to the base model by mistake, the prediction may contain repeated phrases or confident words that were never spoken.",
    "The training notebook should save the manifest, configuration, random seed, prediction file, and metrics summary so that the entire experiment can be reproduced later.",
    "A long sentence with several clauses is useful for testing whether the recognizer preserves word order when reverberation hides the beginning of the next phrase.",
    "The finance department left a detailed voicemail about the invoice number, payment deadline, and approval status, but the recording included strong background noise.",
    "Before we claim that the adapter improves robustness, we need to compare every degraded scenario against the same Qwen three ASR baseline using WER and CER.",
    "The far field recording sounded quiet and muffled, so the evaluator checked whether the model missed short function words or invented extra context at the end.",
    "This synthetic benchmark is not a replacement for real speech data, but it is helpful for catching broken inference code and unstable evaluation scripts early.",
    "When the network connection dropped for a moment, the audio stream lost several syllables, yet a robust recognizer should still recover the main instruction.",
    "The final validation sentence intentionally includes multiple details about meetings, reports, microphones, and noisy rooms to expose omissions in longer transcriptions.",
]


def project_root() -> Path:
    """根据脚本位置返回仓库根目录。"""
    return Path(__file__).resolve().parents[1]


def resolve_path(root: Path, value: str) -> Path:
    """解析命令行路径；相对路径默认相对于项目根目录。"""
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def run(cmd: list[str]) -> None:
    """执行子进程；命令失败时立即抛错。"""
    subprocess.run(cmd, check=True)


def require_command(name: str) -> str:
    """查找必需的系统命令；缺失时给出明确错误。"""
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"Required command not found: {name}")
    return path


def convert_aiff_to_wav(aiff_path: Path, wav_path: Path, sample_rate: int) -> None:
    """把 macOS `say` 生成的 AIFF 转成 16 kHz mono 16-bit WAV。"""
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        run([
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(aiff_path),
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            str(wav_path),
        ])
        return

    afconvert = shutil.which("afconvert")
    if afconvert:
        run([
            afconvert,
            "-f",
            "WAVE",
            "-d",
            f"LEI16@{sample_rate}",
            "-c",
            "1",
            str(aiff_path),
            str(wav_path),
        ])
        return

    raise RuntimeError("Need either ffmpeg or afconvert to convert synthesized audio.")


def synthesize_clean_audio(text: str, wav_path: Path, sample_rate: int, voice: str) -> None:
    """使用 macOS `say` 合成一条 clean 语音。

    如果指定 voice 不可用，`say` 会返回错误。这里保留错误，让调用者尽早
    知道本地环境不匹配，而不是悄悄生成不同声音导致实验不可复现。
    """
    say = require_command("say")
    with tempfile.TemporaryDirectory() as tmpdir:
        aiff_path = Path(tmpdir) / "speech.aiff"
        run([say, "-v", voice, "-o", str(aiff_path), text])
        convert_aiff_to_wav(aiff_path, wav_path, sample_rate)


def read_pcm16_mono(path: Path) -> tuple[wave._wave_params, array]:
    """读取 mono 16-bit PCM WAV，并返回有符号整数采样。"""
    with wave.open(str(path), "rb") as wf:
        params = wf.getparams()
        if params.sampwidth != 2:
            raise ValueError(f"Expected 16-bit PCM WAV, got sample width {params.sampwidth}.")
        if params.nchannels != 1:
            raise ValueError(f"Expected mono WAV, got {params.nchannels} channels.")
        frames = wf.readframes(params.nframes)

    samples = array("h")
    samples.frombytes(frames)
    if sys.byteorder != "little":
        samples.byteswap()
    return params, samples


def write_pcm16(path: Path, params: wave._wave_params, samples: array) -> None:
    """使用源 WAV 参数写回有符号整数采样。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = array("h", samples)
    if sys.byteorder != "little":
        out.byteswap()
    with wave.open(str(path), "wb") as wf:
        wf.setparams(params)
        wf.writeframes(out.tobytes())


def clamp_pcm16(value: float) -> int:
    """把浮点采样限制到 16-bit PCM 范围。"""
    return int(max(-32768, min(32767, round(value))))


def scale_samples(samples: array, gain: float) -> array:
    """整体调节音量，用于避免混响/回声叠加后削波。"""
    return array("h", [clamp_pcm16(sample * gain) for sample in samples])


def rms(samples: list[float] | array) -> float:
    """计算均方根能量，用于输出可解释的退化统计。"""
    if not samples:
        return 0.0
    return math.sqrt(sum(float(sample) * float(sample) for sample in samples) / len(samples))


def degradation_quality(clean: array, degraded: array) -> dict[str, float]:
    """比较 clean/degraded 采样，生成近似退化统计。

    这些统计不是严格声学指标，只用于确认当前 profile 是否真的降低了音频
    质量。例如 noise/far_field 的 SNR 变低、dropout 的 active silence
    变高，都说明这批数据更容易压出 ASR 错误。
    """
    n = min(len(clean), len(degraded))
    if n == 0:
        return {
            "snr_db": 0.0,
            "rms_ratio": 0.0,
            "active_near_silence_ratio": 0.0,
            "clipping_ratio": 0.0,
        }

    clean_part = clean[:n]
    degraded_part = degraded[:n]
    diff = [float(degraded_part[idx]) - float(clean_part[idx]) for idx in range(n)]
    clean_rms = rms(clean_part)
    diff_rms = rms(diff)
    degraded_rms = rms(degraded_part)

    if diff_rms <= 1e-9:
        snr_db = 99.0
    else:
        snr_db = 20.0 * math.log10(max(clean_rms, 1e-9) / diff_rms)

    active_indices = [idx for idx, sample in enumerate(clean_part) if abs(sample) >= 500]
    if active_indices:
        near_silence = sum(1 for idx in active_indices if abs(degraded_part[idx]) < 140)
        active_near_silence_ratio = near_silence / len(active_indices)
    else:
        active_near_silence_ratio = 0.0

    clipping_ratio = sum(1 for sample in degraded_part if abs(sample) >= 32760) / n
    return {
        "snr_db": round(snr_db, 4),
        "rms_ratio": round(degraded_rms / max(clean_rms, 1e-9), 4),
        "active_near_silence_ratio": round(active_near_silence_ratio, 4),
        "clipping_ratio": round(clipping_ratio, 6),
    }


def add_noise(
    params: wave._wave_params,
    clean: array,
    rng: random.Random,
    settings: dict[str, object],
) -> array:
    """生成 noise 场景：给 clean 语音叠加强背景噪声和低频嗡声。"""
    sample_rate = params.framerate
    noise_min, noise_max = settings["noise_level"]  # type: ignore[misc]
    noise_level = rng.randint(int(noise_min), int(noise_max))
    voice_gain = float(settings["noise_voice_gain"])
    hum_amplitude = float(settings["noise_hum"])
    hum_freq = rng.choice([50.0, 60.0, 90.0, 120.0])
    phase = rng.random() * math.tau

    out = array("h")
    for idx, sample in enumerate(clean):
        hum = hum_amplitude * math.sin((math.tau * hum_freq * idx / sample_rate) + phase)
        value = sample * voice_gain + rng.gauss(0, noise_level) + hum
        out.append(clamp_pcm16(value))
    return out


def add_reverb(
    params: wave._wave_params,
    clean: array,
    rng: random.Random,
    settings: dict[str, object],
) -> array:
    """生成 reverb 场景：用多段延迟回声近似房间混响。

    这不是物理精确的 RIR 卷积，只是用于 baseline MVP 阶段制造可复现的
    回声型退化。真实 RIR 增强会在后续音频增强阶段再实现。
    """
    sample_rate = params.framerate
    base_delays = [int(value) for value in settings["reverb_delays_ms"]]  # type: ignore[index]
    jitter = int(settings["reverb_jitter_ms"])
    delays_ms = [delay + rng.randint(-jitter, jitter) for delay in base_delays]
    decays = [float(value) for value in settings["reverb_decays"]]  # type: ignore[index]
    dry_gain = float(settings["reverb_dry_gain"])
    noise_level = float(settings["reverb_noise"])
    out = [float(sample) * dry_gain for sample in clean]
    for delay_ms, decay in zip(delays_ms, decays):
        delay = max(1, int(sample_rate * delay_ms / 1000))
        for idx in range(delay, len(out)):
            out[idx] += clean[idx - delay] * decay
    if noise_level:
        out = [value + rng.gauss(0, noise_level) for value in out]
    return array("h", [clamp_pcm16(value) for value in out])


def add_far_field(
    params: wave._wave_params,
    clean: array,
    rng: random.Random,
    settings: dict[str, object],
) -> array:
    """生成 far_field 场景：衰减音量、低通、弱回声和少量环境噪声。

    远场音频常见问题是直达声变弱、细节变闷、房间反射变明显。这里用
    简化规则模拟这些现象，重点是稳定复现而不是追求声学真实度。
    """
    sample_rate = params.framerate
    alpha = float(settings["far_lowpass_alpha"])
    voice_gain = float(settings["far_voice_gain"])
    echo_gain = float(settings["far_echo_gain"])
    delay_min, delay_max = settings["far_delay_range_ms"]  # type: ignore[misc]
    room_noise = float(settings["far_room_noise"])
    quantize_step = max(1, int(settings["far_quantize_step"]))
    lowpassed: list[float] = []
    prev = 0.0
    for sample in clean:
        prev = alpha * sample + (1 - alpha) * prev
        lowpassed.append(prev)

    delay = int(sample_rate * rng.uniform(float(delay_min) / 1000, float(delay_max) / 1000))
    out: list[float] = []
    for idx, value in enumerate(lowpassed):
        echo = lowpassed[idx - delay] * echo_gain if idx >= delay else 0.0
        degraded = value * voice_gain + echo + rng.gauss(0, room_noise)
        if quantize_step > 1:
            degraded = round(degraded / quantize_step) * quantize_step
        out.append(degraded)
    return array("h", [clamp_pcm16(value) for value in out])


def add_dropout(
    params: wave._wave_params,
    clean: array,
    rng: random.Random,
    settings: dict[str, object],
) -> array:
    """生成 dropout 场景：随机短时间静音或强衰减。

    dropout 用于模拟传输丢包、录音中断、蓝牙连接不稳等情况。这里按
    profile 指定的短窗口进行衰减。hard profile 会故意制造更频繁、更长的
    掉音，用来观察模型是否空输出、漏词或产生幻觉补全。
    """
    sample_rate = params.framerate
    out = array("h", clean)
    duration = len(out) / sample_rate
    gap_rate = float(settings["dropout_gap_rate"])
    gap_min_ms, gap_max_ms = settings["dropout_gap_ms"]  # type: ignore[misc]
    gains = [float(value) for value in settings["dropout_gains"]]  # type: ignore[index]
    residual_noise = float(settings["dropout_residual_noise"])
    gap_count = max(2, int(math.ceil(duration * gap_rate)))

    for _ in range(gap_count):
        gap_ms = rng.randint(int(gap_min_ms), int(gap_max_ms))
        gap_len = max(1, int(sample_rate * gap_ms / 1000))
        if len(out) <= gap_len:
            continue
        start = rng.randint(0, len(out) - gap_len)
        gain = rng.choice(gains)
        for idx in range(start, start + gap_len):
            out[idx] = clamp_pcm16(out[idx] * gain + rng.gauss(0, residual_noise * 0.15))

    if residual_noise:
        out = array("h", [clamp_pcm16(sample + rng.gauss(0, residual_noise)) for sample in out])
    return out


DEGRADERS: dict[str, Callable[[wave._wave_params, array, random.Random, dict[str, object]], array]] = {
    "noise": add_noise,
    "reverb": add_reverb,
    "far_field": add_far_field,
    "dropout": add_dropout,
}


def load_texts(texts_file: str | None, items_per_scenario: int) -> list[str]:
    """读取自定义文本；没有传入时使用内置 30 条英文短句。"""
    if texts_file:
        path = Path(texts_file).expanduser()
        texts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        texts = DEFAULT_TEXTS

    if len(texts) < items_per_scenario:
        raise ValueError(f"Need at least {items_per_scenario} texts, got {len(texts)}.")
    return texts[:items_per_scenario]


def text_length_bucket(index: int, total_items: int) -> str:
    """按 utterance 序号划分短/长文本。

    默认 30 条文本中，前 15 条是 short，后 15 条是 long。若用户通过
    `--items-per-scenario` 生成更小数据集，则按前半/后半保持同样规则。
    """
    midpoint = max(1, total_items // 2)
    return "short" if index <= midpoint else "long"


def reference_word_count(text: str) -> int:
    """统计英文参考文本词数，用于后续按长度分析 WER。"""
    return len(text.split())


def audio_value(root: Path, path: Path, absolute_paths: bool) -> str:
    """按命令行选项写入绝对或相对音频路径。"""
    if absolute_paths:
        return str(path.resolve())
    return str(path.resolve().relative_to(root.resolve()))


def assert_overwrite_allowed(paths: list[Path], force: bool) -> None:
    """避免误删已有本地音频；需要覆盖时必须显式传 `--force`。"""
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        existing_text = "\n".join(str(path) for path in existing)
        raise SystemExit(f"Refusing to overwrite existing files. Use --force.\n{existing_text}")


def summarize_quality(values: list[dict[str, float]]) -> dict[str, float]:
    """聚合一组退化统计，写入 stats 文件。"""
    if not values:
        return {
            "avg_snr_db": 0.0,
            "avg_rms_ratio": 0.0,
            "avg_active_near_silence_ratio": 0.0,
            "max_clipping_ratio": 0.0,
        }
    return {
        "avg_snr_db": round(sum(item["snr_db"] for item in values) / len(values), 4),
        "avg_rms_ratio": round(sum(item["rms_ratio"] for item in values) / len(values), 4),
        "avg_active_near_silence_ratio": round(
            sum(item["active_near_silence_ratio"] for item in values) / len(values),
            4,
        ),
        "max_clipping_ratio": round(max(item["clipping_ratio"] for item in values), 6),
    }


def build_dataset(args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, object]]:
    """生成音频、manifest rows 和 stats。"""
    root = project_root()
    output_dir = resolve_path(root, args.output_dir)
    manifest_path = resolve_path(root, args.manifest)
    stats_path = resolve_path(root, args.stats)
    texts = load_texts(args.texts_file, args.items_per_scenario)
    settings = PROFILE_SETTINGS[args.profile]

    assert_overwrite_allowed([output_dir, manifest_path, stats_path], args.force)
    if output_dir.exists() and args.force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    durations: list[float] = []
    word_counts: list[int] = []
    bucket_counts: Counter[str] = Counter()
    quality_by_scenario: dict[str, list[dict[str, float]]] = {
        "noise": [],
        "reverb": [],
        "far_field": [],
        "dropout": [],
    }

    for index, text in enumerate(texts, start=1):
        base_id = f"utt_{index:04d}"
        bucket = text_length_bucket(index, args.items_per_scenario)
        word_count = reference_word_count(text)
        word_counts.append(word_count)
        bucket_counts[bucket] += 1
        clean_path = output_dir / "clean" / f"clean_{index:04d}.wav"
        synthesize_clean_audio(text, clean_path, args.sample_rate, args.voice)

        params, clean_samples = read_pcm16_mono(clean_path)
        durations.append(round(params.nframes / params.framerate, 4))

        rows.append({
            "audio": audio_value(root, clean_path, args.absolute_paths),
            "answer": text,
            "language": "en",
            "scenario": "clean",
            "source": "macos_say",
            "is_degraded": False,
            "utterance_id": f"{base_id}_clean",
            "base_utterance_id": base_id,
            "degradation": "none",
            "profile": "clean",
            "seed": args.seed,
            "sample_rate": params.framerate,
            "text_length_bucket": bucket,
            "reference_word_count": word_count,
        })

        for scenario in ("noise", "reverb", "far_field", "dropout"):
            # 每个 utterance/scenario 使用独立随机种子，保证生成结果可复现，
            # 同时不同场景不会因为随机序列耦合而互相影响。
            scenario_seed = args.seed + index * 100 + SCENARIOS.index(scenario)
            rng = random.Random(scenario_seed)
            degraded_samples = DEGRADERS[scenario](params, clean_samples, rng, settings)
            quality = degradation_quality(clean_samples, degraded_samples)
            quality_by_scenario[scenario].append(quality)
            degraded_path = output_dir / scenario / f"{scenario}_{index:04d}.wav"
            write_pcm16(degraded_path, params, degraded_samples)

            rows.append({
                "audio": audio_value(root, degraded_path, args.absolute_paths),
                "answer": text,
                "language": "en",
                "scenario": scenario,
                "source": f"macos_say_plus_{scenario}",
                "is_degraded": True,
                "utterance_id": f"{base_id}_{scenario}",
                "base_utterance_id": base_id,
                "degradation": scenario,
                "profile": args.profile,
                "seed": scenario_seed,
                "sample_rate": params.framerate,
                "approx_snr_db": quality["snr_db"],
                "rms_ratio": quality["rms_ratio"],
                "active_near_silence_ratio": quality["active_near_silence_ratio"],
                "text_length_bucket": bucket,
                "reference_word_count": word_count,
            })

    counts = Counter(str(row["scenario"]) for row in rows)
    stats = {
        "total_rows": len(rows),
        "items_per_scenario": args.items_per_scenario,
        "scenario_counts": dict(sorted(counts.items())),
        "sample_rate": args.sample_rate,
        "voice": args.voice,
        "seed": args.seed,
        "profile": args.profile,
        "text_length_design": "first half short, second half long",
        "text_length_bucket_counts_per_scenario": dict(sorted(bucket_counts.items())),
        "avg_reference_word_count": round(sum(word_counts) / len(word_counts), 4),
        "min_reference_word_count": min(word_counts),
        "max_reference_word_count": max(word_counts),
        "manifest": audio_value(root, manifest_path, args.absolute_paths),
        "audio_dir": audio_value(root, output_dir, args.absolute_paths),
        "avg_clean_duration_seconds": round(sum(durations) / len(durations), 4),
        "min_clean_duration_seconds": min(durations),
        "max_clean_duration_seconds": max(durations),
        "degradation_stats": {
            scenario: summarize_quality(values)
            for scenario, values in sorted(quality_by_scenario.items())
        },
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return rows, stats


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items-per-scenario", type=int, default=30)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--voice", default="Alex", help="macOS `say` voice name.")
    parser.add_argument("--profile", default="hard", choices=sorted(PROFILE_SETTINGS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--texts-file", default=None, help="Optional UTF-8 text file, one utterance per line.")
    parser.add_argument("--output-dir", default="data/mvp_eval/audio")
    parser.add_argument("--manifest", default="data/jsonl/baseline_mvp_150.local.jsonl")
    parser.add_argument("--stats", default="data/jsonl/baseline_mvp_150_stats.local.json")
    parser.add_argument("--absolute-paths", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite generated audio/manifest/stats.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, stats = build_dataset(args)
    print("Generated MVP eval audio:")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Manifest rows: {len(rows)}")


if __name__ == "__main__":
    main()
