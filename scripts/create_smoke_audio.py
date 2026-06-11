#!/usr/bin/env python3
"""为 baseline ASR 检查生成本地 clean/noise smoke-test 音频。

这个脚本刻意避开 Python TTS 依赖。在 macOS 上，它使用系统自带的
`say` 命令合成 clean 语音，再转成 16 kHz mono WAV，最后用 Python
标准库生成一个加噪版本。

生成文件只用于 smoke test：验证 manifest 解析、音频加载、推理结果写出
和 WER/CER 评测流程是否能跑通。它们不应被当成真实 benchmark。
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import tempfile
import wave
from array import array
from pathlib import Path


DEFAULT_TEXT = "Please call me when the meeting starts."


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
    """把 macOS `say` 生成的 AIFF 转成模型友好的 16-bit mono WAV。

    smoke test 阶段应尽量让 Qwen3-ASR/音频预处理看到稳定输入格式。因此这里
    统一转成 16 kHz mono WAV，而不是保留平台相关的 `say` 原始输出。
    """
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    # 优先使用 ffmpeg，因为它在 Colab 和本地 ML 环境都常见。
    # 如果没有 ffmpeg，则回退到 macOS 自带 afconvert，减少额外依赖。
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
    """使用 macOS `say` 合成 clean 参考语音。

    这里不用 Python TTS 包，是为了避免它们拉取与 Colab 预装 ML 栈冲突的
    依赖版本。生成的声音虽然是机器音，但足够验证 baseline 流程连通性。
    """
    say = require_command("say")
    with tempfile.TemporaryDirectory() as tmpdir:
        aiff_path = Path(tmpdir) / "clean.aiff"
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


def add_noise(clean_path: Path, noise_path: Path, noise_level: int, seed: int) -> None:
    """叠加高斯采样噪声，生成可复现的 noise 版本。

    这里故意保持简单：目标只是生成一个 degraded smoke-test 文件，而不是
    模拟真实声学环境。真实增强会在 roadmap 的音频增强阶段再实现。
    """
    params, clean = read_pcm16_mono(clean_path)
    rng = random.Random(seed)
    noisy = array("h")
    for sample in clean:
        value = int(sample + rng.gauss(0, noise_level))
        value = max(-32768, min(32767, value))
        noisy.append(value)
    write_pcm16(noise_path, params, noisy)


def write_manifest(
    root: Path,
    manifest_path: Path,
    clean_path: Path,
    noise_path: Path,
    text: str,
    absolute_paths: bool,
) -> None:
    """为 clean/noise 两条音频写入 baseline smoke manifest。

    字段与 baseline 计划中的评测 JSONL 约定保持一致：
    `audio` 指向 WAV 文件，`answer` 是参考转写，`language` 用于后续选择
    WER/CER，`scenario` 用于按场景聚合指标。
    """
    def audio_value(path: Path) -> str:
        # 相对路径适合本地仓库测试；绝对路径适合把 manifest 拷到
        # Colab/Google Drive 后使用固定路径。
        if absolute_paths:
            return str(path.resolve())
        return str(path.resolve().relative_to(root.resolve()))

    rows = [
        {
            "audio": audio_value(clean_path),
            "answer": text,
            "language": "en",
            "scenario": "clean",
            "source": "macos_say",
            "is_degraded": False,
        },
        {
            "audio": audio_value(noise_path),
            "answer": text,
            "language": "en",
            "scenario": "noise",
            "source": "macos_say_plus_noise",
            "is_degraded": True,
        },
    ]

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    """解析命令行参数，用于可复现地生成本地 smoke 数据。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Transcript to synthesize.")
    parser.add_argument("--voice", default="Alex", help="macOS `say` voice name.")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--noise-level", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="data/local_smoke/audio")
    parser.add_argument("--manifest", default="data/jsonl/baseline_smoke.local.jsonl")
    parser.add_argument("--absolute-paths", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    output_dir = resolve_path(root, args.output_dir)
    manifest_path = resolve_path(root, args.manifest)
    clean_path = output_dir / "clean_0001.wav"
    noise_path = output_dir / "noise_0001.wav"

    existing = [p for p in [clean_path, noise_path, manifest_path] if p.exists()]
    if existing and not args.force:
        existing_text = "\n".join(str(p) for p in existing)
        raise SystemExit(f"Refusing to overwrite existing files. Use --force.\n{existing_text}")

    # 构建最小可用 ASR smoke 集：
    # 1. 一条带已知 transcript 的 clean 合成语音
    # 2. 一条同 transcript 的 degraded/noisy 版本
    # 3. 一个可被后续推理/评测脚本读取的 JSONL manifest
    synthesize_clean_audio(args.text, clean_path, args.sample_rate, args.voice)
    add_noise(clean_path, noise_path, args.noise_level, args.seed)
    write_manifest(root, manifest_path, clean_path, noise_path, args.text, args.absolute_paths)

    print("Generated smoke audio:")
    print(f"  clean:    {clean_path}")
    print(f"  degraded: {noise_path}")
    print(f"  manifest: {manifest_path}")


if __name__ == "__main__":
    main()
