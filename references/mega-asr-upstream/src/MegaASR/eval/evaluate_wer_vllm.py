# coding=utf-8
import argparse
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "src"))
sys.path.append(str(Path(__file__).resolve().parent))

from infer_vllm import build_vllm_kwargs, materialized_lora_dir
from evaluate_wer import (
    MAX_NEW_TOKENS,
    compute_one_error,
    detect_language,
    get_audio_field,
    resolve_audio,
    unwrap_prediction,
)

DEFAULT_BATCH_SIZE = 1


def str2bool(value):
    if isinstance(value, bool):
        return value
    return value.lower() in ("1", "true", "yes", "y")


def main():
    parser = argparse.ArgumentParser(
        "Run Mega-ASR vLLM materialized-LoRA inference and compute WER/CER."
    )
    parser.add_argument("--ckpt_dir", required=True, help="Mega-ASR ckpt root")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--vllm_materialize_lora_force",
        type=str2bool,
        default=False,
        help="rebuild the materialized LoRA checkpoint even if the cache is fresh",
    )
    parser.add_argument(
        "--vllm_materialize_lora_device_map",
        default=None,
        help="device_map used only while materializing LoRA, e.g. cpu or cuda:0",
    )
    parser.add_argument("--gpu_memory_utilization", type=float, default=None)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--max_num_seqs", type=int, default=None)
    parser.add_argument("--max_num_batched_tokens", type=int, default=None)
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    with open(args.input_jsonl, "r", encoding="utf-8-sig") as f:
        data = [json.loads(line) for line in f if line.strip()]

    from MegaASR.model.megaASR import MegaASR

    ckpt_dir = Path(args.ckpt_dir).expanduser()
    vllm_kwargs = build_vllm_kwargs(args)

    model = MegaASR(
        model_path=ckpt_dir / "Qwen3-ASR-1.7B",
        lora_dir=ckpt_dir / "mega-asr-merged",
        routing_enabled=False,
        backend="vllm",
        vllm_apply_lora_on_load=True,
        vllm_materialized_lora_dir=materialized_lora_dir(ckpt_dir),
        vllm_materialize_lora_force=args.vllm_materialize_lora_force,
        vllm_materialize_lora_device_map=args.vllm_materialize_lora_device_map,
        max_inference_batch_size=args.batch_size,
        max_new_tokens=MAX_NEW_TOKENS,
        **vllm_kwargs,
    )

    outputs, total_edits, total_ref_len = [], 0, 0

    for i in tqdm(range(0, len(data), args.batch_size), desc="evaluating-vllm"):
        batch = data[i:i + args.batch_size]
        audio_paths = [resolve_audio(get_audio_field(x), args.input_jsonl) for x in batch]
        results = model.batch_infer(audio_paths)
        for item, pred in zip(batch, results):
            pred = unwrap_prediction(pred).strip()
            language = item.get("language") or detect_language(item["answer"], pred)
            score, detail = compute_one_error(item["answer"], pred, language)
            edits = detail["err"]
            ref_len = detail["nref"]
            metric = "cer" if language in {"zh", "yue"} else "wer"
            item["prediction"] = pred
            item["metric"] = metric
            item["wer"] = round(float(score), 6)
            item["num_edits"] = int(edits)
            item["ref_len"] = int(ref_len)
            total_edits += edits
            total_ref_len += ref_len
            outputs.append(item)

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"samples: {len(outputs)}")
    print(f"overall_error: {total_edits / total_ref_len if total_ref_len else 0.0:.6f}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
