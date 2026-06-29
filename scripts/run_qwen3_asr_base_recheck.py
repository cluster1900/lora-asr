#!/usr/bin/env python3
"""Run a controlled Qwen3-ASR base recheck on a fixed manifest.

This script is a thin pipeline wrapper:

1. Run Qwen3-ASR base inference.
2. Score predictions with the shared WER/CER evaluator.
3. Run shared error analysis.
4. Compare the new base metrics against historical base and LoRA metrics.

It intentionally writes to a new output directory by default, so a base recheck
does not overwrite the historical baseline used in earlier experiment notes.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_COMPARE_METRICS = [
    "outputs/baseline_mvp_150/metrics.qwen3_asr_base.mvp_150.json",
    "outputs/lora_mvp_eval/metrics.qwen3_asr_lora_mvp.mvp_150.json",
    "outputs/lora_mvp_v2_eval/metrics.qwen3_asr_lora_mvp_v2.mvp_150.json",
]

DEFAULTS = {
    "manifest": "data/jsonl/baseline_mvp_150.local.jsonl",
    "audio_root": ".",
    "output_dir": "outputs/base_recheck_mvp_150",
    "label": "qwen3_asr_base_recheck",
    "dataset_name": "mvp_150",
    "model_id": "Qwen/Qwen3-ASR-1.7B",
    "dtype": "float16",
    "device_map": "cuda:0",
    "quantization": "4bit",
    "max_inference_batch_size": 1,
    "max_new_tokens": 128,
    "language": "English",
    "limit": 0,
    "top_k": 30,
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_path(path: str | Path) -> Path:
    item = Path(path).expanduser()
    if item.is_absolute():
        return item
    return PROJECT_ROOT / item


def relative_for_command(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = resolve_path(path)
    try:
        import yaml
    except ImportError:
        data = parse_simple_yaml(config_path.read_text(encoding="utf-8"))
    else:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Config must be a mapping: {config_path}")
    return data


def parse_simple_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the tiny YAML subset used by project config files.

    This fallback avoids requiring PyYAML for simple Colab helper scripts. It
    supports nested mappings and lists of scalar values, which is enough for
    `configs/baseline/qwen3_asr_base_recheck_mvp_150.yaml`.
    """
    items: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        items.append((indent, line.strip()))

    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]

    for index, (indent, content) in enumerate(items):
        while indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if content.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError(f"List item without list parent: {content}")
            parent.append(parse_simple_scalar(content[2:]))
            continue

        key, separator, value = content.partition(":")
        if not separator:
            raise ValueError(f"Expected key: value line: {content}")
        key = key.strip()
        value = value.strip()
        if not isinstance(parent, dict):
            raise ValueError(f"Mapping item without mapping parent: {content}")

        if value:
            parent[key] = parse_simple_scalar(value)
            continue

        next_item = items[index + 1] if index + 1 < len(items) else None
        container: Any = []
        if next_item is None or next_item[0] <= indent or not next_item[1].startswith("- "):
            container = {}
        parent[key] = container
        stack.append((indent, container))

    return root


def nested_get(config: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def choose(cli_value: Any, config: dict[str, Any], keys: tuple[str, ...], default_key: str) -> Any:
    if cli_value is not None:
        return cli_value
    configured = nested_get(config, keys)
    if configured is not None:
        return configured
    return DEFAULTS[default_key]


def infer_metric_label(path: Path) -> str:
    name = path.name
    if "base_recheck" in name:
        return "base_recheck"
    if "qwen3_asr_base" in name:
        return "historical_base"
    if "lora_mvp_v2" in name:
        return "lora_v2"
    if "lora_mvp" in name:
        return "lora_v1"
    return path.stem.replace("metrics.", "").replace(".", "_")


def load_metrics(path: Path) -> dict[str, dict[str, Any]]:
    data = read_json(path)
    metrics: dict[str, dict[str, Any]] = {"ALL": data.get("overall", {})}
    for row in data.get("by_scenario", []):
        metrics[str(row.get("group", ""))] = row
    return metrics


def metric_value(metrics: dict[str, dict[str, Any]], scenario: str) -> float | None:
    row = metrics.get(scenario)
    if not row:
        return None
    value = row.get("error_rate")
    return float(value) if value is not None else None


def run_command(cmd: list[str]) -> None:
    print("运行命令:")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def write_comparison_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "scenario",
        "base_recheck",
        "historical_base",
        "lora_v1",
        "lora_v2",
        "historical_base_minus_base_recheck",
        "lora_v1_minus_base_recheck",
        "lora_v2_minus_base_recheck",
    ]
    present = {key for row in rows for key in row}
    fieldnames = [key for key in preferred if key in present]
    fieldnames.extend(sorted(key for key in present if key not in fieldnames))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_metric_comparison(
    current_metrics_path: Path,
    compare_metric_paths: list[Path],
    comparison_json: Path,
    comparison_csv: Path,
) -> None:
    sources: list[tuple[str, Path, dict[str, dict[str, Any]]]] = [
        ("base_recheck", current_metrics_path, load_metrics(current_metrics_path)),
    ]
    seen = {current_metrics_path.resolve()}
    for path in compare_metric_paths:
        if not path.exists():
            print(f"[compare] skip missing metrics: {path}")
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        sources.append((infer_metric_label(path), path, load_metrics(path)))

    scenarios = ["ALL", "clean", "noise", "reverb", "dropout", "far_field"]
    extra_scenarios = sorted({scenario for _, _, metrics in sources for scenario in metrics if scenario not in scenarios})
    scenarios.extend(extra_scenarios)

    rows: list[dict[str, Any]] = []
    base_recheck_metrics = sources[0][2]
    for scenario in scenarios:
        row: dict[str, Any] = {"scenario": scenario}
        base_recheck_value = metric_value(base_recheck_metrics, scenario)
        for label, _, metrics in sources:
            value = metric_value(metrics, scenario)
            row[label] = round(value, 6) if value is not None else None
            if label != "base_recheck" and base_recheck_value is not None and value is not None:
                row[f"{label}_minus_base_recheck"] = round(value - base_recheck_value, 6)
                row[f"{label}_relative_to_base_recheck"] = round((value - base_recheck_value) / base_recheck_value, 6) if base_recheck_value else None
        rows.append(row)

    write_json(comparison_json, {
        "current": {
            "label": "base_recheck",
            "metrics": relative_for_command(current_metrics_path),
        },
        "sources": [
            {"label": label, "metrics": relative_for_command(path)}
            for label, path, _ in sources
        ],
        "by_scenario": rows,
    })
    write_comparison_csv(comparison_csv, rows)
    print(f"[compare] saved {comparison_json}")
    print(f"[compare] saved {comparison_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Optional YAML config for this recheck.")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--audio-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--dtype", default=None, choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--quantization", default=None, choices=["none", "4bit", "8bit", "nf4", "int8"])
    parser.add_argument("--max-inference-batch-size", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--language", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--compare-metrics",
        action="append",
        default=None,
        help="Metrics JSON to compare against. Can be passed multiple times.",
    )
    parser.add_argument(
        "--source-predictions-jsonl",
        default=None,
        help="Use an existing prediction JSONL instead of running model inference. For local pipeline tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    manifest = resolve_path(choose(args.manifest, config, ("input", "manifest"), "manifest"))
    audio_root = resolve_path(choose(args.audio_root, config, ("input", "audio_root"), "audio_root"))
    output_dir = resolve_path(choose(args.output_dir, config, ("output", "dir"), "output_dir"))
    label = str(choose(args.label, config, ("output", "label"), "label"))
    dataset_name = str(choose(args.dataset_name, config, ("output", "dataset_name"), "dataset_name"))
    model_id = str(choose(args.model_id, config, ("model", "id"), "model_id"))
    dtype = str(choose(args.dtype, config, ("model", "dtype"), "dtype"))
    device_map = str(choose(args.device_map, config, ("model", "device_map"), "device_map"))
    quantization = str(choose(args.quantization, config, ("model", "quantization"), "quantization"))
    max_inference_batch_size = int(choose(
        args.max_inference_batch_size,
        config,
        ("model", "max_inference_batch_size"),
        "max_inference_batch_size",
    ))
    max_new_tokens = int(choose(args.max_new_tokens, config, ("inference", "max_new_tokens"), "max_new_tokens"))
    language = str(choose(args.language, config, ("inference", "language"), "language"))
    limit = int(choose(args.limit, config, ("runtime", "limit"), "limit"))
    top_k = int(choose(args.top_k, config, ("analysis", "top_k"), "top_k"))

    configured_compare_metrics = nested_get(config, ("compare", "metrics"), None)
    compare_metrics_raw = args.compare_metrics if args.compare_metrics is not None else configured_compare_metrics
    if compare_metrics_raw is None:
        compare_metrics_raw = DEFAULT_COMPARE_METRICS
    if isinstance(compare_metrics_raw, (str, Path)):
        compare_metrics_raw = [str(compare_metrics_raw)]
    compare_metric_paths = [resolve_path(path) for path in compare_metrics_raw]

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = output_dir / f"predictions.{label}.{dataset_name}.jsonl"
    scored = output_dir / f"predictions.{label}.{dataset_name}.scored.jsonl"
    metrics_json = output_dir / f"metrics.{label}.{dataset_name}.json"
    scenario_csv = output_dir / f"metrics_by_scenario.{label}.{dataset_name}.csv"
    error_analysis_dir = output_dir / "error_analysis"
    comparison_json = output_dir / "comparison.json"
    comparison_csv = output_dir / "comparison_by_scenario.csv"

    source_predictions = resolve_path(args.source_predictions_jsonl) if args.source_predictions_jsonl else None
    if source_predictions:
        print(f"[inference] using existing predictions: {source_predictions}")
        shutil.copyfile(source_predictions, predictions)
        print(f"[inference] copied to {predictions}")
    else:
        infer_cmd = [
            sys.executable,
            "inference/qwen3_asr_base_infer.py",
            "--manifest", relative_for_command(manifest),
            "--audio-root", str(audio_root),
            "--output-jsonl", relative_for_command(predictions),
            "--model-id", model_id,
            "--dtype", dtype,
            "--device-map", device_map,
            "--quantization", quantization,
            "--max-inference-batch-size", str(max_inference_batch_size),
            "--max-new-tokens", str(max_new_tokens),
            "--language", language,
        ]
        if limit > 0:
            infer_cmd.extend(["--limit", str(limit)])
        run_command(infer_cmd)

    eval_cmd = [
        sys.executable,
        "evaluation/eval_wer.py",
        "--predictions-jsonl", relative_for_command(predictions),
        "--scored-jsonl", relative_for_command(scored),
        "--metrics-json", relative_for_command(metrics_json),
        "--metrics-by-scenario-csv", relative_for_command(scenario_csv),
    ]
    run_command(eval_cmd)

    analysis_cmd = [
        sys.executable,
        "evaluation/analyze_errors.py",
        "--scored-jsonl", relative_for_command(scored),
        "--output-dir", relative_for_command(error_analysis_dir),
        "--top-k", str(top_k),
    ]
    run_command(analysis_cmd)

    write_metric_comparison(
        current_metrics_path=metrics_json,
        compare_metric_paths=compare_metric_paths,
        comparison_json=comparison_json,
        comparison_csv=comparison_csv,
    )
    print(f"[done] base recheck outputs: {output_dir}")


if __name__ == "__main__":
    main()
