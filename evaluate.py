from __future__ import annotations

import argparse
import importlib
import json
import math
import re
import statistics as python_statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.container import BarContainer

from benchmarks import load_manifest, write_public_charts
from corpuslib import (
    LOG,
    config_value,
    configure_logging,
    final_summary,
    load_config,
    markdown_table,
    positive_int,
    resolved_path,
    utc_now,
    write_json,
    write_text,
)
from mlx_compat import gemma4_checkpoint_compatibility

matplotlib.use("Agg")

ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
TRAIN_METRIC = re.compile(
    r"Iter (?P<iteration>\d+): Train loss (?P<train_loss>[-+.\deE]+), "
    r"Learning Rate (?P<learning_rate>[-+.\deE]+), "
    r"It/sec (?P<iterations_per_second>[-+.\deE]+), "
    r"Tokens/sec (?P<tokens_per_second>[-+.\deE]+), "
    r"Trained Tokens (?P<trained_tokens>[-+.\deE]+), "
    r"Peak mem (?P<peak_memory_gb>[-+.\deE]+) GB"
)
VALIDATION_METRIC = re.compile(r"Iter (?P<iteration>\d+): Val loss (?P<validation_loss>[-+.\deE]+)")


@dataclass(frozen=True)
class EvaluationSettings:
    max_samples: int
    max_new_tokens: int
    max_seq_length: int
    warmup_runs: int
    timed_runs: int
    batch_size: int
    seed: int
    chart_dpi: int


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MLX adapters and publish a comparison report."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--runs-dir")
    parser.add_argument("--datasets-dir")
    parser.add_argument("--stats-dir")
    parser.add_argument("--report-dir")
    parser.add_argument("--max-samples", type=positive_int)
    parser.add_argument("--max-new-tokens", type=positive_int)
    parser.add_argument("--max-seq-length", type=positive_int)
    parser.add_argument("--warmup-runs", type=nonnegative_int)
    parser.add_argument("--timed-runs", type=positive_int)
    parser.add_argument("--batch-size", type=positive_int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--chart-dpi", type=positive_int)
    parser.add_argument("--force", action="store_true", help="Re-run already completed evaluations")
    parser.add_argument("--report-only", action="store_true", help="Never load MLX models")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def choose(cli_value: Any, config: Mapping[str, Any], key: str, default: Any) -> Any:
    return cli_value if cli_value is not None else config_value(config, key, default)


def make_settings(args: argparse.Namespace, config: Mapping[str, Any]) -> EvaluationSettings:
    settings = EvaluationSettings(
        max_samples=int(choose(args.max_samples, config, "evaluation.max_samples", 16)),
        max_new_tokens=int(choose(args.max_new_tokens, config, "evaluation.max_new_tokens", 32)),
        max_seq_length=int(choose(args.max_seq_length, config, "evaluation.max_seq_length", 1024)),
        warmup_runs=int(choose(args.warmup_runs, config, "evaluation.warmup_runs", 1)),
        timed_runs=int(choose(args.timed_runs, config, "evaluation.timed_runs", 3)),
        batch_size=int(choose(args.batch_size, config, "evaluation.batch_size", 1)),
        seed=int(choose(args.seed, config, "evaluation.seed", 42)),
        chart_dpi=int(choose(args.chart_dpi, config, "evaluation.chart_dpi", 160)),
    )
    if (
        any(
            value <= 0
            for value in (
                settings.max_samples,
                settings.max_new_tokens,
                settings.max_seq_length,
                settings.timed_runs,
                settings.batch_size,
                settings.chart_dpi,
            )
        )
        or settings.warmup_runs < 0
    ):
        raise SystemExit("evaluation counts and caps must be positive; warmup runs can be zero")
    return settings


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def parse_training_log(text: str) -> list[dict[str, int | float]]:
    reports: dict[int, dict[str, int | float]] = {}
    validation: dict[int, float] = {}
    for raw_line in text.splitlines():
        line = ANSI_ESCAPE.sub("", raw_line)
        val_match = VALIDATION_METRIC.search(line)
        if val_match:
            validation[int(val_match["iteration"])] = float(val_match["validation_loss"])
        match = TRAIN_METRIC.search(line)
        if not match:
            continue
        iteration = int(match["iteration"])
        reports[iteration] = {
            "iteration": iteration,
            "train_loss": float(match["train_loss"]),
            "learning_rate": float(match["learning_rate"]),
            "iterations_per_second": float(match["iterations_per_second"]),
            "tokens_per_second": float(match["tokens_per_second"]),
            "trained_tokens": int(float(match["trained_tokens"])),
            "peak_memory_gb": float(match["peak_memory_gb"]),
        }
    for iteration, value in validation.items():
        if iteration in reports:
            reports[iteration]["validation_loss"] = value
    return [reports[iteration] for iteration in sorted(reports)]


def display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def read_sweep(run_root: Path) -> list[dict[str, Any]]:
    plans: list[dict[str, Any]] = []
    if run_root.exists():
        for path in sorted(run_root.glob("*/plan.json")):
            plan = read_json(path)
            if plan and plan.get("mode", "selection") == "selection":
                plans.append(plan)
    if plans:
        return plans
    manifest = read_json(run_root / "sweep.json")
    if not manifest or not isinstance(manifest.get("runs"), list):
        return []
    return [
        item
        for item in manifest["runs"]
        if isinstance(item, dict) and item.get("mode", "selection") == "selection"
    ]


def read_valid_rows(path: Path, maximum: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        source = path.open(encoding="utf-8")
    except FileNotFoundError:
        return rows
    with source:
        for line in source:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and isinstance(row.get("text"), str):
                rows.append(row)
            if len(rows) >= maximum:
                break
    return rows


def next_line_cases(rows: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    cases: list[dict[str, str]] = []
    for row in rows:
        lines = str(row["text"]).splitlines(keepends=True)
        if len(lines) < 2:
            continue
        index = max(1, min(len(lines) - 1, math.floor(len(lines) * 0.75)))
        while index < len(lines) and not lines[index].strip():
            index += 1
        if index >= len(lines):
            continue
        target = lines[index].rstrip("\r\n")
        if not target:
            continue
        cases.append({"prompt": "".join(lines[:index]), "target": target})
    return cases


def vlm_perplexity(
    model: Any, processor: Any, texts: Sequence[str], max_seq_length: int
) -> tuple[float | None, int]:
    mx = importlib.import_module("mlx.core")
    trainer = importlib.import_module("mlx_vlm.trainer.sft_trainer")
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    loss_sum = 0.0
    token_count = 0
    for text in texts:
        tokens = tokenizer.encode(text)[:max_seq_length]
        if len(tokens) < 2:
            continue
        input_ids = mx.array([tokens])
        attention_mask = mx.ones_like(input_ids)
        loss = trainer.vision_language_loss_fn(
            model,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": None,
            },
        )
        mx.eval(loss)
        scored = len(tokens) - 1
        loss_sum += float(loss.item()) * scored
        token_count += scored
    if token_count == 0:
        return None, 0
    mean_loss = loss_sum / token_count
    return (math.exp(mean_loss) if mean_loss < 700 else None), token_count


def benchmark_condition(
    model_id: str,
    adapter_path: Path | None,
    rows: Sequence[dict[str, Any]],
    settings: EvaluationSettings,
) -> dict[str, Any]:
    mx = importlib.import_module("mlx.core")
    mlx_vlm = importlib.import_module("mlx_vlm")

    mx.random.seed(settings.seed)
    mx.reset_peak_memory()
    with gemma4_checkpoint_compatibility(model_id):
        model, processor = mlx_vlm.load(
            model_id,
            adapter_path=str(adapter_path) if adapter_path else None,
        )
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    cases = next_line_cases(rows)
    prompts: list[str] = []
    targets: list[str] = []
    prompt_limit = max(1, settings.max_seq_length - settings.max_new_tokens)
    for case in cases:
        tokens = tokenizer.encode(case["prompt"])[-prompt_limit:]
        if tokens:
            prompt = tokenizer.decode(tokens)
            config = model.config.__dict__
            prompts.append(
                mlx_vlm.apply_chat_template(
                    processor,
                    config,
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    num_images=0,
                    num_audios=0,
                )
            )
            targets.append(case["target"])
    if not prompts:
        fallback = tokenizer.encode(str(rows[0]["text"]))[-prompt_limit:]
        if fallback:
            prompts = [tokenizer.decode(fallback)]
            targets = [""]
    for _ in range(settings.warmup_runs):
        mlx_vlm.generate(
            model,
            processor,
            prompts[0],
            max_tokens=settings.max_new_tokens,
            verbose=False,
        )
    generation_tps: list[float] = []
    latest_texts: list[str] = []
    for _ in range(settings.timed_runs):
        latest_texts = []
        for prompt in prompts:
            response = mlx_vlm.generate(
                model,
                processor,
                prompt,
                max_tokens=settings.max_new_tokens,
                verbose=False,
            )
            latest_texts.append(response.text)
            measured_tps = float(response.generation_tps)
            if math.isfinite(measured_tps):
                generation_tps.append(measured_tps)
    if not generation_tps:
        raise RuntimeError("MLX-VLM returned no finite generation throughput measurement")
    exact_matches = 0
    compared = 0
    for generated, target in zip(latest_texts, targets, strict=True):
        if not target:
            continue
        generated_line = generated.splitlines()[0].rstrip() if generated.splitlines() else ""
        exact_matches += generated_line == target.rstrip()
        compared += 1
    ppl, scored_tokens = vlm_perplexity(
        model,
        processor,
        [str(row["text"]) for row in rows],
        settings.max_seq_length,
    )
    peak_memory = float(mx.get_peak_memory() / 1e9)
    del model
    del processor
    mx.clear_cache()
    return {
        "generation_tps": round(python_statistics.mean(generation_tps), 3),
        "generation_tps_stdev": round(python_statistics.stdev(generation_tps), 3)
        if len(generation_tps) > 1
        else 0.0,
        "perplexity": round(ppl, 4) if ppl is not None else None,
        "next_line_exact_match": round(exact_matches / compared, 6) if compared else None,
        "held_out_rows": len(rows),
        "completion_cases": compared,
        "perplexity_tokens": scored_tokens,
        "timed_runs": settings.timed_runs,
        "warmup_runs": settings.warmup_runs,
        "peak_memory_gb": round(peak_memory, 3),
    }


def comparison(base: Mapping[str, Any], tuned: Mapping[str, Any]) -> dict[str, float | None]:
    base_tps = base.get("generation_tps")
    tuned_tps = tuned.get("generation_tps")
    tps_change = None
    if isinstance(base_tps, (int, float)) and isinstance(tuned_tps, (int, float)) and base_tps:
        tps_change = round((float(tuned_tps) - float(base_tps)) / float(base_tps) * 100, 3)
    base_ppl = base.get("perplexity")
    tuned_ppl = tuned.get("perplexity")
    ppl_change = None
    if isinstance(base_ppl, (int, float)) and isinstance(tuned_ppl, (int, float)) and base_ppl:
        ppl_change = round((float(tuned_ppl) - float(base_ppl)) / float(base_ppl) * 100, 3)
    return {"tps_change_percent": tps_change, "perplexity_change_percent": ppl_change}


def pending_result(plan: Mapping[str, Any], reason: str) -> dict[str, Any]:
    return {
        "run_id": plan.get("run_id", "unknown"),
        "model": plan.get("model", {}),
        "variant": plan.get("variant", "unknown"),
        "status": "pending",
        "reason": reason,
        "sources": {
            "plan": str(Path(str(plan.get("run_path", ""))) / "plan.json"),
            "dataset": str(Path(str(plan.get("dataset_path", ""))) / "statistics.json"),
        },
    }


def file_fingerprint(path: Path) -> dict[str, Any] | None:
    try:
        status = path.stat()
    except OSError:
        return None
    return {"size": status.st_size, "mtime_ns": status.st_mtime_ns}


def evaluation_fingerprint(plan: Mapping[str, Any], settings: EvaluationSettings) -> dict[str, Any]:
    run_path = Path(str(plan.get("run_path", "")))
    dataset_path = Path(str(plan.get("dataset_path", "")))
    model = plan.get("model", {})
    return {
        "model_id": model.get("model_id") if isinstance(model, Mapping) else None,
        "adapter": file_fingerprint(run_path / "adapters/adapters.safetensors"),
        "held_out": file_fingerprint(dataset_path / "valid.jsonl"),
        "settings": {
            "max_samples": settings.max_samples,
            "max_new_tokens": settings.max_new_tokens,
            "max_seq_length": settings.max_seq_length,
            "warmup_runs": settings.warmup_runs,
            "timed_runs": settings.timed_runs,
            "batch_size": settings.batch_size,
            "seed": settings.seed,
        },
    }


def evaluate_plan(
    plan: Mapping[str, Any],
    settings: EvaluationSettings,
    report_only: bool,
) -> dict[str, Any]:
    run_path = Path(str(plan.get("run_path", "")))
    dataset_path = Path(str(plan.get("dataset_path", "")))
    model = plan.get("model", {})
    model_id = str(model.get("model_id", "")) if isinstance(model, Mapping) else ""
    if model_id.startswith("<"):
        return pending_result(plan, "Model ID is still a placeholder.")
    if not (run_path / "COMPLETED").is_file():
        return pending_result(plan, "Training has not completed.")
    adapter_path = run_path / "adapters"
    if not (adapter_path / "adapters.safetensors").is_file():
        return pending_result(plan, "Completion marker exists but adapter weights are missing.")
    rows = read_valid_rows(dataset_path / "valid.jsonl", settings.max_samples)
    if not rows:
        return pending_result(plan, "The repository-level valid split has no usable rows.")
    if report_only:
        return pending_result(plan, "Report-only mode did not load MLX models.")
    result: dict[str, Any] = {
        "run_id": plan.get("run_id", "unknown"),
        "model": model,
        "variant": plan.get("variant", "unknown"),
        "status": "running",
        "evaluated_at": utc_now(),
        "fingerprint": evaluation_fingerprint(plan, settings),
        "sources": {
            "plan": str(run_path / "plan.json"),
            "training": str(run_path / "run.json"),
            "adapter": str(adapter_path / "adapters.safetensors"),
            "held_out": str(dataset_path / "valid.jsonl"),
            "dataset": str(dataset_path / "statistics.json"),
        },
        "settings": {
            "max_samples": settings.max_samples,
            "max_new_tokens": settings.max_new_tokens,
            "max_seq_length": settings.max_seq_length,
            "warmup_runs": settings.warmup_runs,
            "timed_runs": settings.timed_runs,
            "batch_size": settings.batch_size,
            "seed": settings.seed,
        },
    }
    try:
        LOG.info("benchmarking base for %s", result["run_id"])
        result["base"] = benchmark_condition(model_id, None, rows, settings)
        LOG.info("benchmarking tuned adapter for %s", result["run_id"])
        result["tuned"] = benchmark_condition(model_id, adapter_path, rows, settings)
    except Exception as error:
        LOG.exception("evaluation failed for %s", result["run_id"])
        result["status"] = "error"
        result["reason"] = f"{type(error).__name__}: {error}"
        return result
    result["comparison"] = comparison(result["base"], result["tuned"])
    result["status"] = "completed"
    return result


def number(value: Any, decimals: int = 2, suffix: str = "") -> str:
    if not isinstance(value, (int, float)):
        return "pending"
    return f"{value:,.{decimals}f}{suffix}"


def load_dataset_statistics(dataset_root: Path, project_root: Path) -> list[dict[str, Any]]:
    statistics: list[dict[str, Any]] = []
    if not dataset_root.exists():
        return statistics
    for path in sorted(dataset_root.glob("*/statistics.json")):
        payload = read_json(path)
        if payload:
            payload["_source"] = display_path(path, project_root)
            statistics.append(payload)
    return statistics


def load_training_telemetry(run_root: Path, project_root: Path) -> list[dict[str, Any]]:
    telemetry: list[dict[str, Any]] = []
    for run_file in sorted(run_root.glob("*/run.json")):
        record = read_json(run_file)
        if not record:
            continue
        log_path = run_file.parent / "train.log"
        try:
            metrics = parse_training_log(log_path.read_text(encoding="utf-8"))
        except OSError:
            continue
        if not metrics:
            continue
        plan = read_json(run_file.parent / "plan.json") or {}
        telemetry.append(
            {
                "run_id": str(record.get("run_id", run_file.parent.name)),
                "mode": str(plan.get("mode", "selection")),
                "status": str(record.get("status", "unknown")),
                "model": record.get("model", plan.get("model", {})),
                "profile": record.get("profile", plan.get("profile", {})),
                "duration_seconds": record.get("duration_seconds"),
                "source": display_path(log_path, project_root),
                "metrics": metrics,
            }
        )
    return telemetry


def load_deployment_verifications(
    deployment_root: Path, project_root: Path
) -> list[dict[str, Any]]:
    verifications: list[dict[str, Any]] = []
    for evidence_path in sorted(deployment_root.glob("*/verification.json")):
        evidence = read_json(evidence_path)
        if not evidence:
            continue
        smoke = evidence.get("smoke", {})
        runtime = smoke.get("runtime", {}) if isinstance(smoke, Mapping) else {}
        tool_call = smoke.get("tool_call", {}) if isinstance(smoke, Mapping) else {}
        codex = evidence.get("codex", {})
        thresholds = evidence.get("performance_thresholds", {})
        verifications.append(
            {
                "tag": evidence.get("tag", evidence_path.parent.name),
                "status": evidence.get("status", "unknown"),
                "generated_at": evidence.get("generated_at"),
                "decode_tps": runtime.get("decode_tps") if isinstance(runtime, Mapping) else None,
                "ttft_seconds": runtime.get("ttft_seconds")
                if isinstance(runtime, Mapping)
                else None,
                "minimum_decode_tps": thresholds.get("minimum_decode_tps")
                if isinstance(thresholds, Mapping)
                else None,
                "maximum_ttft_seconds": thresholds.get("maximum_ttft_seconds")
                if isinstance(thresholds, Mapping)
                else None,
                "native_tool_call": tool_call.get("passed")
                if isinstance(tool_call, Mapping)
                else None,
                "codex_passed": codex.get("passed") if isinstance(codex, Mapping) else None,
                "source": display_path(evidence_path, project_root),
            }
        )
    return verifications


def make_sources_portable(evaluations: Sequence[dict[str, Any]], project_root: Path) -> None:
    for evaluation in evaluations:
        sources = evaluation.get("sources")
        if not isinstance(sources, dict):
            continue
        for key, value in sources.items():
            if isinstance(value, str):
                sources[key] = display_path(Path(value), project_root)


def placeholder_chart(path: Path, title: str, message: str, dpi: int) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5), layout="constrained")
    axis.set_title(title)
    axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes)
    axis.set_axis_off()
    save_chart(figure, path, dpi)
    plt.close(figure)


def save_chart(figure: Any, path: Path, dpi: int) -> None:
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    figure.savefig(path.with_suffix(".svg"), bbox_inches="tight")


def grouped_chart(
    path: Path,
    title: str,
    ylabel: str,
    labels: Sequence[str],
    first: Sequence[float],
    second: Sequence[float],
    first_label: str,
    second_label: str,
    dpi: int,
) -> None:
    if not labels:
        placeholder_chart(path, title, "Pending completed evaluations", dpi)
        return
    figure, axis = plt.subplots(figsize=(max(8, len(labels) * 1.6), 4.8), layout="constrained")
    sns.barplot(
        x=[label for label in labels for _ in range(2)],
        y=[value for pair in zip(first, second, strict=True) for value in pair],
        hue=[value for _ in labels for value in (first_label, second_label)],
        order=list(labels),
        hue_order=[first_label, second_label],
        errorbar=None,
        ax=axis,
    )
    axis.set_title(title)
    axis.set_xlabel("")
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=25)
    legend = axis.get_legend()
    if legend is not None:
        legend.set_title("")
    save_chart(figure, path, dpi)
    plt.close(figure)


def write_training_chart(
    path: Path,
    telemetry: Sequence[Mapping[str, Any]],
    dpi: int,
) -> None:
    points: list[tuple[str, int, float, float, float]] = []
    for run in telemetry:
        metrics = run.get("metrics", [])
        if not isinstance(metrics, list):
            continue
        label = f"{str(run.get('mode', 'run')).title()} · {str(run.get('status', 'unknown'))}"
        for point in metrics:
            if not isinstance(point, Mapping):
                continue
            iteration = point.get("iteration")
            loss = point.get("train_loss")
            tokens_per_second = point.get("tokens_per_second")
            peak_memory = point.get("peak_memory_gb")
            if not isinstance(iteration, (int, float)):
                continue
            if not isinstance(loss, (int, float)):
                continue
            if not isinstance(tokens_per_second, (int, float)):
                continue
            if not isinstance(peak_memory, (int, float)):
                continue
            points.append(
                (
                    label,
                    int(iteration),
                    float(loss),
                    float(tokens_per_second),
                    float(peak_memory),
                )
            )
    if not points:
        placeholder_chart(path, "Training telemetry", "Pending completed training", dpi)
        return
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    labels = [point[0] for point in points]
    iterations = [point[1] for point in points]
    for axis, values, title, ylabel in (
        (axes[0], [point[2] for point in points], "Optimization", "completion loss"),
        (axes[1], [point[3] for point in points], "Throughput", "training tokens / second"),
        (axes[2], [point[4] for point in points], "Memory", "peak unified memory (GB)"),
    ):
        sns.lineplot(
            x=iterations,
            y=values,
            hue=labels,
            marker="o",
            errorbar=None,
            estimator=None,
            ax=axis,
        )
        axis.set_title(title, loc="left", weight="bold")
        axis.set_xlabel("iteration")
        axis.set_ylabel(ylabel)
        legend = axis.get_legend()
        if legend is not None:
            legend.set_title("")
            legend.set_frame_on(False)
    figure.suptitle("Gemma 4 E4B training telemetry", x=0.01, ha="left", weight="bold")
    save_chart(figure, path, dpi)
    plt.close(figure)


def write_deployment_chart(
    path: Path,
    verifications: Sequence[Mapping[str, Any]],
    dpi: int,
) -> None:
    rows: list[tuple[str, str, float]] = []
    for verification in verifications:
        tag = str(verification.get("tag", "deployment"))
        decode_tps = verification.get("decode_tps")
        minimum_tps = verification.get("minimum_decode_tps")
        ttft = verification.get("ttft_seconds")
        maximum_ttft = verification.get("maximum_ttft_seconds")
        if (
            isinstance(decode_tps, (int, float))
            and decode_tps > 0
            and isinstance(minimum_tps, (int, float))
            and minimum_tps > 0
        ):
            rows.append((tag, "Decode throughput", float(decode_tps) / float(minimum_tps)))
        if (
            isinstance(ttft, (int, float))
            and ttft > 0
            and isinstance(maximum_ttft, (int, float))
            and maximum_ttft > 0
        ):
            rows.append((tag, "Time to first token", float(maximum_ttft) / float(ttft)))
    if not rows:
        placeholder_chart(path, "Offline deployment gate", "Pending Ollama verification", dpi)
        return
    figure, axis = plt.subplots(figsize=(9, 4.8), layout="constrained")
    sns.barplot(
        x=[row[2] for row in rows],
        y=[row[1] for row in rows],
        hue=[row[0] for row in rows],
        errorbar=None,
        ax=axis,
    )
    axis.axvline(1, color="#343A40", linestyle="--", linewidth=1.2, label="Pass boundary")
    for container in axis.containers:
        if isinstance(container, BarContainer):
            axis.bar_label(container, fmt="%.1f×", padding=4)
    axis.set_title("Offline deployment headroom", loc="left", weight="bold")
    axis.set_xlabel("threshold-normalized pass margin (higher is better)")
    axis.set_ylabel("")
    legend = axis.get_legend()
    if legend is not None:
        legend.set_title("")
        legend.set_frame_on(False)
    save_chart(figure, path, dpi)
    plt.close(figure)


def write_charts(
    report_root: Path,
    evaluations: Sequence[Mapping[str, Any]],
    dataset_statistics: Sequence[Mapping[str, Any]],
    dpi: int,
    training_telemetry: Sequence[Mapping[str, Any]] = (),
    deployment_verifications: Sequence[Mapping[str, Any]] = (),
) -> None:
    sns.set_theme(
        context="paper",
        style="ticks",
        palette="colorblind",
        font_scale=1.15,
        rc={"axes.titlepad": 12, "figure.facecolor": "white", "axes.facecolor": "white"},
    )
    chart_root = report_root / "charts"
    chart_root.mkdir(parents=True, exist_ok=True)
    for stale in (
        "edge-throughput-q4-vs-q5.png",
        "edge-throughput-q4-vs-q5.svg",
        "edge-memory-q4-vs-q5.png",
        "edge-memory-q4-vs-q5.svg",
    ):
        (chart_root / stale).unlink(missing_ok=True)
    completed = [item for item in evaluations if item.get("status") == "completed"]
    labels = [f"{item['model'].get('name', 'model')}\n{item['variant']}" for item in completed]
    base_tps = [float(item["base"]["generation_tps"]) for item in completed]
    tuned_tps = [float(item["tuned"]["generation_tps"]) for item in completed]
    grouped_chart(
        chart_root / "tps-base-vs-tuned.png",
        "Generation throughput: base vs tuned",
        "tokens / second",
        labels,
        base_tps,
        tuned_tps,
        "Base",
        "Tuned",
        dpi,
    )
    if labels:
        figure, axis = plt.subplots(
            figsize=(max(8, len(labels) * 1.4), 4.8),
            layout="constrained",
        )
        sns.barplot(x=labels, y=tuned_tps, errorbar=None, ax=axis)
        axis.set_title("Tuned generation TPS by model and dataset")
        axis.set_xlabel("")
        axis.set_ylabel("tokens / second")
        axis.tick_params(axis="x", rotation=25)
        save_chart(figure, chart_root / "tps-by-model.png", dpi)
        plt.close(figure)
    else:
        placeholder_chart(
            chart_root / "tps-by-model.png",
            "Tuned generation TPS by model and dataset",
            "Pending completed evaluations",
            dpi,
        )
    ppl_items = [
        item
        for item in completed
        if item["base"].get("perplexity") is not None
        and item["tuned"].get("perplexity") is not None
    ]
    grouped_chart(
        chart_root / "perplexity-by-model.png",
        "Held-out perplexity: base vs tuned",
        "perplexity (lower is better)",
        [f"{item['model'].get('name', 'model')}\n{item['variant']}" for item in ppl_items],
        [float(item["base"]["perplexity"]) for item in ppl_items],
        [float(item["tuned"]["perplexity"]) for item in ppl_items],
        "Base",
        "Tuned",
        dpi,
    )
    dataset_labels = [str(item.get("variant", "unknown")) for item in dataset_statistics]
    dataset_rows = [int(item.get("rows", {}).get("total", 0)) for item in dataset_statistics]
    if dataset_labels and any(dataset_rows):
        figure, axis = plt.subplots(figsize=(8, 4.8), layout="constrained")
        sns.barplot(x=dataset_labels, y=dataset_rows, errorbar=None, ax=axis)
        axis.set_title("Dataset rows after filtering and deduplication")
        axis.set_xlabel("")
        axis.set_ylabel("rows")
        save_chart(figure, chart_root / "dataset-sizes.png", dpi)
        plt.close(figure)
    else:
        placeholder_chart(
            chart_root / "dataset-sizes.png",
            "Dataset rows after filtering and deduplication",
            "No retained dataset rows" if dataset_labels else "Pending built datasets",
            dpi,
        )
    raw_stats = next(
        (item for item in dataset_statistics if item.get("variant") == "raw-max"),
        None,
    )
    languages = raw_stats.get("by_language", {}) if isinstance(raw_stats, Mapping) else {}
    if isinstance(languages, Mapping) and languages:
        ordered = sorted(languages, key=lambda name: int(languages[name]), reverse=True)[:15]
        figure, axis = plt.subplots(figsize=(10, 5.4), layout="constrained")
        sns.barplot(
            x=[int(languages[name]) for name in ordered],
            y=ordered,
            color="#2A9D8F",
            errorbar=None,
            ax=axis,
        )
        axis.set_title("Retained corpus by language", loc="left", weight="bold")
        axis.set_xlabel("deduplicated chunks")
        axis.set_ylabel("")
        save_chart(figure, chart_root / "corpus-language-distribution.png", dpi)
        plt.close(figure)
    else:
        placeholder_chart(
            chart_root / "corpus-language-distribution.png",
            "Retained corpus by language",
            "Pending full corpus build",
            dpi,
        )
    runtime_rows = []
    for item in completed:
        for condition in ("base", "tuned"):
            values = item.get(condition, {})
            if not isinstance(values, Mapping):
                continue
            tps = values.get("generation_tps")
            memory = values.get("peak_memory_gb")
            if isinstance(tps, (int, float)) and isinstance(memory, (int, float)):
                runtime_rows.append((condition.title(), float(tps), float(memory)))
    if runtime_rows:
        figure, axis = plt.subplots(figsize=(8.5, 5), layout="constrained")
        sns.scatterplot(
            x=[row[1] for row in runtime_rows],
            y=[row[2] for row in runtime_rows],
            hue=[row[0] for row in runtime_rows],
            style=[row[0] for row in runtime_rows],
            s=150,
            ax=axis,
        )
        axis.set_title("Local runtime profile", loc="left", weight="bold")
        axis.set_xlabel("generation tokens / second")
        axis.set_ylabel("peak unified memory (GB)")
        axis.legend(title="", frameon=False)
        save_chart(figure, chart_root / "local-runtime-profile.png", dpi)
        plt.close(figure)
    else:
        placeholder_chart(
            chart_root / "local-runtime-profile.png",
            "Local runtime profile",
            "Pending completed E4B evaluation",
            dpi,
        )
    write_training_chart(chart_root / "training-telemetry.png", training_telemetry, dpi)
    write_deployment_chart(
        chart_root / "offline-deployment-headroom.png",
        deployment_verifications,
        dpi,
    )


def report_markdown(
    results: Mapping[str, Any],
    dataset_statistics: Sequence[Mapping[str, Any]],
    personal_stats: Mapping[str, Any] | None,
) -> str:
    evaluations = list(results.get("evaluations", []))
    completed = sum(item.get("status") == "completed" for item in evaluations)
    pending = sum(item.get("status") == "pending" for item in evaluations)
    errors = sum(item.get("status") == "error" for item in evaluations)
    training_telemetry = results.get("training_telemetry", [])
    if not isinstance(training_telemetry, list):
        training_telemetry = []
    deployment_verifications = results.get("deployment_verifications", [])
    if not isinstance(deployment_verifications, list):
        deployment_verifications = []
    dataset_rows: list[list[Any]] = []
    for item in dataset_statistics:
        rows = item.get("rows", {})
        source = str(item.get("_source", "datasets/unknown/statistics.json"))
        dataset_rows.append(
            [
                item.get("variant", "unknown"),
                f"{int(rows.get('total', 0)):,}",
                f"{int(rows.get('train', 0)):,}",
                f"{int(rows.get('valid', 0)):,}",
                f"{int(item.get('tokens', {}).get('total', 0)):,}",
                len(item.get("by_language", {})),
                len(item.get("by_repo", {})),
                f"[{Path(source).name}](../{source})",
            ]
        )
    evaluation_rows: list[list[Any]] = []
    for index, item in enumerate(evaluations):
        base = item.get("base", {})
        tuned = item.get("tuned", {})
        compared = item.get("comparison", {})
        model = item.get("model", {})
        evaluation_rows.append(
            [
                model.get("name", "unknown") if isinstance(model, Mapping) else "unknown",
                item.get("variant", "unknown"),
                item.get("status", "pending"),
                number(base.get("generation_tps")),
                number(tuned.get("generation_tps")),
                number(compared.get("tps_change_percent"), suffix="%"),
                number(base.get("perplexity")),
                number(tuned.get("perplexity")),
                number(tuned.get("next_line_exact_match"), decimals=3),
                f"[results.json](results.json) → evaluations[{index}]",
            ]
        )
    if personal_stats:
        overall = personal_stats.get("overall", {})
        stats_block = markdown_table(
            ["Commits", "Added", "Removed", "Net LOC", "Source"],
            [
                [
                    f"{int(overall.get('commits', 0)):,}",
                    f"{int(overall.get('additions', 0)):,}",
                    f"{int(overall.get('deletions', 0)):,}",
                    f"{int(overall.get('net', 0)):,}",
                    "[stats.json](../stats/stats.json) → overall",
                ]
            ],
        )
    else:
        stats_block = "Pending `stats/stats.json`."
    dataset_block = (
        markdown_table(
            ["Variant", "Rows", "Train", "Valid", "Lexical tokens", "Languages", "Repos", "Source"],
            dataset_rows,
        )
        if dataset_rows
        else "Pending built datasets."
    )
    evaluation_block = (
        markdown_table(
            [
                "Model",
                "Dataset",
                "Status",
                "Base TPS",
                "Tuned TPS",
                "TPS Δ",
                "Base PPL",
                "Tuned PPL",
                "Tuned exact",
                "Source",
            ],
            evaluation_rows,
        )
        if evaluation_rows
        else "Pending prepared sweep entries."
    )
    training_rows: list[list[Any]] = []
    for run in training_telemetry:
        metrics = run.get("metrics", []) if isinstance(run, Mapping) else []
        if not isinstance(metrics, list) or not metrics:
            continue
        last = metrics[-1]
        throughput = [
            float(point["tokens_per_second"])
            for point in metrics
            if isinstance(point, Mapping)
            and isinstance(point.get("tokens_per_second"), (int, float))
        ]
        memory = [
            float(point["peak_memory_gb"])
            for point in metrics
            if isinstance(point, Mapping) and isinstance(point.get("peak_memory_gb"), (int, float))
        ]
        source = str(run.get("source", "runs/unknown/train.log"))
        training_rows.append(
            [
                run.get("mode", "unknown"),
                run.get("status", "unknown"),
                len(metrics),
                number(last.get("train_loss") if isinstance(last, Mapping) else None, 4),
                number(python_statistics.median(throughput) if throughput else None, 1),
                number(max(memory) if memory else None, 2),
                f"[{Path(source).name}](../{source})",
            ]
        )
    training_block = (
        markdown_table(
            ["Mode", "Status", "Reports", "Final loss", "Median tok/s", "Peak GB", "Source"],
            training_rows,
        )
        if training_rows
        else "Pending training telemetry."
    )
    deployment_rows: list[list[Any]] = []
    for verification in deployment_verifications:
        if not isinstance(verification, Mapping):
            continue
        source = str(verification.get("source", "deployment/verification.json"))
        deployment_rows.append(
            [
                verification.get("tag", "unknown"),
                verification.get("status", "unknown"),
                number(verification.get("decode_tps"), 1),
                number(verification.get("ttft_seconds"), 3),
                "pass" if verification.get("native_tool_call") is True else "fail",
                "pass" if verification.get("codex_passed") is True else "fail",
                f"[{Path(source).name}](../{source})",
            ]
        )
    deployment_block = (
        markdown_table(
            ["Ollama tag", "Status", "Decode tok/s", "TTFT (s)", "Tool call", "Codex", "Source"],
            deployment_rows,
        )
        if deployment_rows
        else "Pending Ollama and Codex verification."
    )
    return f"""# Personalized Gemma 4 E4B engineering results

Generated: {results["generated_at"]}  
Canonical evaluation data: [`results.json`](results.json)

## Status

{completed} completed, {pending} pending, {errors} errored evaluations. Pending values are never
replaced with estimates.

## Published upstream context

These charts contain upstream, unadapted release results. They are contextual public benchmarks,
not measurements of the personalized adapter and not one locally reproduced leaderboard.

![Gemma benchmark profile](charts/upstream-gemma4-benchmarks.png)

![Gemma Codeforces rating](charts/upstream-gemma4-codeforces.png)

![Published LiveCodeBench v6 comparison](charts/upstream-livecodebench-v6.png)

Canonical rows and source URLs: [`public-benchmarks.json`](../benchmarks/public-benchmarks.json).
Vendor and evaluator attribution remains attached to every observation; unrelated metrics are not
averaged into an artificial overall score.

## Personal contribution statistics

{stats_block}

## Corpus

{dataset_block}

![Dataset sizes](charts/dataset-sizes.png)

![Retained corpus by language](charts/corpus-language-distribution.png)

Chart source: each listed `datasets/<variant>/statistics.json`.

## Local base-versus-personalized evaluation

{evaluation_block}

![TPS by model](charts/tps-by-model.png)

![Base versus tuned TPS](charts/tps-base-vs-tuned.png)

![Perplexity by model](charts/perplexity-by-model.png)


![Local runtime profile](charts/local-runtime-profile.png)

Chart sources: `results.json → evaluations[*].base`, `.tuned`, and `.comparison`.

## Training telemetry

{training_block}

![Training telemetry](charts/training-telemetry.png)

Chart source: parsed MLX-VLM training reports in the linked `train.log` files.

## Offline Ollama and Codex deployment

{deployment_block}

![Offline deployment headroom](charts/offline-deployment-headroom.png)

The normalized chart uses `observed decode TPS / minimum TPS` and
`maximum TTFT / observed TTFT`; values above 1× pass. Native tool use and the Codex result remain
separate binary gates in the evidence table.

## Methodology

Configuration source: [`config.yaml`](../config.yaml).

Repositories are walked at their checked-out state, filtered by `config.yaml`, chunked to an
approximate lexical-token target, exact-deduplicated by SHA-256, and near-deduplicated by MinHash.
High-confidence credential paths and token signatures are discarded before chunking. The
deterministic 90/10 target split is assigned by repository, with at least three validation
repositories when the corpus size permits, so no repository appears in both train and valid.
Dataset parameters and actual split assignments are recorded in `statistics.json`.

Training uses one pinned instruction-tuned Gemma 4 E4B checkpoint and MLX-VLM QLoRA on Apple
Silicon. Vision and audio towers remain frozen; rows contain text-only code prefix/completion pairs
and loss is masked to the completion. A selection adapter is evaluated on untouched repositories.
Only after configuration is locked is a separate production adapter trained for one epoch on the
merged safe corpus. Run inputs, revisions, logs, adapter paths, and wall times are recorded under
`runs/<run>/`.

Evaluation samples only `valid.jsonl`. TPS is MLX-VLM's measured generation throughput after the
configured warm-up runs. Perplexity is exponentiated mean next-token cross-entropy on held-out
tokens. Next-line exact match compares the first generated line with a withheld source line. The
exact limits, sample counts, scored tokens, repeated-run dispersion, and peak memory are stored in
`results.json` for every completed condition.

## Limitations

- This is a repo-walked corpus, not a curated set of accepted or known-correct solutions.
- Personal data is small and correlated; results should not be generalized to broad coding ability.
- LoRA adapts a subset of weights and is not a full-model fine-tune.
- Throughput and memory are Apple-Silicon-only numbers from the local machine and software stack.
- Public scores are vendor/model-card results and may differ in harness, sampling, or tool settings.
- Repository-level splitting prevents direct repository leakage but cannot remove shared concepts
  or copied code that survives near-deduplication.
- Secret filtering is intentionally high precision, not a proof that private source is publishable.
- Source-repository licenses continue to govern retained code; dataset metadata does not override
  them.
"""


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)
    config, root = load_config(args.config)
    settings = make_settings(args, config)
    run_root = resolved_path(root, args.runs_dir or config_value(config, "paths.runs", "runs"))
    dataset_root = resolved_path(
        root,
        args.datasets_dir or config_value(config, "paths.datasets", "datasets"),
    )
    stats_root = resolved_path(root, args.stats_dir or config_value(config, "paths.stats", "stats"))
    report_root = resolved_path(
        root, args.report_dir or config_value(config, "paths.report", "report")
    )
    deployment_root = resolved_path(root, config_value(config, "paths.deployment", "deployment"))
    report_root.mkdir(parents=True, exist_ok=True)
    plans = read_sweep(run_root)
    previous = read_json(report_root / "results.json") or {}
    previous_by_run = {
        item.get("run_id"): item
        for item in previous.get("evaluations", [])
        if isinstance(item, dict) and item.get("status") == "completed"
    }
    evaluations: list[dict[str, Any]] = []
    for plan in plans:
        run_id = plan.get("run_id")
        fingerprint = evaluation_fingerprint(plan, settings)
        if (
            not args.force
            and run_id in previous_by_run
            and previous_by_run[run_id].get("fingerprint") == fingerprint
        ):
            evaluations.append(previous_by_run[run_id])
            continue
        evaluations.append(evaluate_plan(plan, settings, args.report_only))
    make_sources_portable(evaluations, root)
    dataset_statistics = load_dataset_statistics(dataset_root, root)
    benchmark_path = resolved_path(
        root,
        config_value(config, "benchmarks.data", "benchmarks/public-benchmarks.json"),
    )
    source_files = [
        "config.yaml",
        display_path(run_root / "sweep.json", root),
        display_path(benchmark_path, root),
    ]
    source_files.extend(str(item.get("_source")) for item in dataset_statistics)
    personal_stats = read_json(stats_root / "stats.json")
    if personal_stats:
        source_files.append(display_path(stats_root / "stats.json", root))
    training_telemetry = load_training_telemetry(run_root, root)
    deployment_verifications = load_deployment_verifications(deployment_root, root)
    source_files.extend(str(item["source"]) for item in training_telemetry)
    source_files.extend(str(item["source"]) for item in deployment_verifications)
    results = {
        "generated_at": utc_now(),
        "status": "complete"
        if evaluations and all(item["status"] == "completed" for item in evaluations)
        else "partial",
        "source_files": source_files,
        "settings": {
            "max_samples": settings.max_samples,
            "max_new_tokens": settings.max_new_tokens,
            "max_seq_length": settings.max_seq_length,
            "warmup_runs": settings.warmup_runs,
            "timed_runs": settings.timed_runs,
            "batch_size": settings.batch_size,
            "seed": settings.seed,
        },
        "evaluations": evaluations,
        "training_telemetry": training_telemetry,
        "deployment_verifications": deployment_verifications,
    }
    write_json(report_root / "results.json", results)
    write_charts(
        report_root,
        evaluations,
        dataset_statistics,
        settings.chart_dpi,
        training_telemetry,
        deployment_verifications,
    )
    benchmark_manifest = load_manifest(benchmark_path)
    write_public_charts(
        benchmark_manifest,
        report_root / "charts",
        settings.chart_dpi,
    )
    write_text(
        report_root / "REPORT.md",
        report_markdown(results, dataset_statistics, personal_stats),
    )
    completed = sum(item["status"] == "completed" for item in evaluations)
    pending = sum(item["status"] == "pending" for item in evaluations)
    errors = sum(item["status"] == "error" for item in evaluations)
    final_summary(
        "Evaluation and report",
        [
            ("Completed evaluations", completed),
            ("Pending evaluations", pending),
            ("Errored evaluations", errors),
            ("Report", report_root / "REPORT.md"),
            ("Results", report_root / "results.json"),
        ],
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
