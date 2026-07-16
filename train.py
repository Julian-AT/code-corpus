from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from corpuslib import (
    LOG,
    config_value,
    configure_logging,
    final_summary,
    load_config,
    positive_int,
    resolved_path,
    safe_name,
    utc_now,
    write_json,
    write_text,
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    model_id: str
    note: str
    trainer: str = "mlx-vlm"
    quantization_bits: int = 4
    optimization: str = "MLX native"
    target_device: str = "Apple Silicon edge"
    upstream_model_id: str = ""
    revision: str = ""
    ollama_base: str = ""

    @property
    def placeholder(self) -> bool:
        return self.model_id.startswith("<") and self.model_id.endswith(">")


@dataclass(frozen=True)
class TrainingProfile:
    batch_size: int
    iterations: int | None
    max_seq_length: int
    learning_rate: float
    gradient_checkpointing: bool
    gradient_accumulation_steps: int
    lora_rank: int
    lora_alpha: float
    lora_dropout: float
    prompt_fraction: float
    validation_batches: int
    steps_per_report: int
    steps_per_eval: int
    save_every: int
    seed: int
    epochs: int | None = None


@dataclass(frozen=True)
class SweepRun:
    run_id: str
    model: ModelSpec
    variant: str
    dataset_path: Path
    run_path: Path
    adapter_path: Path
    profile: TrainingProfile
    mode: str = "selection"

    @property
    def training_data_path(self) -> Path:
        return self.run_path / "mlx-data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare or execute a sequential MLX-VLM LoRA sweep."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--execute", action="store_true", help="Run prepared jobs; default is prepare only"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--smoke", action="store_true", help="Run the configured model for smoke iterations"
    )
    mode.add_argument(
        "--production",
        action="store_true",
        help="Merge train and valid data and train the final all-data adapter",
    )
    parser.add_argument("--models", help="Comma-separated configured model names")
    parser.add_argument("--variants", help="Comma-separated configured dataset variants")
    parser.add_argument("--dataset-dir")
    parser.add_argument("--runs-dir")
    parser.add_argument("--batch-size", type=positive_int)
    parser.add_argument("--iterations", type=positive_int)
    parser.add_argument("--epochs", type=positive_int)
    parser.add_argument("--max-seq-length", type=positive_int)
    parser.add_argument("--learning-rate", type=positive_float)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--gradient-accumulation-steps", type=positive_int)
    parser.add_argument("--lora-rank", type=positive_int)
    parser.add_argument("--lora-alpha", type=positive_float)
    parser.add_argument("--lora-dropout", type=unit_interval)
    parser.add_argument("--prompt-fraction", type=open_unit_interval)
    parser.add_argument("--validation-batches", type=positive_int)
    parser.add_argument("--steps-per-report", type=positive_int)
    parser.add_argument("--steps-per-eval", type=positive_int)
    parser.add_argument("--save-every", type=positive_int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def unit_interval(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("must be between zero and one")
    return parsed


def open_unit_interval(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed < 1:
        raise argparse.ArgumentTypeError("must be greater than zero and less than one")
    return parsed


def choose(cli_value: Any, config: Mapping[str, Any], key: str, default: Any) -> Any:
    return cli_value if cli_value is not None else config_value(config, key, default)


def parse_models(config: Mapping[str, Any]) -> list[ModelSpec]:
    raw_models = config_value(config, "training.models", [])
    if not isinstance(raw_models, list):
        raise SystemExit("training.models must be a list")
    models: list[ModelSpec] = []
    for raw in raw_models:
        if not isinstance(raw, Mapping):
            raise SystemExit("each training model must be a mapping")
        raw_name = str(raw.get("name", "")).strip()
        model_id = str(raw.get("model_id", "")).strip()
        if not raw_name or not model_id:
            raise SystemExit("each training model needs name and model_id")
        model = ModelSpec(
            name=safe_name(raw_name),
            family=str(raw.get("family", "")),
            model_id=model_id,
            note=str(raw.get("note", "")),
            trainer=str(raw.get("trainer", "mlx-vlm")),
            quantization_bits=int(raw.get("quantization_bits", 4)),
            optimization=str(raw.get("optimization", "MLX native")),
            target_device=str(raw.get("target_device", "Apple Silicon edge")),
            upstream_model_id=str(raw.get("upstream_model_id", "")),
            revision=str(raw.get("revision", "")),
            ollama_base=str(raw.get("ollama_base", "")),
        )
        if model.trainer != "mlx-vlm":
            raise SystemExit(f"Unsupported trainer for {model.name}: {model.trainer}")
        if model.quantization_bits not in {4, 5}:
            raise SystemExit(
                f"Unsupported quantization for {model.name}: Q{model.quantization_bits}"
            )
        models.append(model)
    names = [model.name for model in models]
    if len(names) != len(set(names)):
        raise SystemExit("training model names must be unique after filename normalization")
    return models


def selected_names(value: str | None) -> set[str] | None:
    if value is None:
        return None
    return {safe_name(item.strip()) for item in value.split(",") if item.strip()}


def make_profile(args: argparse.Namespace, config: Mapping[str, Any]) -> TrainingProfile:
    if args.smoke:
        iterations = int(
            choose(args.iterations, config, "training.smoke_iterations", 1)
        )
        epochs = None
    elif args.iterations is not None:
        iterations = args.iterations
        epochs = None
    else:
        configured_epochs = choose(args.epochs, config, "training.epochs", None)
        epochs = int(configured_epochs) if configured_epochs is not None else None
        iterations = (
            None
            if epochs is not None
            else int(config_value(config, "training.iterations", 600))
        )
    configured_save_every = int(choose(args.save_every, config, "training.save_every", 100))
    learning_rate = float(choose(args.learning_rate, config, "training.learning_rate", 1e-5))
    if learning_rate <= 0:
        raise SystemExit("training.learning_rate must be greater than zero")
    profile = TrainingProfile(
        batch_size=int(choose(args.batch_size, config, "training.batch_size", 1)),
        iterations=iterations,
        max_seq_length=int(choose(args.max_seq_length, config, "training.max_seq_length", 1024)),
        learning_rate=learning_rate,
        gradient_checkpointing=bool(
            choose(args.gradient_checkpointing, config, "training.gradient_checkpointing", True)
        ),
        gradient_accumulation_steps=int(
            args.gradient_accumulation_steps
            if args.gradient_accumulation_steps is not None
            else 1
            if args.smoke
            else config_value(config, "training.gradient_accumulation_steps", 1)
        ),
        lora_rank=int(choose(args.lora_rank, config, "training.lora_rank", 8)),
        lora_alpha=float(choose(args.lora_alpha, config, "training.lora_alpha", 16)),
        lora_dropout=float(choose(args.lora_dropout, config, "training.lora_dropout", 0.0)),
        prompt_fraction=float(
            choose(args.prompt_fraction, config, "training.prompt_fraction", 0.5)
        ),
        validation_batches=int(
            choose(args.validation_batches, config, "training.validation_batches", 10)
        ),
        steps_per_report=int(
            choose(args.steps_per_report, config, "training.steps_per_report", 10)
        ),
        steps_per_eval=int(choose(args.steps_per_eval, config, "training.steps_per_eval", 100)),
        save_every=(
            min(configured_save_every, iterations)
            if iterations is not None
            else configured_save_every
        ),
        seed=int(choose(args.seed, config, "training.seed", 42)),
        epochs=epochs,
    )
    if any(
        value <= 0
        for value in (
            profile.batch_size,
            profile.max_seq_length,
            profile.gradient_accumulation_steps,
            profile.lora_rank,
            profile.validation_batches,
            profile.steps_per_report,
            profile.steps_per_eval,
            profile.save_every,
        )
    ):
        raise SystemExit("training counts and caps must be greater than zero")
    if profile.iterations is not None and profile.iterations <= 0:
        raise SystemExit("training.iterations must be greater than zero")
    if profile.epochs is not None and profile.epochs <= 0:
        raise SystemExit("training.epochs must be greater than zero")
    if profile.iterations is None and profile.epochs is None:
        raise SystemExit("training requires either iterations or epochs")
    if profile.lora_alpha <= 0:
        raise SystemExit("training.lora_alpha must be greater than zero")
    if not 0 <= profile.lora_dropout <= 1:
        raise SystemExit("training.lora_dropout must be between zero and one")
    if not 0 < profile.prompt_fraction < 1:
        raise SystemExit("training.prompt_fraction must be greater than zero and less than one")
    return profile


def make_sweep(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    root: Path,
) -> list[SweepRun]:
    models = parse_models(config)
    requested_models = selected_names(args.models)
    if requested_models is not None:
        models = [model for model in models if model.name in requested_models]
        missing = requested_models - {model.name for model in models}
        if missing:
            raise SystemExit(f"Unknown model names: {', '.join(sorted(missing))}")
    variants = [str(value) for value in config_value(config, "training.variants", [])]
    if len(variants) != len(set(variants)):
        raise SystemExit("training variants must be unique")
    if args.variants:
        requested_variants = [item.strip() for item in args.variants.split(",") if item.strip()]
        missing_variants = set(requested_variants) - set(variants)
        if missing_variants:
            raise SystemExit(f"Unknown variants: {', '.join(sorted(missing_variants))}")
        variants = requested_variants
    if args.smoke:
        models = models[:1]
        variants = variants[:1]
    if not models or not variants:
        raise SystemExit("Sweep selection is empty")
    dataset_root = resolved_path(
        root, args.dataset_dir or config_value(config, "paths.datasets", "datasets")
    )
    run_root = resolved_path(root, args.runs_dir or config_value(config, "paths.runs", "runs"))
    profile = make_profile(args, config)
    mode = "smoke" if args.smoke else "production" if args.production else "selection"
    sweep: list[SweepRun] = []
    for model in models:
        for variant in variants:
            run_id = f"{model.name}-{safe_name(variant)}-{mode}"
            run_path = run_root / run_id
            sweep.append(
                SweepRun(
                    run_id=run_id,
                    model=model,
                    variant=variant,
                    dataset_path=dataset_root / variant,
                    run_path=run_path,
                    adapter_path=run_path / "adapters",
                    profile=profile,
                    mode=mode,
                )
            )
    return sweep


def base_command(item: SweepRun) -> list[str]:
    profile = item.profile
    command = [
        "uv",
        "run",
        "--extra",
        "train",
        "python",
        "train_vlm.py",
        "--model-path",
        item.model.model_id,
        "--dataset",
        str(item.training_data_path),
        "--split",
        "train",
        "--seed",
        str(profile.seed),
        "--batch-size",
        str(profile.batch_size),
        "--val-batches",
        str(profile.validation_batches),
        "--learning-rate",
        str(profile.learning_rate),
        "--steps-per-report",
        str(profile.steps_per_report),
        "--steps-per-eval",
        str(profile.steps_per_eval),
        "--gradient-accumulation-steps",
        str(profile.gradient_accumulation_steps),
        "--lora-rank",
        str(profile.lora_rank),
        "--lora-alpha",
        str(profile.lora_alpha),
        "--lora-dropout",
        str(profile.lora_dropout),
        "--output-path",
        str(item.adapter_path),
        "--steps-per-save",
        str(profile.save_every),
        "--max-seq-length",
        str(profile.max_seq_length),
        "--train-on-completions",
    ]
    if profile.epochs is not None:
        command.extend(["--epochs", str(profile.epochs)])
    elif profile.iterations is not None:
        command.extend(["--iters", str(profile.iterations)])
    if profile.gradient_checkpointing:
        command.append("--grad-checkpoint")
    return command


def runnable_state(item: SweepRun) -> tuple[str, str | None]:
    if item.model.placeholder:
        return (
            "pending",
            "Replace <FILL_MODEL_ID> in config.yaml with a verified quantized MLX model ID.",
        )
    folded = item.model.model_id.casefold()
    bits = item.model.quantization_bits
    if f"{bits}bit" not in folded and f"{bits}-bit" not in folded:
        return (
            "pending",
            f"Model ID does not identify the configured Q{bits} base; verify it explicitly.",
        )
    train_file = item.training_data_path / "train.jsonl"
    if not train_file.is_file() or train_file.stat().st_size == 0:
        return "pending", f"Missing or empty MLX-VLM training dataset: {train_file}"
    return "ready", None


def shell_script(item: SweepRun) -> str:
    command = " ".join(shlex.quote(part) for part in base_command(item))
    resume_file = item.adapter_path / "adapters.safetensors"
    return f"""#!/usr/bin/env bash
set -euo pipefail

command=({command})
if [[ -f {shlex.quote(str(resume_file))} ]]; then
  command+=(--adapter-path {shlex.quote(str(resume_file))})
fi
"${{command[@]}}"
"""


def split_completion(text: str, prompt_fraction: float) -> tuple[str, str] | None:
    if len(text) < 2:
        return None
    lines = text.splitlines(keepends=True)
    if len(lines) > 1:
        boundary = min(len(lines) - 1, max(1, round(len(lines) * prompt_fraction)))
        prompt = "".join(lines[:boundary])
        completion = "".join(lines[boundary:])
    else:
        boundary = min(len(text) - 1, max(1, round(len(text) * prompt_fraction)))
        prompt = text[:boundary]
        completion = text[boundary:]
    if not prompt or not completion:
        return None
    return prompt, completion


def write_vlm_sources(
    sources: Sequence[Path], destination: Path, prompt_fraction: float
) -> int:
    available = [source for source in sources if source.is_file() and source.stat().st_size > 0]
    if not available:
        destination.unlink(missing_ok=True)
        return 0
    written = 0
    with destination.open("w", encoding="utf-8") as output_file:
        for source in available:
            with source.open(encoding="utf-8") as input_file:
                for line in input_file:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                        continue
                    parts = split_completion(row["text"], prompt_fraction)
                    if parts is None:
                        continue
                    prompt, completion = parts
                    output = {key: value for key, value in row.items() if key != "text"}
                    output.update({"question": prompt, "answer": completion})
                    output_file.write(json.dumps(output, ensure_ascii=False) + "\n")
                    written += 1
    if written == 0:
        destination.unlink(missing_ok=True)
    return written


def write_vlm_split(source: Path, destination: Path, prompt_fraction: float) -> int:
    return write_vlm_sources([source], destination, prompt_fraction)


def prepare_training_data(item: SweepRun) -> None:
    item.training_data_path.mkdir(parents=True, exist_ok=True)
    if item.mode == "production":
        write_vlm_sources(
            [item.dataset_path / "train.jsonl", item.dataset_path / "valid.jsonl"],
            item.training_data_path / "train.jsonl",
            item.profile.prompt_fraction,
        )
        (item.training_data_path / "valid.jsonl").unlink(missing_ok=True)
        return
    for split in ("train", "valid"):
        source = item.dataset_path / f"{split}.jsonl"
        destination = item.training_data_path / f"{split}.jsonl"
        write_vlm_split(source, destination, item.profile.prompt_fraction)


def prepare_sweep(sweep: Sequence[SweepRun], run_root: Path, root: Path) -> dict[str, Any]:
    run_root.mkdir(parents=True, exist_ok=True)
    manifest_runs: list[dict[str, Any]] = []
    command_lines = [
        "# Prepared MLX-VLM LoRA commands",
        "",
        "Generated from `config.yaml`. Placeholder jobs are intentionally not runnable.",
        "",
    ]
    for item in sweep:
        item.run_path.mkdir(parents=True, exist_ok=True)
        prepare_training_data(item)
        state, reason = runnable_state(item)
        script_path = item.run_path / "command.sh"
        write_text(script_path, shell_script(item))
        os.chmod(script_path, 0o755)
        command = base_command(item)
        plan = {
            "run_id": item.run_id,
            "model": asdict(item.model),
            "variant": item.variant,
            "dataset_path": str(item.dataset_path),
            "training_data_path": str(item.training_data_path),
            "run_path": str(item.run_path),
            "adapter_path": str(item.adapter_path),
            "profile": asdict(item.profile),
            "mode": item.mode,
            "state": state,
            "reason": reason,
            "command": command,
            "command_file": str(script_path),
        }
        write_json(item.run_path / "plan.json", plan)
        manifest_runs.append(plan)
        command_lines.extend(
            [
                f"## {item.run_id}",
                "",
                f"State: **{state}**" + (f" — {reason}" if reason else ""),
                "",
                "```bash",
                shlex.join(command),
                "```",
                "",
            ]
        )
    manifest = {
        "generated_at": utc_now(),
        "project_root": str(root),
        "execution": "sequential",
        "runs": manifest_runs,
    }
    write_json(run_root / "sweep.json", manifest)
    write_text(run_root / "COMMANDS.md", "\n".join(command_lines))
    return manifest


def execute_one(item: SweepRun, root: Path) -> tuple[str, str]:
    completed = item.run_path / "COMPLETED"
    if completed.is_file():
        return "skipped", "completion marker exists"
    state, reason = runnable_state(item)
    if state != "ready":
        record = {
            "run_id": item.run_id,
            "status": "pending",
            "reason": reason,
            "checked_at": utc_now(),
            "sources": ["config.yaml", str(item.dataset_path / "statistics.json")],
        }
        write_json(item.run_path / "run.json", record)
        return "pending", reason or "not runnable"
    command = base_command(item)
    resume_file = item.adapter_path / "adapters.safetensors"
    if resume_file.is_file():
        command.extend(["--adapter-path", str(resume_file)])
    started_at = utc_now()
    started = time.monotonic()
    record: dict[str, Any] = {
        "run_id": item.run_id,
        "status": "running",
        "started_at": started_at,
        "command": command,
        "model": asdict(item.model),
        "variant": item.variant,
        "profile": asdict(item.profile),
        "dataset_statistics": str(item.dataset_path / "statistics.json"),
        "adapter_path": str(item.adapter_path),
    }
    write_json(item.run_path / "run.json", record)
    log_path = item.run_path / "train.log"
    LOG.info("starting %s", item.run_id)
    process: subprocess.Popen[str] | None = None
    try:
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                log.write(line)
                log.flush()
            return_code = process.wait()
    except KeyboardInterrupt:
        if process is not None:
            process.terminate()
        return_code = 130
    except OSError as error:
        return_code = 127
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"Could not start training command: {error}\n")
    duration = time.monotonic() - started
    record.update(
        {
            "finished_at": utc_now(),
            "duration_seconds": round(duration, 3),
            "return_code": return_code,
            "log": str(log_path),
        }
    )
    if return_code == 130:
        record["status"] = "interrupted"
        record["reason"] = "Training was interrupted by the user."
        write_json(item.run_path / "run.json", record)
        return "interrupted", str(record["reason"])
    if return_code == 0 and resume_file.is_file():
        record["status"] = "completed"
        for checkpoint in item.adapter_path.glob("*_adapters.safetensors"):
            checkpoint.unlink()
        write_text(completed, f"{utc_now()}\n")
        write_json(item.run_path / "run.json", record)
        return "completed", f"{duration / 3600:.2f} hours"
    record["status"] = "failed"
    record["reason"] = (
        f"mlx_vlm.lora exited {return_code}"
        if return_code != 0
        else "command exited successfully but adapters.safetensors is missing"
    )
    write_json(item.run_path / "run.json", record)
    return "failed", str(record["reason"])


def execute_sweep(sweep: Sequence[SweepRun], root: Path) -> int:
    has_ready_runs = any(runnable_state(item)[0] == "ready" for item in sweep)
    if has_ready_runs and (platform.system() != "Darwin" or platform.machine() != "arm64"):
        LOG.error("MLX training requires Apple Silicon macOS; prepared commands remain in runs/.")
        return 1
    summaries: list[tuple[str, str]] = []
    failures = 0
    for item in sweep:
        status, detail = execute_one(item, root)
        summaries.append((item.run_id, f"{status}: {detail}"))
        if status == "interrupted":
            final_summary("Training sweep", summaries + [("Failures", failures)])
            return 130
        failures += status == "failed"
    final_summary("Training sweep", summaries + [("Failures", failures)])
    return 1 if failures else 0


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)
    config, root = load_config(args.config)
    sweep = make_sweep(args, config, root)
    run_root = sweep[0].run_path.parent
    manifest = prepare_sweep(sweep, run_root, root)
    ready = sum(item["state"] == "ready" for item in manifest["runs"])
    pending = len(manifest["runs"]) - ready
    if not args.execute:
        final_summary(
            "Training preparation",
            [
                ("Commands", len(sweep)),
                ("Ready", ready),
                ("Pending", pending),
                ("Manifest", run_root / "sweep.json"),
                ("Heavy training started", "no"),
            ],
        )
        return 0
    return execute_sweep(sweep, root)


if __name__ == "__main__":
    raise SystemExit(main())
