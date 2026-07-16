from __future__ import annotations

import argparse
import logging
import random

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx_vlm.lora import (
    setup_model_for_training,
    transform_dataset_to_messages,
)
from mlx_vlm.trainer.datasets import VisionDataset
from mlx_vlm.trainer.sft_trainer import TrainingArgs, train
from mlx_vlm.trainer.utils import not_supported_for_training, print_trainable_parameters
from mlx_vlm.utils import load

from datasets import Dataset, DatasetDict, load_dataset
from mlx_compat import gemma4_checkpoint_compatibility

LOG = logging.getLogger(__name__)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seeded MLX-VLM LoRA training entry point.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--dataset-config")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--iters", type=positive_int, default=600)
    parser.add_argument("--epochs", type=positive_int)
    parser.add_argument("--steps-per-report", type=positive_int, default=10)
    parser.add_argument("--steps-per-eval", type=positive_int, default=100)
    parser.add_argument("--steps-per-save", type=positive_int, default=100)
    parser.add_argument("--val-batches", type=positive_int, default=10)
    parser.add_argument("--max-seq-length", type=positive_int, default=1024)
    parser.add_argument("--gradient-accumulation-steps", type=positive_int, default=1)
    parser.add_argument("--lora-rank", type=positive_int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--output-path", default="adapters")
    parser.add_argument("--adapter-path")
    parser.add_argument("--assistant-id", type=int, default=77091)
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--train-on-completions", action="store_true")
    args = parser.parse_args()
    args.image_resize_shape = None
    args.custom_prompt_format = None
    args.grad_clip = None
    args.full_finetune = False
    args.train_vision = False
    args.train_mode = "sft"
    args.beta = 0.1
    args.eps = 1e-8
    if args.learning_rate <= 0 or args.lora_alpha <= 0:
        parser.error("learning rate and LoRA alpha must be greater than zero")
    if not 0 <= args.lora_dropout <= 1:
        parser.error("LoRA dropout must be between zero and one")
    return args


def training_iterations(args: argparse.Namespace, dataset_size: int) -> int:
    if args.epochs is None:
        return args.iters
    batches = (dataset_size // args.batch_size) * args.epochs
    remainder = batches % args.gradient_accumulation_steps
    if remainder:
        batches += args.gradient_accumulation_steps - remainder
    return batches


def prepare_dataset(
    dataset: DatasetDict, split: str, model_type: str
) -> tuple[Dataset, Dataset | None]:
    if split not in dataset:
        raise ValueError(f"Dataset does not contain the requested split: {split}")
    train_data = transform_dataset_to_messages(dataset[split], model_type)
    validation = dataset.get("validation")
    valid_data = (
        transform_dataset_to_messages(validation, model_type) if validation is not None else None
    )
    return train_data, valid_data


def run_training(args: argparse.Namespace) -> None:
    output_path = (
        args.output_path
        if args.output_path.endswith(".safetensors")
        else f"{args.output_path}/adapters.safetensors"
    )
    LOG.info("Loading model from %s", args.model_path)
    model, processor = load(
        args.model_path,
        processor_config={"trust_remote_code": True},
    )
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if model_type in not_supported_for_training:
        raise ValueError(f"Model type {model_type} is not supported for training")
    if not isinstance(model_type, str):
        raise ValueError("Loaded model does not expose a model type")

    LOG.info("Loading dataset from %s", args.dataset)
    dataset = load_dataset(args.dataset, args.dataset_config or None)
    if not isinstance(dataset, DatasetDict):
        raise ValueError("Training dataset must provide named train and validation splits")
    train_data, valid_data = prepare_dataset(dataset, args.split, model_type)
    config = model.config.__dict__
    train_dataset = VisionDataset(
        train_data,
        config,
        processor,
        image_resize_shape=None,
        train_on_completions=args.train_on_completions,
    )
    valid_dataset = (
        VisionDataset(
            valid_data,
            config,
            processor,
            image_resize_shape=None,
            train_on_completions=args.train_on_completions,
        )
        if valid_data is not None
        else None
    )
    iterations = training_iterations(args, len(train_data))
    model = setup_model_for_training(model, args, args.adapter_path)
    print_trainable_parameters(model)
    optimizer = optim.Adam(learning_rate=args.learning_rate)
    training_args = TrainingArgs(
        batch_size=args.batch_size,
        iters=iterations,
        steps_per_report=args.steps_per_report,
        steps_per_eval=args.steps_per_eval,
        steps_per_save=args.steps_per_save,
        val_batches=args.val_batches,
        max_seq_length=args.max_seq_length,
        adapter_file=output_path,
        grad_checkpoint=args.grad_checkpoint,
        learning_rate=args.learning_rate,
        grad_clip=None,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        full_finetune=False,
    )
    train(
        model=model,
        optimizer=optimizer,
        train_dataset=train_dataset,
        val_dataset=valid_dataset,
        args=training_args,
        train_on_completions=args.train_on_completions,
        assistant_id=args.assistant_id,
    )
    LOG.info("Training completed; model saved to %s", output_path)


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    mx.random.seed(args.seed)
    with gemma4_checkpoint_compatibility(args.model_path):
        run_training(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
