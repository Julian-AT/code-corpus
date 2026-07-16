# Gemma 4 E4B training and deployment protocol

## Model contract

Training is intentionally limited to the pinned instruction-tuned Gemma 4 E4B MLX checkpoint in
`config.yaml`. The adapter targets the language model's linear layers; the vision and audio stacks
remain frozen. The project does not use CUDA, `bitsandbytes`, or a PyTorch QLoRA training loop.

The MLX checkpoint and the Ollama `gemma4:e4b` base share the `google/gemma-4-E4B-it` lineage.
Deployment still requires an explicit tensor conversion because Ollama cannot directly attach an
MLX Safetensors adapter to Gemma 4.

## Three modes

`train.py` always prepares a machine-readable plan, staged MLX-VLM data, and an exact command before
it executes anything.

| Mode | Command | Data | Purpose |
| --- | --- | --- | --- |
| smoke | `uv run python train.py --smoke --execute` | bounded existing split, one step | compatibility and memory gate |
| selection | `uv run python train.py --execute` | training repositories only | held-out model evaluation |
| production | `uv run python train.py --production --execute` | train and valid merged | final offline adapter |

The selection split is assigned at repository granularity. `min_validation_repos` requires at least
three validation repositories when enough repositories contain data. Production is a separate run
directory and never overwrites the evidence-producing selection adapter.

## Training profile

The checked 24 GB profile uses batch size 1, 1,024 tokens, rank-8 LoRA, alpha 16, completion-only
loss, gradient checkpointing, four-step gradient accumulation, and one full data pass. Epoch length
is padded by at most three batches so the final accumulated gradient is applied; every staged row
is still visited. Validation runs at the first iteration and then at the configured interval for
selection jobs. Production has no validation split because all eligible data is deliberately used.

The local `train_vlm.py` wrapper wires the `validation` split into MLX-VLM's trainer. This is
intentional: MLX-VLM 0.6.4 exposes validation arguments but its stock LoRA entry point passes
`val_dataset=None`.

Numbered recovery checkpoints are written every 1,000 iterations. After a successful run they are
removed because the final `adapters.safetensors` is canonical. If a process is interrupted, the
next invocation warm-starts from the last adapter weights and runs the configured schedule again;
MLX-VLM does not persist optimizer state, so this is not an exact iteration-level resume.

## Run artifacts

Each `runs/<run-id>/` contains:

- `plan.json`: model revision, profile, paths, mode, and exact argv
- `command.sh`: reproducible command with adapter warm-start behavior
- `mlx-data/`: deterministic `question`/`answer` staging data
- `train.log`: validation loss, training loss, tokens/s, and peak unified memory
- `run.json`: timestamps, return code, duration, and completion status
- `adapters/adapter_config.json` and `adapters/adapters.safetensors`: final LoRA package
- `COMPLETED`: written only after a successful process and verified adapter output

Monitor a live run with:

```bash
tail -f runs/gemma-4-e4b-it-4bit-raw-max-selection/train.log
```

## Evaluation boundary

Run `uv run python evaluate.py` before production training. It compares the untouched MLX base and
selection adapter on the same held-out rows with:

- generation tokens per second after warm-up
- peak unified memory
- next-token perplexity
- next-line exact match

The evaluator records sample counts, token counts, repetition settings, and source fingerprints in
`report/results.json`. A production adapter is not evaluated on the former validation repositories
because they became training data; the selection metrics remain the honest generalization result.

## Ollama and Codex gate

```bash
uv run --extra train --extra deploy python deploy_ollama.py \
  --adapter-path runs/gemma-4-e4b-it-4bit-raw-max-production/adapters \
  --tag gemma4-e4b-julian:latest \
  --execute
```

The deployment stage performs this chain:

1. Validate the MLX adapter and preserve rank/alpha metadata.
2. Transpose and rename all paired LoRA factors into PEFT Safetensors form.
3. Convert the PEFT adapter with the pinned llama.cpp revision in `config.yaml`.
4. Attach the GGUF adapter to Ollama's official `gemma4:e4b` model at a 32K context.
5. Require a native Ollama tool call, minimum decode throughput, maximum time to first token, and a
   real Codex edit-and-test task.

Evidence is written to `deployment/<tag>/verification.json`. A failed gate leaves its evidence but
returns nonzero; it is never silently promoted as a working local model.
