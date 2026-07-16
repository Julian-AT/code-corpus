# Prepared MLX-VLM LoRA commands

Generated from `config.yaml`. Placeholder jobs are intentionally not runnable.

## gemma-4-e4b-it-4bit-raw-max-selection

State: **ready**

```bash
uv run --extra train python train_vlm.py --model-path unsloth/gemma-4-E4B-it-UD-MLX-4bit --dataset /Users/julianschmidt/train-corpus/code-corpus/runs/gemma-4-e4b-it-4bit-raw-max-selection/mlx-data --split train --seed 42 --batch-size 1 --val-batches 10 --learning-rate 1e-05 --steps-per-report 10 --steps-per-eval 500 --gradient-accumulation-steps 4 --lora-rank 8 --lora-alpha 16.0 --lora-dropout 0.0 --output-path /Users/julianschmidt/train-corpus/code-corpus/runs/gemma-4-e4b-it-4bit-raw-max-selection/adapters --steps-per-save 1000 --max-seq-length 1024 --train-on-completions --epochs 1 --grad-checkpoint
```
