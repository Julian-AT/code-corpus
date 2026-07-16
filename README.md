# Personalized Gemma 4 E4B

This project builds a source-traceable code corpus, fine-tunes one pinned Gemma 4 E4B
checkpoint with MLX-VLM LoRA, evaluates the adapter against its untouched base, and packages it as
an offline Ollama model that Codex can use. Public model-card results are report context only; they
do not influence model selection and are never presented as local measurements.

The target is fixed:

- training: `unsloth/gemma-4-E4B-it-UD-MLX-4bit` at revision
  `f29de68cb284ca208446e647b339569935025ef3`
- upstream lineage: `google/gemma-4-E4B-it`
- deployment base: `gemma4:e4b`
- local runtime: Apple Silicon MLX for training, Ollama for inference, Codex for agent use

## Verified smoke result

The one-step E4B gate completed on this machine with 19.44M trainable parameters, 1.3896 training
loss, 96.35 training tokens/s, and 8.91 GB peak unified memory. The exported GGUF LoRA contains 343
tensor pairs. Ollama then passed a native tool-call check at 85.58 decode tokens/s and 0.301 s time
to first token, and Codex fixed and tested a deliberately broken Python fixture through the local
model. Canonical evidence is in
[`deployment/gemma4-e4b-julian-latest/verification.json`](deployment/gemma4-e4b-julian-latest/verification.json).

The current public corpus was generated after that compatibility checkpoint. [Hugging Face revision
`9f9829f9dbd75400595ea2e211e291fbcc47ba0f`](https://huggingface.co/datasets/JulianAT/personal-codex-model/tree/9f9829f9dbd75400595ea2e211e291fbcc47ba0f) contains 18,361 rows from 58 repositories, split into
15,226 training and 3,135 validation rows. The checkpoint verifies the systems path; it is not
presented as a completed adaptation result on this newer dataset revision.

## Requirements

- Apple Silicon macOS; the checked profile targets an M4 Pro with 24 GB unified memory
- Python 3.11 or newer and [`uv`](https://docs.astral.sh/uv/)
- `git`, authenticated GitHub CLI (`gh`), Ollama, and Codex CLI

```bash
uv sync --extra train --extra deploy
gh auth status
ollama --version
codex --version
```

## Reproduce the pipeline

The configured fetch includes every non-fork, non-archived repository visible to the authenticated
GitHub account, including private repositories. It clones full history because contribution
statistics and quality metadata need it.

```bash
uv run python fetch_repos.py
uv run python stats.py
uv run python build_dataset.py
```

The dataset builder rejects configured sensitive paths and high-confidence credential signatures,
then performs exact SHA-256 and MinHash near-deduplication. Splits are deterministic by repository;
no repository contributes to both train and validation. Every generated dataset includes a
`datasets.load_dataset(...)` self-test in `statistics.json`.

## Publish the dataset to Hugging Face

The configured Hub target is `JulianAT/personal-codex-model`. Publication stages only the dataset card,
Parquet shards, schema metadata, and a sanitized statistics audit; local Arrow/JSONL duplicates and
absolute paths are not uploaded. It validates the card and schema, scans every staged row for the
same high-confidence secret patterns as the builder, then reloads the remote dataset after upload.

This repository is intentionally public even though the configured corpus includes retained rows
from private GitHub repositories. That exceptional opt-in is recorded as
`hub.allow_private_sources: true`; without it, the publisher refuses a public upload.

```bash
# Validate the exact Hub payload without changing remote state
uv run python publish_dataset.py --dry-run

# Create/update the public dataset repo and verify the result
uv run python publish_dataset.py
```

The generated dataset card declares Hugging Face metadata for code language, text generation,
dataset size, source-code modality, completion use, deduplication, and the `datasets` library so the
Hub can recognize and index the relevant tags automatically.

Run the small E4B gate before a long job:

```bash
uv run python train.py --smoke --execute
uv run --extra train --extra deploy python deploy_ollama.py \
  --adapter-path runs/gemma-4-e4b-it-4bit-raw-max-smoke/adapters \
  --tag gemma4-e4b-julian:latest \
  --execute
```

Train the selection adapter on training repositories and evaluate it on held-out repositories:

```bash
uv run python train.py --execute
uv run python evaluate.py
```

Once the configuration is locked, train a distinct production adapter on the merged safe corpus
and deploy it:

```bash
uv run python train.py --production --execute
uv run --extra train --extra deploy python deploy_ollama.py \
  --adapter-path runs/gemma-4-e4b-it-4bit-raw-max-production/adapters \
  --tag gemma4-e4b-julian:latest \
  --execute
uv run python evaluate.py
```

`deploy_ollama.py` converts MLX LoRA tensors to PEFT Safetensors, uses a pinned llama.cpp converter
to produce a Gemma 4 GGUF adapter, attaches it to the official Ollama base, and refuses success
unless tool use, throughput, latency, and the Codex file-edit smoke all pass.

## Use it in Codex

```bash
codex --oss --local-provider ollama --model gemma4-e4b-julian:latest \
  -c model_context_window=32768 \
  -c model_auto_compact_token_limit=24576
```

The Ollama tag itself also fixes `num_ctx` at 32,768. Both settings matter: Codex needs the client
budget, while Ollama needs the server-side context allocation.

The persistent assistant identity and Julian-specific context live in
`deployment/system-prompt.txt`. Deployment embeds that file as the Modelfile `SYSTEM` instruction.

## Report artifacts

- `report/REPORT.md`: current methodology, evidence tables, scope, and reproducibility notes
- `report/environment.json`: sanitized reference workstation and toolchain snapshot
- `report/results.json`: generated evaluation-plan state; use only completed observations
- `stats/STATS.md` and `stats/charts/`: contribution history and aggregate provenance figures
- `www/`: the publication layer for the current measured record

## Web report

`www/` is the publication layer for the project. It presents the source-traceable corpus, dataset publication, compatibility training, conversion, and deployment record as a responsive Next.js report while keeping repository checkouts, weights, local paths, and secrets out of the rendered page. Its narrative explicitly separates the current Hugging Face revision from the earlier compatibility checkpoint.

```bash
cd www
bun install
bun run check
bun run dev
```

The web report's maintenance guide and evidence map are documented in [`www/README.md`](www/README.md). Set `NEXT_PUBLIC_SITE_URL` before deployment so canonical and structured metadata use the public origin.

The public benchmark manifest keeps organization, evaluator context, benchmark version, unit, and
URL on every observation. It deliberately avoids averaging unrelated benchmarks into a synthetic
score.

## Privacy and interpretation

Raw repository checkouts and adapters stay local unless explicitly published. The Hugging Face
dataset is an intentional public release of staged, scanned Parquet rows and sanitized metadata;
some approved rows originated in repositories that were private at collection time. Secret
filtering is a high-precision safety layer, not proof that a corpus is publishable. Inspect retained
paths and honor every source repository's license before sharing data or weights.

Local next-line exact match and perplexity measure personalization on held-out repositories; they
do not establish broad software-engineering ability. Vendor model-card scores are useful context,
but differences in harnesses and sampling prevent treating them as one controlled leaderboard.

## Development checks

```bash
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest
```
