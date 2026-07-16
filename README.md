# Personalized Gemma 4 E4B

This project builds a private, source-traceable code corpus, fine-tunes one pinned Gemma 4 E4B
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

`uv run python evaluate.py` regenerates PNG and SVG versions of every figure. Seaborn supplies all
quantitative chart marks.

- `report/REPORT.md`: report-ready methodology, evidence tables, caveats, and figure references
- `report/results.json`: canonical local evaluation, training telemetry, and deployment evidence
- `report/charts/upstream-*.{png,svg}`: sourced, unadapted public model comparisons
- `report/charts/training-telemetry.{png,svg}`: loss, throughput, and peak-memory traces
- `report/charts/local-*.{png,svg}`: measured base-versus-adapter results
- `report/charts/offline-deployment-headroom.{png,svg}`: threshold-normalized Ollama margins
- `stats/STATS.md` and `stats/charts/`: contribution history and corpus provenance

## Web report

`www/` is the publication layer for the project. It presents the source-traceable corpus, training, conversion, and deployment record as a responsive Next.js report while keeping private repositories, dataset rows, weights, and local paths out of the rendered page. Its narrative deliberately distinguishes the completed one-step compatibility adapter from the interrupted selection run and the unrun production training.

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

Private source, datasets, and adapters stay local unless you publish them. Secret filtering is a
high-precision safety layer, not proof that a corpus is publishable. Inspect retained paths and
honor every source repository's license before sharing any data or weights.

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
