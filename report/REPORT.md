# From Repository History to a Verified Local Coding Agent

I built a source-traceable pipeline that turns my Git history into an audited code corpus, validates a pinned Gemma 4 E4B LoRA path on Apple Silicon, converts the adapter to GGUF, and verifies it through Ollama and Codex. This report contains only measured results and explicitly separates the current published dataset from the earlier compatibility checkpoint.

## Executive record

| Stage | Status | Measured evidence |
| --- | --- | --- |
| Repository provenance | Complete | 60 repositories; 3,018 matched commits |
| Published dataset | Complete | 18,361 rows from 58 repositories |
| MLX LoRA compatibility | Passed | One iteration; return code 0 |
| Adapter conversion | Passed | 343 PEFT/GGUF tensor pairs |
| Ollama runtime | Passed | 85.583 decode tokens/s; 0.301 s TTFT |
| Codex integration | Passed | File edited and test passed |

The local adapter is a one-step compatibility artifact created before the current dataset revision. It verifies the training, conversion, packaging, native-tool, and agent integration contracts. It is not a completed adaptation result on the current 18,361-row corpus, and I do not make held-out coding-quality claims from it.

## Corpus provenance

The statistics stage walks complete Git history with rename detection and attributes commits against configured identities. The current aggregate record contains:

| Measure | Value |
| --- | ---: |
| Repositories processed | 60 |
| Matched commits | 3,018 |
| Added lines | 868,278 |
| Removed lines | 302,760 |
| Net lines | 565,518 |
| Changed lines | 1,171,038 |

These are contribution-history measurements, not physical lines in the current checkouts. Canonical evidence: [`stats/stats.json`](../stats/stats.json).

## Published dataset

The current corpus matches Hugging Face revision [`9f9829f9dbd75400595ea2e211e291fbcc47ba0f`](https://huggingface.co/datasets/JulianAT/personal-codex-model/tree/9f9829f9dbd75400595ea2e211e291fbcc47ba0f).

| Measure | Value | Control |
| --- | ---: | --- |
| Repositories with rows | 58 | 52 train; 6 validation |
| Files considered | 8,970 | Extension, path, size, and secret filters |
| Files selected | 8,924 | 7,913 source files emitted rows |
| Candidate chunks | 20,858 | 896-token target; 64-token overlap |
| Exact duplicates removed | 1,470 | SHA-256 |
| Near duplicates removed | 1,027 | MinHash; threshold 0.85 |
| Rows retained | 18,361 | 15,226 train; 3,135 validation |
| Lexical tokens | 12,472,126 | All retained rows |
| Emitted lines | 1,305,187 | 1,153,093 nonblank |
| UTF-8 source text | 50.52 MiB | 52,976,638 bytes |

Repository-level splitting prevents one repository from contributing to both train and validation. The local `datasets.load_dataset(...)` self-test and the published Parquet split counts both passed. Canonical local evidence: [`datasets/raw-max/statistics.json`](../datasets/raw-max/statistics.json).

The builder rejected 31 sensitive-path hits and one high-confidence OpenAI-key signature in this build. The public Hub payload contains explicitly staged and scanned Parquet rows, schema metadata, a dataset card, and sanitized statistics. Raw repository checkouts, local paths, adapter weights, and credentials are excluded. Some approved rows originated in repositories that were private at collection time; automated filtering does not replace manual review, copyright review, or source-license compliance.

## Model contract

| Property | Value |
| --- | --- |
| Training checkpoint | `unsloth/gemma-4-E4B-it-UD-MLX-4bit` |
| Revision | `f29de68cb284ca208446e647b339569935025ef3` |
| Upstream lineage | `google/gemma-4-E4B-it` |
| Runtime base | `gemma4:e4b` |
| Trainer | MLX-VLM |
| LoRA | Rank 8; alpha 16; dropout 0 |
| Sequence limit | 1,024 tokens |
| Trainable parameters | 19.441488 M (0.254%) |

Only language-model linear layers receive adapters; vision and audio stacks remain frozen. The model, revision, optimization profile, and runtime lineage are pinned in [`config.yaml`](../config.yaml) and repeated in the run manifest.

## Compatibility checkpoint

| Metric | Measured value |
| --- | ---: |
| Iterations | 1 |
| Training loss | 1.38960338 |
| Training throughput | 96.345 tokens/s |
| Trained tokens | 169 |
| Peak unified memory | 8.906 GB |
| Wall time | 16.308 s |
| Return code | 0 |

Canonical evidence: `runs/gemma-4-e4b-it-4bit-raw-max-smoke/run.json` and its `train.log`.

## Packaging and deployment

I transpose and rename MLX LoRA factors into PEFT Safetensors, convert the PEFT adapter with pinned llama.cpp revision `b15ca938ad00aa6b3ee6c2edda7363fd02826b18`, and attach the resulting GGUF adapter to the matching Ollama base.

| Gate | Measured result | Threshold |
| --- | ---: | ---: |
| GGUF tensor pairs | 343 | Structural validation |
| GGUF size | 38,926,080 bytes | Recorded artifact |
| Decode throughput | 85.583 tokens/s | At least 20 tokens/s |
| Time to first token | 0.301 s | At most 5.0 s |
| Native tool selection | Passed | `read_project_status` |
| Codex edit-and-test | Passed | `pytest`: 1 passed |

Canonical evidence: [`deployment/gemma4-e4b-julian-latest/verification.json`](../deployment/gemma4-e4b-julian-latest/verification.json).

## Reference environment

| Layer | Recorded value |
| --- | --- |
| Hardware | Apple M4 Pro; 14 cores; 24 GB unified memory |
| Operating system | macOS 26.5.1 (25F80) |
| Python | 3.14.2 |
| MLX / MLX-VLM | 0.32.0 / 0.6.4 |
| Ollama | 0.32.0 |
| Codex CLI | 0.144.5 |

This sanitized current workstation snapshot is stored in [`report/environment.json`](environment.json). Historical run artifacts remain authoritative for run-specific telemetry.

## Reproducibility

Rebuild the pipeline from repositories visible to the authenticated GitHub account:

```bash
uv sync --extra train --extra deploy
gh auth status
uv run python fetch_repos.py
uv run python stats.py
uv run python build_dataset.py
```

Those commands reproduce the process, not my exact private source snapshot. Load the immutable public boundary directly:

```python
from datasets import load_dataset

dataset = load_dataset(
    "JulianAT/personal-codex-model",
    revision="9f9829f9dbd75400595ea2e211e291fbcc47ba0f",
)
print({name: len(split) for name, split in dataset.items()})
```

Recreate and deploy the compatibility adapter:

```bash
uv run python train.py --smoke --execute
uv run --extra train --extra deploy python deploy_ollama.py \
  --adapter-path runs/gemma-4-e4b-it-4bit-raw-max-smoke/adapters \
  --tag gemma4-e4b-julian:latest \
  --execute
```

## Evaluation boundary

- The checkpoint verifies systems compatibility, not broad model quality.
- Repository history is correlated and does not represent independent programming tasks.
- Secret filtering reduces obvious risk but is not publication clearance.
- Performance measurements are specific to the recorded Apple Silicon environment.
- The Codex task is an integration gate, not a general software-engineering benchmark.

The next evidence layer is a controlled base-versus-adapter evaluation on repository-isolated held-out data with fixed sampling, perplexity, next-line exact match, throughput, and peak-memory measurements.
