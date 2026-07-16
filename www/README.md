# Julian local model report

A self-contained technical report for the private code-corpus pipeline in the parent repository. The site documents how repository history becomes a filtered training dataset, a Gemma 4 E4B LoRA adapter, and a verified local Ollama model for Codex.

The report deliberately distinguishes the completed one-step compatibility adapter from an uncompleted full-corpus training run. Every published metric points back to a machine-readable artifact in the parent project.

## What the site covers

- Contribution provenance across 60 repositories and 3,018 matched commits
- Dataset filtering, repository-level splitting, deduplication, and token distribution
- The pinned Gemma 4 E4B model and rank-8 MLX-VLM LoRA contract
- One-step training telemetry and its exact limitations
- MLX to PEFT to GGUF conversion and Ollama packaging
- Tool-call, latency, throughput, and Codex edit-and-test verification
- The persistent Julian-specific system prompt and local usage commands

## Canonical sources

The site is a presentation layer. These files remain authoritative:

| Topic | Source |
| --- | --- |
| Contribution history | `../stats/stats.json` |
| Dataset construction | `../datasets/raw-max/statistics.json` |
| Training run | `../runs/gemma-4-e4b-it-4bit-raw-max-smoke/run.json` and `train.log` |
| Deployment gate | `../deployment/gemma4-e4b-julian-latest/verification.json` |
| Model and pipeline configuration | `../config.yaml` |
| Assistant identity | `../deployment/system-prompt.txt` |

## Stack

- Next.js 16 App Router and React 19
- TypeScript in strict mode
- MDX for the report body
- Tailwind CSS v4
- Next.js Image and Metadata APIs

The page remains a Server Component. Client JavaScript is limited to the table of contents, copy controls, and mobile navigation.

## Development

```bash
bun install
bun run dev
```

Open `http://localhost:3000`.

Run the complete verification gate with:

```bash
bun run check
```

The command runs ESLint, TypeScript, and a production build. Set `NEXT_PUBLIC_SITE_URL` to the deployed origin before publishing so canonical, sitemap, and structured-data URLs are correct.

## Content maintenance

The narrative lives in `src/content/report.mdx`. Shared metrics and metadata live in `src/lib/constants.ts`; report-specific visualizations live in `src/components/report-visuals.tsx`.

When upstream evidence changes, update the displayed values from the canonical source files above and rerun `bun run check`. Never replace a pending or interrupted result with an estimate.
