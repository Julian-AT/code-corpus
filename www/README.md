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

## Content and visual architecture

- `src/content/report.mdx` contains the long-form narrative, evidence tables, and figure placement.
- `src/components/report-visuals.tsx` contains the dataset histogram, verification scorecard, and scoped status callouts using the same monochrome primitives as the original report.
- `src/lib/constants.ts` is the single source for publication metadata, section navigation, Julian's links, and the two external editorial image URLs.
- `public/images/` contains publication-safe copies of the generated aggregate charts from `../stats/charts/`. It must never contain raw code, dataset rows, adapter weights, local paths, or secrets.

The visual source of truth is the original Attention Seekers implementation at `../repos/zero_one_hack_01/www`. This report retains its exact cream and ink palette, Geist typography, 68px header, centered editorial hero, 775px reading column, 820px figures, table treatment, sticky table of contents, code blocks, and footer structure. Only the subject matter, navigation labels, Julian Schmidt identity, portrait, and hero artwork are substituted.

The portrait and supplied LoRA hero image are allow-listed in `next.config.mjs` for Next.js image optimization.

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

Use this maintenance sequence:

1. Regenerate the parent project's statistics, dataset manifest, training record, and deployment verification.
2. Compare the report's prose and visual constants against the canonical JSON files.
3. Copy only aggregate, publication-safe charts into `public/images/` and record accurate dimensions in the MDX figure.
4. Check every section link, table caption, threshold, unit, and completion-status statement.
5. Run `bun run check`, then inspect desktop and mobile renders before publishing.

The default canonical URL is `http://localhost:3000`. Set `NEXT_PUBLIC_SITE_URL` in the deployment environment so metadata, JSON-LD, `robots.txt`, and `sitemap.xml` point to the production origin.
