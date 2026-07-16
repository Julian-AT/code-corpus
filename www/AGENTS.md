<!-- BEGIN:nextjs-agent-rules -->
# This is NOT the Next.js you know

This version has breaking changes. Read the relevant guide in `node_modules/next/dist/docs/` before writing code and heed deprecation notices.
<!-- END:nextjs-agent-rules -->

# Julian local model report

This Next.js site is the presentation layer for the source-traceable model pipeline in the parent directory. Its single job is to explain what was built, what was measured, and what remains incomplete without overstating the one-step adapter.

## Evidence rules

- Treat `../stats/stats.json`, `../datasets/raw-max/statistics.json`, the completed smoke `run.json`, and `../deployment/gemma4-e4b-julian-latest/verification.json` as canonical.
- Label the deployed adapter as a one-step compatibility artifact, not a completed full-corpus fine-tune.
- Do not publish estimates as measurements.
- Keep private source code, dataset rows, local filesystem paths, and secrets out of the rendered page.

## Tech stack

- Next.js 16 App Router, React 19, TypeScript strict
- MDX via `@next/mdx` and `rehype-slug`
- Tailwind CSS v4
- Geist Sans and Geist Mono from the existing local package

## Commands

- `bun run dev`
- `bun run lint`
- `bun run typecheck`
- `bun run build`
- `bun run check`

## Code style

- Named exports, PascalCase components, kebab-case component filenames
- Server Components by default; use client boundaries only for interaction
- No `any`, no inline styles, no source comments
- Use `next/image` with dimensions and responsive `sizes`
- Mobile-first responsive layouts, visible focus, reduced-motion support
- Keep data prose in MDX and reusable visual structures in components
