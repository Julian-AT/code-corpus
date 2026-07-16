export const PROSE_MAX_WIDTH = "max-w-[760px]";
export const FIGURE_MAX_WIDTH = "max-w-[1040px]";

export const GITHUB_URL = "https://github.com/Julian-AT";
export const PORTFOLIO_URL = "https://julianschmidt.cv";
export const SITE_URL =
  process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost:3000";

export const ARTICLE = {
  title: "A local coding model with a paper trail",
  description:
    "A source-traceable report on turning Julian Schmidt's repository history into a filtered code corpus, a Gemma 4 E4B LoRA adapter, and a verified local Ollama model for Codex.",
  datePublished: "2026-07-16",
  dateModified: "2026-07-16",
  authors: ["Julian Schmidt"],
} as const;

export const TOC_SECTIONS = [
  { id: "executive-summary", label: "Executive summary" },
  { id: "corpus-provenance", label: "Corpus provenance" },
  { id: "dataset-construction", label: "Dataset construction" },
  { id: "model-and-adapter", label: "Model and adapter" },
  { id: "verified-deployment", label: "Verified deployment" },
  { id: "personalization", label: "Personalization" },
  { id: "reproducibility", label: "Reproducibility" },
  { id: "limitations", label: "Limits and status" },
] as const;
