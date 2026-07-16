export const PROSE_MAX_WIDTH = "max-w-[775px]";
export const FIGURE_MAX_WIDTH = "max-w-[820px]";

export const GITHUB_URL = "https://github.com/Julian-AT";
export const HF_DATASET_URL =
  "https://huggingface.co/datasets/JulianAT/personal-codex-model";
export const PORTFOLIO_URL = "https://julianschmidt.cv";
export const PROFILE_IMAGE_URL = "/images/julian-schmidt.jpg";
export const HERO_IMAGE_URL = "/images/corpus-to-agent-pipeline.png";
export const SITE_URL =
  process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost:3000";

export const ARTICLE = {
  title: "From Repository History to a Verified Local Coding Agent",
  description:
    "I built a source-traceable code corpus, published its audited dataset, validated a Gemma 4 LoRA path on Apple Silicon, and deployed it through Ollama and Codex.",
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
  { id: "evaluation-scope", label: "Evaluation scope" },
] as const;
