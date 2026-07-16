export const PROSE_MAX_WIDTH = "max-w-[775px]";
export const FIGURE_MAX_WIDTH = "max-w-[820px]";

export const GITHUB_URL = "https://github.com/Julian-AT";
export const PORTFOLIO_URL = "https://julianschmidt.cv";
export const PROFILE_IMAGE_URL =
  "https://www.julianschmidt.cv/_next/image?url=%2Fassets%2Fimages%2Fprofile.jpg&w=96&q=90&dpl=dpl_GZP5i46do69MAXsSiNbok81iUWt7";
export const HERO_IMAGE_URL =
  "https://media.licdn.com/dms/image/v2/D5612AQEv0hpRCYHXqA/article-cover_image-shrink_600_2000/article-cover_image-shrink_600_2000/0/1709146661824?e=2147483647&v=beta&t=ZlcB0jh0eNVZ-IOMbv7KVGjsUsyI3DxR_6DCxxwxIl0";
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
  { id: "limits-and-status", label: "Limits and status" },
] as const;
