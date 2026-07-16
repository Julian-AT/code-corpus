import Image from "next/image";
import {
  ARTICLE,
  FIGURE_MAX_WIDTH,
  HERO_IMAGE_URL,
} from "@/lib/constants";

export function ArticleHero() {
  return (
    <header className="mt-12 flex flex-col items-center gap-8 text-center">
      <div className="flex flex-col items-center gap-8 text-center">
        <p className="font-sans text-[15px] font-bold leading-[21px] text-ink">
          Applied AI systems · Reproducible ML engineering
        </p>
        <h1 className="max-w-4xl text-balance font-sans text-[32px] font-bold leading-[1.1] text-ink lg:text-[52px] lg:leading-[57.2px]">
          {ARTICLE.title}
        </h1>
        <div>
          <span className="font-sans text-[15px] leading-[21px] text-ink">
            Julian Schmidt · July 16, 2026
          </span>
        </div>
      </div>

      <div
        className={`relative mx-auto aspect-[1200/630] w-full ${FIGURE_MAX_WIDTH} overflow-hidden rounded-[24px] bg-white`}
      >
        <Image
          src={HERO_IMAGE_URL}
          alt="Repository history flowing through an audited corpus, Gemma 4 LoRA adapter, GGUF conversion, and a verified Ollama and Codex runtime"
          fill
          priority
          sizes="(max-width: 820px) 100vw, 820px"
          className="object-contain"
        />
      </div>
    </header>
  );
}
