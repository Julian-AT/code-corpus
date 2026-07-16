import { EvidenceSpine, MetricRail } from "@/components/report-visuals";

export function ArticleHero() {
  return (
    <header className="telemetry-grid border-b border-white/10 text-white">
      <div className="mx-auto grid w-full max-w-[1440px] gap-14 px-5 py-16 sm:px-8 sm:py-20 lg:grid-cols-[minmax(0,1.05fr)_minmax(420px,0.95fr)] lg:gap-20 lg:px-12 lg:py-28">
        <div className="hero-enter flex flex-col justify-center">
          <div className="mb-8 flex flex-wrap items-center gap-3 font-mono text-[11px] font-medium uppercase tracking-[0.16em] text-white/65">
            <span>Private model engineering</span>
            <span className="h-1 w-1 rounded-full bg-signal" />
            <span>Julian Schmidt</span>
            <span className="h-1 w-1 rounded-full bg-signal" />
            <time dateTime="2026-07-16">July 2026</time>
          </div>
          <h1 className="max-w-[820px] text-balance text-[46px] font-semibold leading-[0.98] tracking-[-0.055em] text-white sm:text-[64px] lg:text-[78px]">
            A local coding model with a paper trail.
          </h1>
          <p className="mt-8 max-w-2xl font-serif text-[20px] leading-[1.55] text-white/76 sm:text-[22px]">
            Sixty repositories became a filtered, source-traceable code corpus,
            then a Gemma 4 E4B LoRA adapter, then a local Ollama model that Codex
            could call, edit with, and test.
          </p>
          <div className="mt-9 flex flex-wrap gap-3">
            <a
              href="#executive-summary"
              className="inline-flex min-h-11 items-center justify-center rounded-full bg-signal px-5 text-sm font-semibold text-ink transition-transform hover:-translate-y-0.5"
            >
              Read the evidence
            </a>
            <a
              href="#reproducibility"
              className="inline-flex min-h-11 items-center justify-center rounded-full border border-white/25 px-5 text-sm font-semibold text-white transition-colors hover:border-white/60 hover:bg-white/5"
            >
              Run it locally
            </a>
          </div>
          <p className="mt-8 max-w-xl border-l-2 border-copper pl-4 font-mono text-[12px] leading-5 text-white/62">
            Current artifact: verified one-step compatibility adapter. Full
            selection training was interrupted and production training was not
            run.
          </p>
        </div>

        <div className="hero-enter hero-enter-delay-1">
          <EvidenceSpine />
        </div>
      </div>
      <div className="hero-enter hero-enter-delay-2 mx-auto w-full max-w-[1440px] px-5 pb-8 sm:px-8 lg:px-12 lg:pb-10">
        <MetricRail />
      </div>
    </header>
  );
}
