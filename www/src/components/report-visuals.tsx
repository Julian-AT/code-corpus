import type { ReactNode } from "react";
import { FIGURE_MAX_WIDTH, PROSE_MAX_WIDTH } from "@/lib/constants";

const evidenceSteps = [
  {
    stage: "Source",
    value: "60 repositories",
    detail: "3,018 matched commits",
    caution: false,
  },
  {
    stage: "Corpus",
    value: "8,361 rows",
    detail: "4.97M lexical tokens",
    caution: false,
  },
  {
    stage: "Adapter",
    value: "19.44M trainable",
    detail: "rank 8, one verified step",
    caution: true,
  },
  {
    stage: "Package",
    value: "343 tensor pairs",
    detail: "38.9 MB GGUF LoRA",
    caution: false,
  },
  {
    stage: "Runtime",
    value: "85.58 tokens/s",
    detail: "Ollama and Codex passed",
    caution: false,
  },
] as const;

const metrics = [
  ["3,018", "personal commits"],
  ["565,518", "net lines contributed"],
  ["8,361", "retained training rows"],
  ["8.91 GB", "training peak memory"],
  ["0.301 s", "time to first token"],
] as const;

export function EvidenceSpine() {
  return (
    <section
      aria-labelledby="evidence-spine-title"
      className="rounded-[26px] border border-white/15 bg-white/[0.055] p-6 shadow-[0_24px_90px_rgba(0,0,0,0.22)] backdrop-blur-sm sm:p-8"
    >
      <div className="flex items-center justify-between gap-4 border-b border-white/12 pb-5">
        <div>
          <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-signal">
            Evidence chain
          </p>
          <h2 id="evidence-spine-title" className="mt-2 text-lg font-semibold">
            Repository to verified agent
          </h2>
        </div>
        <span className="rounded-full border border-signal/35 bg-signal/10 px-3 py-1 font-mono text-[10px] uppercase tracking-[0.12em] text-signal">
          local only
        </span>
      </div>
      <ol className="relative mt-6 space-y-0 before:absolute before:bottom-5 before:left-[7px] before:top-5 before:w-px before:bg-white/18">
        {evidenceSteps.map((step) => (
          <li
            key={step.stage}
            className="relative grid grid-cols-[16px_76px_1fr] gap-3 py-3.5"
          >
            <span
              className={`relative z-10 mt-1.5 h-[15px] w-[15px] rounded-full border-[4px] border-ink ${
                step.caution ? "bg-copper" : "bg-signal"
              }`}
            />
            <span className="pt-0.5 font-mono text-[10px] uppercase tracking-[0.14em] text-white/45">
              {step.stage}
            </span>
            <span>
              <strong className="block text-[15px] font-semibold text-white">
                {step.value}
              </strong>
              <span className="mt-1 block font-mono text-[11px] text-white/50">
                {step.detail}
              </span>
            </span>
          </li>
        ))}
      </ol>
    </section>
  );
}

export function MetricRail() {
  return (
    <dl className="grid border-y border-white/15 sm:grid-cols-2 lg:grid-cols-5">
      {metrics.map(([value, label], index) => (
        <div
          key={label}
          className={`py-5 sm:px-5 lg:py-4 ${
            index > 0 ? "sm:border-l sm:border-white/15" : ""
          } ${index % 2 === 0 ? "pr-4" : "pl-4 lg:pr-4"}`}
        >
          <dt className="font-mono text-[10px] uppercase tracking-[0.14em] text-white/42">
            {label}
          </dt>
          <dd className="mt-1.5 text-[22px] font-semibold tracking-[-0.03em] text-white">
            {value}
          </dd>
        </div>
      ))}
    </dl>
  );
}

export function TokenHistogram() {
  const buckets = [
    { label: "0 to 255", value: "1,891", width: "w-[38%]" },
    { label: "256 to 511", value: "1,495", width: "w-[30%]" },
    { label: "512 to 1,023", value: "4,975", width: "w-full" },
  ] as const;

  return (
    <figure
      className={`mx-auto my-12 w-full ${FIGURE_MAX_WIDTH} rounded-2xl border border-rule bg-surface p-5 shadow-[0_18px_50px_rgba(17,35,45,0.07)] sm:p-7`}
    >
      <div className="flex flex-wrap items-end justify-between gap-3 border-b border-rule pb-5">
        <div>
          <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-cobalt">
            Sequence-length distribution
          </p>
          <h3 className="mt-2 text-xl font-semibold tracking-[-0.02em] text-ink">
            Most rows use the upper half of the context budget
          </h3>
        </div>
        <p className="font-mono text-xs text-muted-ink">8,361 total rows</p>
      </div>
      <div className="mt-6 space-y-5">
        {buckets.map((bucket) => (
          <div key={bucket.label}>
            <div className="mb-2 flex items-center justify-between gap-4 font-mono text-[11px]">
              <span className="text-ink-soft">{bucket.label} tokens</span>
              <span className="font-semibold text-ink">{bucket.value}</span>
            </div>
            <div className="h-3 overflow-hidden rounded-full bg-panel">
              <div className={`h-full rounded-full bg-cobalt ${bucket.width}`} />
            </div>
          </div>
        ))}
      </div>
      <figcaption className="mt-5 text-sm leading-6 text-muted-ink">
        Lexical-token buckets from <code>datasets/raw-max/statistics.json</code>.
        Training caps model input at 1,024 tokens.
      </figcaption>
    </figure>
  );
}

export function VerificationGrid() {
  const checks = [
    {
      label: "Native tool call",
      value: "Passed",
      detail: "read_project_status selected",
    },
    {
      label: "Decode throughput",
      value: "85.58 tok/s",
      detail: "20 tok/s minimum",
    },
    {
      label: "First token",
      value: "0.301 s",
      detail: "5.0 s maximum",
    },
    {
      label: "Codex task",
      value: "Passed",
      detail: "edited file, pytest passed",
    },
  ] as const;

  return (
    <section
      aria-label="Deployment verification results"
      className={`mx-auto my-12 grid w-full ${FIGURE_MAX_WIDTH} overflow-hidden rounded-2xl border border-rule bg-surface sm:grid-cols-2`}
    >
      {checks.map((check, index) => (
        <div
          key={check.label}
          className={`p-6 sm:p-7 ${index % 2 === 1 ? "sm:border-l" : ""} ${
            index > 1 ? "border-t" : index === 1 ? "border-t sm:border-t-0" : ""
          } border-rule`}
        >
          <div className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.15em] text-muted-ink">
            <span className="h-2 w-2 rounded-full bg-signal ring-4 ring-signal/15" />
            {check.label}
          </div>
          <p className="mt-5 text-[28px] font-semibold tracking-[-0.04em] text-ink">
            {check.value}
          </p>
          <p className="mt-1 font-mono text-[11px] text-muted-ink">
            {check.detail}
          </p>
        </div>
      ))}
    </section>
  );
}

type ReportCalloutProps = {
  label: string;
  tone?: "info" | "warning" | "success";
  children: ReactNode;
};

export function ReportCallout({
  label,
  tone = "info",
  children,
}: ReportCalloutProps) {
  const toneClasses = {
    info: "border-cobalt/25 bg-cobalt/[0.055] text-cobalt",
    warning: "border-copper/35 bg-copper/[0.07] text-copper",
    success: "border-signal/50 bg-signal/[0.08] text-[#17726f]",
  } as const;

  return (
    <aside
      className={`mx-auto my-8 w-full ${PROSE_MAX_WIDTH} rounded-xl border p-5 sm:p-6 ${toneClasses[tone]}`}
    >
      <p className="font-mono text-[10px] font-semibold uppercase tracking-[0.16em]">
        {label}
      </p>
      <div className="mt-3 font-sans text-[15px] leading-7 text-ink-soft">
        {children}
      </div>
    </aside>
  );
}
