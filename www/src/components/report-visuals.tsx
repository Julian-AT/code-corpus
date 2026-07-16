import type { ReactNode } from "react";
import { FIGURE_MAX_WIDTH, PROSE_MAX_WIDTH } from "@/lib/constants";

export function DatasetLanguageProfile() {
  const languages = [
    {
      label: "Markdown",
      value: "6,559",
      share: "35.7%",
      width: "w-[35.7%]",
    },
    {
      label: "TypeScript",
      value: "6,170",
      share: "33.6%",
      width: "w-[33.6%]",
    },
    { label: "JSON", value: "2,828", share: "15.4%", width: "w-[15.4%]" },
    { label: "Python", value: "1,532", share: "8.3%", width: "w-[8.3%]" },
    {
      label: "Other formats",
      value: "1,272",
      share: "6.9%",
      width: "w-[6.9%]",
    },
  ] as const;

  return (
    <figure
      className={`mx-auto my-12 w-full ${FIGURE_MAX_WIDTH} rounded-2xl border border-code-border p-5 sm:p-7`}
    >
      <div className="flex flex-wrap items-end justify-between gap-3 border-b border-code-border pb-5">
        <div>
          <p className="font-sans text-[13px] font-semibold text-muted-ink">
            Published corpus composition
          </p>
          <h3 className="mt-2 font-sans text-[25px] font-semibold leading-[1.2] text-ink">
            Documentation and TypeScript account for most retained rows
          </h3>
        </div>
        <p className="font-mono text-[13px] text-muted-ink">18,361 total rows</p>
      </div>
      <div className="mt-6 space-y-5">
        {languages.map((language) => (
          <div key={language.label}>
            <div className="mb-2 grid grid-cols-[1fr_auto_auto] items-center gap-4 font-sans text-[14px]">
              <span className="text-ink">{language.label}</span>
              <span className="font-semibold tabular-nums text-ink">
                {language.value}
              </span>
              <span className="w-12 text-right tabular-nums text-muted-ink">
                {language.share}
              </span>
            </div>
            <div className="h-3 overflow-hidden rounded-full bg-code-bg">
              <div className={`h-full rounded-full bg-ink ${language.width}`} />
            </div>
          </div>
        ))}
      </div>
      <figcaption className="mt-5 font-sans text-[14px] leading-[16.8px] tracking-[0.15px] text-muted-ink">
        Row counts from the published Hugging Face revision and its matching
        <code> datasets/raw-max/statistics.json</code>. “Other formats” combines
        the remaining 31 language and repository-file labels.
      </figcaption>
    </figure>
  );
}

export function TokenHistogram() {
  const buckets = [
    { label: "0 to 255", value: "2,847", width: "w-[21.9%]" },
    { label: "256 to 511", value: "2,515", width: "w-[19.3%]" },
    { label: "512 to 1,023", value: "12,999", width: "w-full" },
  ] as const;

  return (
    <figure
      className={`mx-auto my-12 w-full ${FIGURE_MAX_WIDTH} rounded-2xl border border-code-border p-5 sm:p-7`}
    >
      <div className="flex flex-wrap items-end justify-between gap-3 border-b border-code-border pb-5">
        <div>
          <p className="font-sans text-[13px] font-semibold text-muted-ink">
            Sequence-length distribution
          </p>
          <h3 className="mt-2 font-sans text-[25px] font-semibold leading-[1.2] text-ink">
            Most rows use the upper half of the context budget
          </h3>
        </div>
        <p className="font-mono text-[13px] text-muted-ink">18,361 total rows</p>
      </div>
      <div className="mt-6 space-y-5">
        {buckets.map((bucket) => (
          <div key={bucket.label}>
            <div className="mb-2 flex items-center justify-between gap-4 font-sans text-[14px]">
              <span className="text-ink">{bucket.label} tokens</span>
              <span className="font-semibold tabular-nums text-ink">
                {bucket.value}
              </span>
            </div>
            <div className="h-3 overflow-hidden rounded-full bg-code-bg">
              <div className={`h-full rounded-full bg-ink ${bucket.width}`} />
            </div>
          </div>
        ))}
      </div>
      <figcaption className="mt-5 font-sans text-[14px] leading-[16.8px] tracking-[0.15px] text-muted-ink">
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
      className={`mx-auto my-12 grid w-full ${FIGURE_MAX_WIDTH} overflow-hidden rounded-2xl border border-code-border sm:grid-cols-2`}
    >
      {checks.map((check, index) => (
        <div
          key={check.label}
          className={`p-6 sm:p-7 ${index % 2 === 1 ? "sm:border-l" : ""} ${
            index > 1 ? "border-t" : index === 1 ? "border-t sm:border-t-0" : ""
          } border-code-border`}
        >
          <p className="font-sans text-[13px] font-semibold text-muted-ink">
            {check.label}
          </p>
          <p className="mt-4 font-sans text-[25px] font-semibold leading-[1.2] text-ink">
            {check.value}
          </p>
          <p className="mt-1 font-sans text-[14px] text-muted-ink">
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

export function ReportCallout({ label, children }: ReportCalloutProps) {
  return (
    <aside
      className={`mx-auto my-6 w-full ${PROSE_MAX_WIDTH} rounded-xl border border-code-border bg-code-bg p-5 sm:p-6`}
    >
      <p className="font-sans text-[13px] font-semibold text-muted-ink">{label}</p>
      <div className="mt-2 font-sans text-[16px] leading-[1.55] text-ink">
        {children}
      </div>
    </aside>
  );
}
