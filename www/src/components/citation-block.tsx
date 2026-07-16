"use client";

import { useState } from "react";
import { CopyIcon } from "@/components/icons";
import { PROSE_MAX_WIDTH } from "@/lib/constants";

export function CitationBlock({ code }: { code: string }) {
  const [copied, setCopied] = useState(false);

  const onCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);
    }
  };

  return (
    <div
      className={`relative mx-auto my-8 w-full ${PROSE_MAX_WIDTH} overflow-hidden rounded-2xl border border-code-border bg-code-bg px-5 pb-6 pt-14 shadow-[0_18px_50px_rgba(17,35,45,0.12)] sm:px-7`}
    >
      <div className="absolute inset-x-0 top-0 flex h-10 items-center justify-between border-b border-code-border px-4">
        <span className="font-mono text-[9px] uppercase tracking-[0.16em] text-code-ink/45">
          terminal
        </span>
        <button
          type="button"
          onClick={onCopy}
          aria-label="Copy command"
          className="copy-control inline-flex min-h-8 items-center gap-1.5 rounded-md px-2 font-mono text-[10px] text-code-ink/65 transition-colors hover:bg-white/5 hover:text-code-ink"
        >
          <CopyIcon className="h-[15px] w-[11px]" />
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre className="overflow-x-auto whitespace-pre font-mono text-[13px] leading-6 text-code-ink">
        {code}
      </pre>
    </div>
  );
}
