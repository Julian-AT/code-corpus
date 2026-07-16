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
      className={`relative mx-auto my-6 w-full ${PROSE_MAX_WIDTH} overflow-hidden rounded-xl border border-code-border bg-code-bg pb-6 pl-8 pr-4 pt-8`}
    >
      <button
        type="button"
        onClick={onCopy}
        aria-label="Copy code"
        className="absolute right-4 top-4 inline-flex items-center gap-1.5 rounded px-1.5 py-0.5 font-sans text-[13px] text-ink-soft transition-colors hover:bg-ink/5"
      >
        <CopyIcon className="h-[15px] w-[11px]" />
        {copied ? "Copied" : "Copy code"}
      </button>
      <pre className="overflow-x-auto whitespace-pre-wrap [overflow-wrap:anywhere] font-mono text-[16px] leading-[20px] text-ink">
        {code}
      </pre>
    </div>
  );
}
