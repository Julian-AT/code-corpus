import type { ReactNode } from "react";
import type { MDXComponents } from "mdx/types";
import { CitationBlock } from "@/components/citation-block";
import { DataTable } from "@/components/data-table";
import { Figure } from "@/components/figure";
import {
  DatasetLanguageProfile,
  ReportCallout,
  TokenHistogram,
  VerificationGrid,
} from "@/components/report-visuals";
import { PROSE_MAX_WIDTH } from "@/lib/constants";

function MdxLink({
  href,
  children,
}: {
  href?: string;
  children?: ReactNode;
}) {
  const isExternal = href?.startsWith("http") ?? false;
  return (
    <a
      href={href}
      {...(isExternal
        ? { target: "_blank", rel: "noopener noreferrer" }
        : {})}
      className="underline decoration-[1.36px] underline-offset-[3.06px] transition-opacity hover:opacity-70"
    >
      {children}
    </a>
  );
}

const components: MDXComponents = {
  p: ({ children }) => (
    <div className={`mx-auto w-full ${PROSE_MAX_WIDTH}`}>
      <p className="mb-4 font-serif text-[17px] leading-[1.55] text-ink [overflow-wrap:anywhere]">
        {children}
      </p>
    </div>
  ),
  h2: ({ children, id }) => (
    <div className={`mx-auto w-full ${PROSE_MAX_WIDTH}`}>
      <h2
        id={id}
        className="mb-8 mt-16 scroll-mt-28 font-sans text-[32px] font-semibold leading-[1.2] text-ink"
      >
        {children}
      </h2>
    </div>
  ),
  h3: ({ children, id }) => (
    <div className={`mx-auto w-full ${PROSE_MAX_WIDTH}`}>
      <h3
        id={id}
        className="mb-2 mt-8 scroll-mt-28 font-sans text-[25px] font-semibold leading-[1.2] text-ink"
      >
        {children}
      </h3>
    </div>
  ),
  h4: ({ children, id }) => (
    <div className={`mx-auto w-full ${PROSE_MAX_WIDTH}`}>
      <h4
        id={id}
        className="mb-2 mt-8 scroll-mt-28 font-sans text-[19px] font-semibold leading-[1.2] text-ink"
      >
        {children}
      </h4>
    </div>
  ),
  a: MdxLink,
  ul: ({ children }) => (
    <div className={`mx-auto w-full ${PROSE_MAX_WIDTH}`}>
      <ul className="mb-4 list-disc pl-5 font-serif text-[17px] leading-[1.4] text-ink">
        {children}
      </ul>
    </div>
  ),
  ol: ({ children }) => (
    <div className={`mx-auto w-full ${PROSE_MAX_WIDTH}`}>
      <ol className="mb-4 list-decimal pl-5 font-serif text-[17px] leading-[1.4] text-ink">
        {children}
      </ol>
    </div>
  ),
  li: ({ children }) => (
    <li className="mb-3 [overflow-wrap:anywhere]">{children}</li>
  ),
  strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
  code: ({ children }) => (
    <code className="rounded bg-code-bg px-1.5 py-0.5 font-mono text-[15px] text-ink [overflow-wrap:anywhere]">
      {children}
    </code>
  ),
  Figure,
  CitationBlock,
  DataTable,
  DatasetLanguageProfile,
  ReportCallout,
  TokenHistogram,
  VerificationGrid,
};

export function useMDXComponents(passed: MDXComponents): MDXComponents {
  return { ...components, ...passed };
}
