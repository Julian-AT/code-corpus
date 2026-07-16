import type { ReactNode } from "react";
import type { MDXComponents } from "mdx/types";
import { CitationBlock } from "@/components/citation-block";
import { DataTable } from "@/components/data-table";
import { Figure } from "@/components/figure";
import {
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
      className="font-medium text-cobalt underline decoration-cobalt/30 underline-offset-4 transition-colors hover:decoration-cobalt"
    >
      {children}
    </a>
  );
}

const components: MDXComponents = {
  p: ({ children }) => (
    <div className={`mx-auto w-full ${PROSE_MAX_WIDTH}`}>
      <p className="mb-5 font-serif text-[18px] leading-[1.68] text-ink-soft [overflow-wrap:anywhere]">
        {children}
      </p>
    </div>
  ),
  h2: ({ children, id }) => (
    <div className={`mx-auto w-full ${PROSE_MAX_WIDTH}`}>
      <div className="mt-24 flex items-center gap-3 border-t border-rule pt-7">
        <span className="h-2 w-2 rounded-full bg-cobalt" />
        <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-muted-ink">
          Report section
        </span>
      </div>
      <h2
        id={id}
        className="mb-8 mt-5 scroll-mt-28 text-[36px] font-semibold leading-[1.06] tracking-[-0.045em] text-ink sm:text-[46px]"
      >
        {children}
      </h2>
    </div>
  ),
  h3: ({ children, id }) => (
    <div className={`mx-auto w-full ${PROSE_MAX_WIDTH}`}>
      <h3
        id={id}
        className="mb-4 mt-12 scroll-mt-28 text-[24px] font-semibold leading-[1.18] tracking-[-0.025em] text-ink sm:text-[28px]"
      >
        {children}
      </h3>
    </div>
  ),
  h4: ({ children, id }) => (
    <div className={`mx-auto w-full ${PROSE_MAX_WIDTH}`}>
      <h4
        id={id}
        className="mb-3 mt-9 scroll-mt-28 font-mono text-[13px] font-semibold uppercase tracking-[0.1em] text-cobalt"
      >
        {children}
      </h4>
    </div>
  ),
  a: MdxLink,
  ul: ({ children }) => (
    <div className={`mx-auto w-full ${PROSE_MAX_WIDTH}`}>
      <ul className="mb-6 space-y-3 pl-5 font-serif text-[18px] leading-[1.55] text-ink-soft marker:text-cobalt">
        {children}
      </ul>
    </div>
  ),
  ol: ({ children }) => (
    <div className={`mx-auto w-full ${PROSE_MAX_WIDTH}`}>
      <ol className="mb-6 list-decimal space-y-3 pl-5 font-serif text-[18px] leading-[1.55] text-ink-soft marker:font-mono marker:text-cobalt">
        {children}
      </ol>
    </div>
  ),
  li: ({ children }) => <li className="pl-1 [overflow-wrap:anywhere]">{children}</li>,
  strong: ({ children }) => <strong className="font-semibold text-ink">{children}</strong>,
  code: ({ children }) => (
    <code className="rounded-md bg-panel px-1.5 py-0.5 font-mono text-[0.86em] text-ink [overflow-wrap:anywhere]">
      {children}
    </code>
  ),
  blockquote: ({ children }) => (
    <div className={`mx-auto w-full ${PROSE_MAX_WIDTH}`}>
      <blockquote className="my-8 border-l-2 border-copper pl-5 font-serif text-xl italic leading-8 text-ink">
        {children}
      </blockquote>
    </div>
  ),
  Figure,
  CitationBlock,
  DataTable,
  ReportCallout,
  TokenHistogram,
  VerificationGrid,
};

export function useMDXComponents(passed: MDXComponents): MDXComponents {
  return { ...components, ...passed };
}
