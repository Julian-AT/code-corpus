import Link from "next/link";
import { GitHubIcon, Logomark } from "@/components/icons";
import { GITHUB_URL, PORTFOLIO_URL } from "@/lib/constants";

const columns = [
  {
    heading: "Evidence",
    links: [
      { label: "Corpus provenance", href: "#corpus-provenance" },
      { label: "Dataset construction", href: "#dataset-construction" },
      { label: "Verified deployment", href: "#verified-deployment" },
    ],
  },
  {
    heading: "Implementation",
    links: [
      { label: "Model and adapter", href: "#model-and-adapter" },
      { label: "Personalization", href: "#personalization" },
      { label: "Reproducibility", href: "#reproducibility" },
    ],
  },
  {
    heading: "Julian",
    links: [
      { label: "GitHub", href: GITHUB_URL },
      { label: "Portfolio", href: PORTFOLIO_URL },
      { label: "Limits and status", href: "#limitations" },
    ],
  },
] as const;

export function SiteFooter() {
  return (
    <footer id="footer" className="border-t border-white/10 bg-ink text-white">
      <div className="mx-auto grid w-full max-w-[1440px] gap-14 px-5 py-16 sm:px-8 lg:grid-cols-[1.1fr_1.9fr] lg:px-12 lg:py-20">
        <div>
          <Link
            href="/"
            aria-label="Julian local model report home"
            className="inline-flex items-center gap-3"
          >
            <Logomark className="h-9 w-9 text-signal" />
            <span className="text-base font-semibold">Local model report</span>
          </Link>
          <p className="mt-6 max-w-sm font-serif text-[17px] leading-7 text-white/65">
            A source-traceable record of a private code corpus, a local LoRA
            adapter, and the evidence required to call the deployment usable.
          </p>
          <a
            href={GITHUB_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="mt-7 inline-flex items-center gap-2 text-sm text-white/65 transition-colors hover:text-white"
          >
            <GitHubIcon className="h-4 w-4" />
            github.com/Julian-AT
          </a>
        </div>

        <nav
          aria-label="Footer"
          className="grid grid-cols-2 gap-x-8 gap-y-10 sm:grid-cols-3"
        >
          {columns.map((column) => (
            <div key={column.heading}>
              <h2 className="font-mono text-[10px] uppercase tracking-[0.16em] text-white/40">
                {column.heading}
              </h2>
              <ul className="mt-5 space-y-3">
                {column.links.map((link) => (
                  <li key={link.label}>
                    <a
                      href={link.href}
                      className="text-sm text-white/75 transition-colors hover:text-white"
                    >
                      {link.label}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </nav>
      </div>
      <div className="border-t border-white/10">
        <div className="mx-auto flex w-full max-w-[1440px] flex-col gap-2 px-5 py-5 font-mono text-[10px] uppercase tracking-[0.12em] text-white/38 sm:flex-row sm:items-center sm:justify-between sm:px-8 lg:px-12">
          <span>Evidence snapshot · 16 July 2026</span>
          <span>Built for local, private operation</span>
        </div>
      </div>
    </footer>
  );
}
