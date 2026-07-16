import Image from "next/image";
import Link from "next/link";
import { GitHubIcon } from "@/components/icons";
import {
  GITHUB_URL,
  PORTFOLIO_URL,
  PROFILE_IMAGE_URL,
} from "@/lib/constants";

type FooterColumn = {
  heading: string;
  links: { label: string; href: string }[];
};

const columns: FooterColumn[] = [
  {
    heading: "Report",
    links: [
      { label: "Executive summary", href: "#executive-summary" },
      { label: "Corpus provenance", href: "#corpus-provenance" },
      { label: "Dataset construction", href: "#dataset-construction" },
    ],
  },
  {
    heading: "Implementation",
    links: [
      { label: "Model and adapter", href: "#model-and-adapter" },
      { label: "Verified deployment", href: "#verified-deployment" },
      { label: "Reproducibility", href: "#reproducibility" },
    ],
  },
  {
    heading: "Julian",
    links: [
      { label: "GitHub", href: GITHUB_URL },
      { label: "Portfolio", href: PORTFOLIO_URL },
      { label: "Limits and status", href: "#limits-and-status" },
    ],
  },
];

export function SiteFooter() {
  return (
    <footer id="footer" className="w-full bg-ink font-sans text-cream">
      <div className="mx-auto w-full max-w-[1400px] px-8 py-16 lg:px-16">
        <div className="flex flex-col gap-12 lg:flex-row lg:justify-between">
          <div className="flex flex-col gap-6">
            <Link
              href="/"
              aria-label="Julian Schmidt home"
              className="inline-flex items-center gap-2.5"
            >
              <Image
                src={PROFILE_IMAGE_URL}
                alt="Julian Schmidt"
                width={96}
                height={96}
                sizes="28px"
                className="h-7 w-7 rounded-full object-cover"
              />
              <span className="text-[16px] font-semibold tracking-tight">
                Julian Schmidt
              </span>
            </Link>
            <p className="max-w-xs text-[14px] leading-relaxed text-cream/70">
              A source-traceable record of a private code corpus, a local LoRA
              adapter, and its verified Ollama and Codex deployment.
            </p>
            <a
              href={GITHUB_URL}
              target="_blank"
              rel="noopener noreferrer"
              aria-label="View Julian Schmidt on GitHub"
              className="inline-flex w-fit items-center gap-2 text-cream/70 transition-colors hover:text-cream"
            >
              <GitHubIcon className="h-5 w-5" />
              <span className="text-[14px]">Source on GitHub</span>
            </a>
            <p className="mt-auto text-[13px] text-cream/60">
              &copy; 2026 Julian Schmidt
            </p>
          </div>

          <nav
            aria-label="Footer"
            className="grid grid-cols-2 gap-x-12 gap-y-10 sm:grid-cols-3 lg:gap-x-16"
          >
            {columns.map((column) => (
              <div key={column.heading}>
                <h3 className="mb-4 text-[13px] font-semibold uppercase tracking-wide text-cream/60">
                  {column.heading}
                </h3>
                <ul className="space-y-3">
                  {column.links.map((link) => (
                    <li key={`${column.heading}-${link.label}`}>
                      <a
                        href={link.href}
                        className="text-[14px] text-cream/90 transition-colors hover:text-cream hover:underline"
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
      </div>
    </footer>
  );
}
