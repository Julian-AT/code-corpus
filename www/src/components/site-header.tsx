import Image from "next/image";
import Link from "next/link";
import { GitHubIcon } from "@/components/icons";
import { MobileMenu } from "@/components/mobile-menu";
import { GITHUB_URL, PROFILE_IMAGE_URL } from "@/lib/constants";

export function SiteHeader() {
  return (
    <header className="site-navigation sticky top-0 z-50 border-b border-rule/80 bg-frost/90 backdrop-blur-xl">
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-3 focus:z-[60] focus:rounded-md focus:bg-ink focus:px-3 focus:py-2 focus:text-white"
      >
        Skip to main content
      </a>
      <div className="mx-auto flex h-16 w-full max-w-[1440px] items-center justify-between px-5 sm:px-8 lg:px-12">
        <Link
          href="/"
          aria-label="Julian local model report home"
          className="flex items-center gap-3 text-ink"
        >
          <Image
            src={PROFILE_IMAGE_URL}
            width={96}
            height={96}
            sizes="32px"
            alt="Julian Schmidt"
            className="h-8 w-8 rounded-full border border-ink/10 object-cover shadow-sm"
          />
          <span>
            <span className="block text-sm font-semibold leading-none tracking-[-0.02em]">
              Julian Schmidt
            </span>
            <span className="mt-1 block font-mono text-[9px] uppercase tracking-[0.16em] text-muted-ink">
              Gemma 4 E4B · MLX · Ollama
            </span>
          </span>
        </Link>

        <nav aria-label="Primary" className="hidden items-center gap-7 lg:flex">
          <a href="#corpus-provenance" className="text-sm text-ink-soft hover:text-ink">
            Corpus
          </a>
          <a href="#verified-deployment" className="text-sm text-ink-soft hover:text-ink">
            Verification
          </a>
          <a href="#reproducibility" className="text-sm text-ink-soft hover:text-ink">
            Run locally
          </a>
          <a
            href={GITHUB_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex min-h-10 items-center gap-2 rounded-full border border-ink/20 px-4 text-sm font-semibold text-ink transition-colors hover:border-ink/50 hover:bg-white"
          >
            <GitHubIcon className="h-4 w-4" />
            Julian on GitHub
          </a>
        </nav>

        <MobileMenu className="lg:hidden" />
      </div>
    </header>
  );
}
