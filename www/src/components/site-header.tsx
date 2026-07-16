import Image from "next/image";
import Link from "next/link";
import { GitHubIcon } from "@/components/icons";
import { MobileMenu } from "@/components/mobile-menu";
import { GITHUB_URL, PROFILE_IMAGE_URL } from "@/lib/constants";

export function SiteHeader() {
  return (
    <header className="sticky top-0 z-[9999] mx-auto h-[68px] max-w-[1300px] bg-cream">
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded focus:bg-ink focus:px-3 focus:py-2 focus:text-cream focus:outline-none"
      >
        Skip to main content
      </a>
      <a
        href="#footer"
        className="sr-only focus:not-sr-only focus:absolute focus:left-40 focus:top-4 focus:z-50 focus:rounded focus:bg-ink focus:px-3 focus:py-2 focus:text-cream focus:outline-none"
      >
        Skip to footer
      </a>

      <div className="flex h-full w-full items-center justify-between px-5 lg:px-8">
        <Link
          href="/"
          aria-label="Julian Schmidt home"
          className="flex items-center gap-2.5 text-ink"
        >
          <Image
            src={PROFILE_IMAGE_URL}
            alt="Julian Schmidt"
            width={96}
            height={96}
            priority
            sizes="28px"
            className="h-7 w-7 rounded-full object-cover"
          />
          <span className="font-sans text-[16px] font-semibold tracking-tight">
            Julian Schmidt
          </span>
        </Link>

        <div className="hidden items-center lg:flex">
          <a
            href={GITHUB_URL}
            target="_blank"
            rel="noopener noreferrer"
            aria-label="View Julian Schmidt on GitHub"
            className="group/button inline-flex h-8 shrink-0 select-none items-center justify-center gap-2 whitespace-nowrap rounded-lg border border-ink-soft bg-primary bg-clip-padding px-4 py-2 text-sm font-medium text-primary-foreground outline-none transition-all hover:cursor-pointer hover:bg-primary/80 hover:opacity-70 focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 active:translate-y-px"
          >
            <GitHubIcon className="h-5 w-5" />
            GitHub Profile
          </a>
        </div>

        <MobileMenu className="lg:hidden" />
      </div>
    </header>
  );
}
