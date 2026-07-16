import Image from "next/image";
import Link from "next/link";
import { GitHubIcon } from "@/components/icons";
import { MobileMenu } from "@/components/mobile-menu";
import { GITHUB_URL, PROFILE_IMAGE_URL } from "@/lib/constants";
import { Button } from "./ui/button";

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
            tabIndex={-1}
          >
            <Button className="inline-flex items-center gap-2 border-ink-soft px-4 py-2 hover:cursor-pointer hover:opacity-70">
              <GitHubIcon className="h-5 w-5" />
              GitHub Repository
            </Button>
          </a>
        </div>

        <MobileMenu className="lg:hidden" />
      </div>
    </header>
  );
}
