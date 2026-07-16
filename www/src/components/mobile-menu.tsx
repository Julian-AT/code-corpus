"use client";

import { useState } from "react";
import {
  Sheet,
  SheetContent,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { GitHubIcon, MenuIcon } from "@/components/icons";
import { GITHUB_URL } from "@/lib/constants";
import { cn } from "@/lib/utils";

const navItems = [
  { label: "Executive summary", href: "#executive-summary" },
  { label: "Corpus provenance", href: "#corpus-provenance" },
  { label: "Dataset construction", href: "#dataset-construction" },
  { label: "Model and adapter", href: "#model-and-adapter" },
  { label: "Verified deployment", href: "#verified-deployment" },
  { label: "Run locally", href: "#reproducibility" },
] as const;

export function MobileMenu({ className }: { className?: string }) {
  const [open, setOpen] = useState(false);

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetTrigger
        aria-label="Open menu"
        className={cn("lg:hidden", className)}
      >
        <MenuIcon className="h-10 w-10 text-ink" />
      </SheetTrigger>
      <SheetContent
        side="right"
        className="w-[300px] max-w-[85vw] bg-cream px-6 py-8"
      >
        <SheetTitle className="px-1 font-sans text-[13px] font-semibold uppercase tracking-wide text-muted-ink">
          Navigation
        </SheetTitle>
        <nav aria-label="Mobile" className="mt-6 flex flex-col gap-1">
          {navItems.map((item) => (
            <a
              key={item.label}
              href={item.href}
              onClick={() => setOpen(false)}
              className="rounded-lg px-3 py-3 font-sans text-[16px] text-ink-soft transition-colors hover:bg-ink/5 hover:text-ink"
            >
              {item.label}
            </a>
          ))}
          <a
            href={GITHUB_URL}
            target="_blank"
            rel="noopener noreferrer"
            aria-label="View Julian Schmidt on GitHub"
            onClick={() => setOpen(false)}
            className="mt-6 inline-flex h-11 items-center justify-center gap-2 rounded-lg border border-ink-soft px-4 font-sans text-[16px] text-ink-soft transition-opacity hover:opacity-90"
          >
            <GitHubIcon className="h-5 w-5" />
            GitHub
          </a>
        </nav>
      </SheetContent>
    </Sheet>
  );
}
