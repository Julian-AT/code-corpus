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
        aria-label="Open navigation"
        className={cn("rounded-md text-ink", className)}
      >
        <MenuIcon className="h-10 w-10" />
      </SheetTrigger>
      <SheetContent
        side="right"
        className="w-[320px] max-w-[88vw] border-l border-rule bg-frost px-6 py-8"
      >
        <SheetTitle className="font-mono text-[10px] uppercase tracking-[0.16em] text-muted-ink">
          Report navigation
        </SheetTitle>
        <nav aria-label="Mobile" className="mt-7 flex flex-col">
          {navItems.map((item) => (
            <a
              key={item.label}
              href={item.href}
              onClick={() => setOpen(false)}
              className="border-b border-rule py-3.5 text-[15px] font-medium text-ink-soft transition-colors hover:text-cobalt"
            >
              {item.label}
            </a>
          ))}
          <a
            href={GITHUB_URL}
            target="_blank"
            rel="noopener noreferrer"
            onClick={() => setOpen(false)}
            className="mt-8 inline-flex min-h-11 items-center justify-center gap-2 rounded-full bg-ink px-4 text-sm font-semibold text-white"
          >
            <GitHubIcon className="h-4 w-4" />
            Julian on GitHub
          </a>
        </nav>
      </SheetContent>
    </Sheet>
  );
}
