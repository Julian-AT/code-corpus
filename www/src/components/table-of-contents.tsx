"use client";

import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";

interface Section {
  id: string;
  label: string;
}

const SCROLL_THRESHOLD = 124;

export function TableOfContents({ sections }: { sections: Section[] }) {
  const [activeId, setActiveId] = useState(sections[0]?.id ?? "");

  useEffect(() => {
    if (sections.length === 0) return;

    let frame = 0;
    const updateActive = () => {
      frame = 0;
      let current = sections[0]?.id ?? "";
      for (const { id } of sections) {
        const element = document.getElementById(id);
        if (!element) continue;
        if (element.getBoundingClientRect().top <= SCROLL_THRESHOLD) current = id;
        else break;
      }
      setActiveId(current);
    };

    const onScroll = () => {
      if (frame) return;
      frame = requestAnimationFrame(updateActive);
    };

    updateActive();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      window.removeEventListener("scroll", onScroll);
      if (frame) cancelAnimationFrame(frame);
    };
  }, [sections]);

  return (
    <aside className="table-of-contents sticky top-24 hidden w-[190px] shrink-0 self-start py-20 xl:block">
      <p className="mb-4 font-mono text-[9px] uppercase tracking-[0.17em] text-muted-ink">
        On this page
      </p>
      <nav aria-label="Table of contents" className="border-l border-rule">
        {sections.map(({ id, label }) => {
          const isActive = activeId === id;
          return (
            <a
              key={id}
              href={`#${id}`}
              aria-current={isActive ? "location" : undefined}
              className={cn(
                "relative block py-2.5 pl-4 text-[12px] leading-4 transition-colors",
                isActive
                  ? "font-semibold text-cobalt before:absolute before:-left-px before:inset-y-1 before:w-0.5 before:bg-cobalt"
                  : "text-muted-ink hover:text-ink",
              )}
            >
              {label}
            </a>
          );
        })}
      </nav>
    </aside>
  );
}
