import Image from "next/image";
import { FIGURE_MAX_WIDTH } from "@/lib/constants";

interface FigureProps {
  src: string;
  width: number;
  height: number;
  label: string;
  caption: string;
  alt?: string;
}

export function Figure({ src, width, height, label, caption, alt }: FigureProps) {
  return (
    <figure className={`mx-auto my-16 flex w-full ${FIGURE_MAX_WIDTH} flex-col`}>
      <div className="overflow-hidden rounded-2xl border border-rule bg-white p-2 shadow-[0_22px_65px_rgba(17,35,45,0.08)] sm:p-3">
        <Image
          src={src}
          width={width}
          height={height}
          sizes="(max-width: 1040px) 100vw, 1040px"
          alt={alt ?? caption}
          className="h-auto w-full rounded-xl"
        />
      </div>
      <figcaption className="mt-3 font-mono text-[11px] leading-5 text-muted-ink">
        <strong className="font-semibold uppercase tracking-[0.08em] text-ink-soft">
          {label}{" "}
        </strong>
        {caption}
      </figcaption>
    </figure>
  );
}
