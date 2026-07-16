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
    <figure className={`mx-auto my-24 flex w-full ${FIGURE_MAX_WIDTH} flex-col`}>
      <Image
        src={src}
        width={width}
        height={height}
        alt={alt ?? caption}
        className="h-auto w-full rounded-2xl"
      />
      <figcaption className="mt-2 font-sans text-[14px] leading-[16.8px] tracking-[0.15px] text-muted-ink">
        <strong className="font-semibold italic">{label} </strong>
        {caption}
      </figcaption>
    </figure>
  );
}
