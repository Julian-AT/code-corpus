import { Fragment, type ReactNode } from "react";
import { FIGURE_MAX_WIDTH } from "@/lib/constants";

interface DataTableProps {
  head: string[];
  rows: string[][];
  label?: string;
  caption?: string;
}

function renderCell(value: string): ReactNode {
  const parts = value.split(/(\*[^*]+\*)/g).filter(Boolean);
  return parts.map((part, index) => {
    if (part.startsWith("*") && part.endsWith("*")) {
      return (
        <strong key={index} className="font-semibold text-ink">
          {part.slice(1, -1)}
        </strong>
      );
    }
    return <Fragment key={index}>{part}</Fragment>;
  });
}

export function DataTable({ head, rows, label, caption }: DataTableProps) {
  return (
    <figure className={`mx-auto my-12 flex w-full ${FIGURE_MAX_WIDTH} flex-col`}>
      <div className="overflow-hidden rounded-2xl border border-rule bg-surface shadow-[0_18px_50px_rgba(17,35,45,0.055)]">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[640px] border-collapse text-[13px] leading-[1.45]">
            <thead>
              <tr className="border-b border-rule bg-panel/70">
                {head.map((cell, index) => (
                  <th
                    key={cell}
                    scope="col"
                    className={`px-5 py-4 font-mono text-[10px] font-semibold uppercase tracking-[0.1em] text-muted-ink ${
                      index === 0 ? "text-left" : "text-right"
                    }`}
                  >
                    {renderCell(cell)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, rowIndex) => (
                <tr
                  key={`${row[0]}-${rowIndex}`}
                  className="border-b border-rule/70 last:border-b-0 hover:bg-frost/70"
                >
                  {row.map((cell, cellIndex) => (
                    <td
                      key={`${cell}-${cellIndex}`}
                      className={`px-5 py-3.5 text-ink-soft ${
                        cellIndex === 0
                          ? "text-left font-medium text-ink"
                          : "text-right font-mono tabular-nums"
                      }`}
                    >
                      {renderCell(cell)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      {(label || caption) && (
        <figcaption className="mt-3 font-mono text-[11px] leading-5 text-muted-ink">
          {label && (
            <strong className="font-semibold uppercase tracking-[0.08em] text-ink-soft">
              {label}{" "}
            </strong>
          )}
          {caption}
        </figcaption>
      )}
    </figure>
  );
}
