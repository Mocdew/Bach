import { createContext, useContext, useEffect, useRef, type ReactNode } from "react";
import * as Plot from "@observablehq/plot";
import { ApiError } from "../api/client";
import type { Palette } from "../theme";

export const PaletteContext = createContext<Palette | null>(null);
export const usePalette = () => useContext(PaletteContext)!;

/** Observable Plot, re-rendered on data/size/theme change. `build` gets the
 *  container width so charts stay crisp at phone width. */
export function PlotView({ build, height = 240, deps }: {
  build: (width: number, p: Palette) => (SVGSVGElement | HTMLElement);
  height?: number;
  deps: unknown[];
}) {
  const ref = useRef<HTMLDivElement>(null);
  const palette = usePalette();
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    let width = -1;
    const render = () => {
      const w = Math.max(el.clientWidth, 280);
      // Width only: a re-render changes the height, and reacting to that
      // would feed the observer its own output.
      if (w === width) return;
      width = w;
      el.replaceChildren(build(w, palette));
    };
    render();
    const ro = new ResizeObserver(() => render());
    ro.observe(el);
    return () => ro.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [palette, ...deps]);
  return <div ref={ref} className="plot" style={{ minHeight: height }} />;
}

/** Shared Plot defaults: theme colours, mono tick labels, quiet grid. */
export function plotBase(p: Palette, width: number, height: number): Plot.PlotOptions {
  return {
    width, height, marginLeft: 48, marginBottom: 32,
    style: { background: "transparent", color: p.ink2, fontSize: "11px", fontFamily: "IBM Plex Mono, monospace" },
    x: { tickSize: 0 }, y: { grid: true, tickSize: 0 },
  };
}

export function Loading({ h = 120 }: { h?: number }) {
  return <div className="skeleton" style={{ minHeight: h }} aria-busy="true" aria-label="Loading" />;
}

export function ErrorState({ error }: { error: unknown }) {
  if (error instanceof ApiError && error.status === 503) {
    return <div className="state">{error.message}</div>;
  }
  const msg = error instanceof Error ? error.message : String(error);
  return <div className="state error">Could not load: {msg}</div>;
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="state">{children}</div>;
}

export function Card({ title, aside, children, note }: {
  title: ReactNode; aside?: ReactNode; children: ReactNode; note?: ReactNode;
}) {
  return (
    <section className="card">
      <header><h2>{title}</h2>{aside && <span className="aside">{aside}</span>}</header>
      {children}
      {note && <p className="note">{note}</p>}
    </section>
  );
}

export function Tag({ kind, children }: { kind?: string | null; children?: ReactNode }) {
  return <span className={`tag ${kind ?? ""}`}>{children ?? (kind ?? "—").replace(/_/g, " ")}</span>;
}

export function Pager({ total, offset, limit, onChange }: {
  total: number; offset: number; limit: number; onChange: (offset: number) => void;
}) {
  const end = Math.min(offset + limit, total);
  return (
    <div className="pager">
      <span>{total ? `${(offset + 1).toLocaleString()}–${end.toLocaleString()} of ${total.toLocaleString()}` : "no rows"}</span>
      <span className="grow" />
      <button disabled={offset === 0} onClick={() => onChange(Math.max(0, offset - limit))}>Previous</button>
      <button disabled={end >= total} onClick={() => onChange(offset + limit)}>Next</button>
    </div>
  );
}

export function Legend({ items }: { items: [string, string][] }) {
  return (
    <div className="legend">
      {items.map(([name, color]) => <span key={name}><i style={{ background: color }} />{name}</span>)}
    </div>
  );
}
