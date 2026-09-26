import { useEffect, useState } from "react";

export type Theme = "light" | "dark" | "system";
export type Palette = Record<
  "ink" | "ink2" | "ink3" | "grid" | "line" | "accent" | "incumbent" | "explore" | "go" | "hold" | "crit" | "surface",
  string
>;

const VARS: Record<keyof Palette, string> = {
  ink: "--ink", ink2: "--ink-2", ink3: "--ink-3", grid: "--grid", line: "--line-2", accent: "--accent",
  incumbent: "--incumbent", explore: "--explore", go: "--go", hold: "--hold", crit: "--crit", surface: "--surface",
};

// Charts get concrete colours, not var(): SVG presentation attributes do not
// resolve custom properties everywhere, and Plot computes some colours itself.
function read(): Palette {
  const cs = getComputedStyle(document.documentElement);
  return Object.fromEntries(
    Object.entries(VARS).map(([k, v]) => [k, cs.getPropertyValue(v).trim()]),
  ) as Palette;
}

const KEY = "recoup-theme";

function apply(t: Theme) {
  if (t === "system") document.documentElement.removeAttribute("data-theme");
  else document.documentElement.setAttribute("data-theme", t);
}

export function initialTheme(): Theme {
  try {
    const t = localStorage.getItem(KEY) as Theme | null;
    if (t === "light" || t === "dark") return t;
  } catch { /* storage blocked: follow the system */ }
  return "system";
}

/** Theme state + the palette charts should draw with. Re-reads on any change. */
export function useTheme(): [Theme, (t: Theme) => void, Palette] {
  const [theme, setTheme] = useState<Theme>(initialTheme);
  const [palette, setPalette] = useState<Palette>(() => { apply(theme); return read(); });

  useEffect(() => {
    apply(theme);
    try { theme === "system" ? localStorage.removeItem(KEY) : localStorage.setItem(KEY, theme); } catch { /* ignore */ }
    setPalette(read());
    const mq = matchMedia("(prefers-color-scheme: dark)");
    const on = () => setPalette(read());
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, [theme]);

  return [theme, setTheme, palette];
}
