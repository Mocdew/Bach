import { useQuery } from "@tanstack/react-query";
import { api } from "./api/client";
import { PaletteContext } from "./components/ui";
import { day } from "./format";
import Decisions from "./pages/Decisions";
import Invoices from "./pages/Invoices";
import Model from "./pages/Model";
import Overview from "./pages/Overview";
import Simulation from "./pages/Simulation";
import { href, useRoute } from "./router";
import { useTheme, type Theme } from "./theme";

const PAGES = [
  ["overview", "Overview"], ["decisions", "Decisions"], ["invoices", "Invoices"],
  ["model", "Model"], ["simulation", "Simulation"],
] as const;

export default function App() {
  const route = useRoute();
  const [theme, setTheme, palette] = useTheme();
  const health = useQuery({
    queryKey: ["health"], refetchInterval: 60_000,
    queryFn: async () => (await api.GET("/api/health")).data ?? null,
  });
  const h = health.data;
  const page = PAGES.some(([k]) => k === route.page) ? route.page : "overview";
  return (
    <PaletteContext.Provider value={palette}>
      <header className="top">
        <h1>Recoup Console</h1>
        <span className="muted">failed-payment recovery</span>
        <div className="meta">
          {h?.synthetic && <span className="chip" title="Dataset written by recoup-ops synth">synthetic data</span>}
          {h?.data_as_of && <span>data to <b>{day(h.data_as_of)}</b></span>}
          {health.isError && <span style={{ color: "var(--crit)" }}>API unreachable</span>}
          <label className="small">theme{" "}
            <select value={theme} onChange={(e) => setTheme(e.target.value as Theme)} aria-label="Colour theme">
              <option value="system">system</option><option value="light">light</option><option value="dark">dark</option>
            </select>
          </label>
        </div>
      </header>
      <nav className="nav" aria-label="Sections">
        {PAGES.map(([k, name]) => (
          <a key={k} href={href(k)} aria-current={page === k ? "page" : undefined}>{name}</a>
        ))}
      </nav>
      <main>
        {page === "overview" && <Overview />}
        {page === "decisions" && <Decisions />}
        {page === "invoices" && <Invoices selected={route.id} />}
        {page === "model" && <Model />}
        {page === "simulation" && <Simulation />}
      </main>
    </PaletteContext.Provider>
  );
}
