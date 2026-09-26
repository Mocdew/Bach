import * as Plot from "@observablehq/plot";
import { useCoverage, useDaily, useGateHistory, useOverview, type GateStatus } from "../api/client";
import { Card, Empty, ErrorState, Legend, Loading, PlotView, plotBase, usePalette } from "../components/ui";
import { day, num, pct, signedMoney, when } from "../format";
import type { Palette } from "../theme";

export const MODE_LABEL: Record<string, string> = {
  LEGACY: "legacy ladder (ε uniform)", HOLD: "HOLD (ladder + exploration)", SWITCH: "SWITCH (planner)",
};
export const modeColor = (p: Palette, m: string) =>
  m === "SWITCH" ? p.go : m === "HOLD" ? p.explore : p.incumbent;
export const policyColor = (p: Palette, k: string) =>
  ({ ladder: p.incumbent, legacy: p.ink3, explore: p.explore, planner: p.accent, constraint: p.crit } as Record<string, string>)[k] ?? p.ink2;

function GateBanner({ g }: { g: GateStatus }) {
  const go = g.decision === "SWITCH";
  const essOk = g.ess != null && g.min_ess != null && g.ess >= g.min_ess;
  return (
    <section className={`gate ${go ? "go" : ""}`} aria-label="Deployment gate">
      <div className="pill">{g.decision}</div>
      <div className="why">
        <strong>{go ? "The planner is live." : "The incumbent ladder stays live."}</strong>{" "}
        {g.reason}
        <small>
          {g.on_file
            ? <>Last gate run {when(g.generated_at)} on data to {day(g.data_as_of)} · window from {day(g.window_start)} · model {g.model_version}</>
            : <>Run <code>recoup-ops gate</code> to evaluate. Until then the plan job behaves as HOLD.</>}
        </small>
      </div>
      {g.on_file && g.delta != null && (
        <div className="ci">
          <span>Δ planner − ladder</span><span className="v">{signedMoney(g.delta)}/inv</span>
          <span>95% interval</span><span className="v">{signedMoney(g.lo)} … {signedMoney(g.hi)}</span>
          <span>ESS</span><span className="v" style={{ color: essOk ? undefined : "var(--hold)" }}>{num(g.ess)} / {num(g.min_ess)}</span>
          <span>logged</span><span className="v">{num(g.n_logged)}</span>
        </div>
      )}
    </section>
  );
}

function GateTimeline() {
  const q = useGateHistory();
  const p = usePalette();
  if (q.isPending) return <Loading />;
  if (q.error) return <ErrorState error={q.error} />;
  const pts = q.data.filter((d) => d.delta != null && d.data_as_of).map((d) => ({ ...d, t: new Date(d.data_as_of!) }));
  if (pts.length < 1) return <Empty>No gate history yet — each <code>recoup-ops gate</code> run adds a point.</Empty>;
  return (
    <>
      <PlotView deps={[q.data]} height={220} build={(w) => Plot.plot({
        ...plotBase(p, w, 220),
        x: { type: "utc", label: null, tickSize: 0 },
        y: { label: "Δ USD / invoice", grid: true, tickSize: 0 },
        marks: [
          Plot.ruleY([0], { stroke: p.ink3 }),
          Plot.ruleX(pts, { x: "t", y1: "lo", y2: "hi", stroke: (d) => (d.decision === "SWITCH" ? p.go : p.hold), strokeWidth: 3, strokeOpacity: 0.55 }),
          Plot.dot(pts, { x: "t", y: "delta", r: 4.5, fill: (d) => (d.decision === "SWITCH" ? p.go : p.hold), stroke: p.surface,
            tip: true, title: (d) => `${d.decision} · Δ ${signedMoney(d.delta)} [${signedMoney(d.lo)}, ${signedMoney(d.hi)}] · ESS ${num(d.ess)}` }),
        ],
      })} />
      <Legend items={[["HOLD", p.hold], ["SWITCH", p.go]]} />
    </>
  );
}

function CoverageChart() {
  const q = useCoverage();
  const p = usePalette();
  if (q.isPending) return <Loading />;
  if (q.error) return <ErrorState error={q.error} />;
  const modes = Object.keys(q.data.by_mode);
  if (!modes.length) return <Empty>No first-retry decisions logged yet.</Empty>;
  const rows = modes.flatMap((m) => q.data.by_mode[m].map((share, b) => ({ mode: m, label: q.data.buckets[b].label, b, share })));
  const order = q.data.buckets.map((b) => b.label);
  return (
    <>
      <PlotView deps={[q.data]} height={230} build={(w) => Plot.plot({
        ...plotBase(p, w, 230),
        marginBottom: 36,
        fx: { domain: order, label: "first retry, hours since failure", padding: 0.12 },
        x: { axis: null, domain: modes },
        y: { label: "share of first retries", tickFormat: "%", grid: true, tickSize: 0 },
        color: { domain: modes, range: modes.map((m) => modeColor(p, m)) },
        marks: [
          Plot.barY(rows, { fx: "label", x: "mode", y: "share", fill: "mode", tip: true,
            title: (d) => `${MODE_LABEL[d.mode] ?? d.mode}\n${d.label}: ${pct(d.share, 1)}` }),
          Plot.ruleY([0], { stroke: p.line }),
        ],
      })} />
      <Legend items={modes.map((m) => [`${MODE_LABEL[m] ?? m} · ${num(q.data.n_by_mode[m])}`, modeColor(p, m)])} />
    </>
  );
}

function DailyChart() {
  const q = useDaily(45);
  const p = usePalette();
  if (q.isPending) return <Loading />;
  if (q.error) return <ErrorState error={q.error} />;
  if (!q.data.length) return <Empty>No decisions in the last 45 days.</Empty>;
  const rows = q.data.map((d) => ({ ...d, t: new Date(d.day) }));
  const keys = [...new Set(rows.map((r) => r.policy))];
  return (
    <>
      <PlotView deps={[q.data]} height={200} build={(w) => Plot.plot({
        ...plotBase(p, w, 200),
        x: { type: "utc", label: null, tickSize: 0 },
        y: { label: "decisions / day", grid: true, tickSize: 0 },
        color: { domain: keys, range: keys.map((k) => policyColor(p, k)) },
        marks: [Plot.rectY(rows, Plot.stackY({ x: "t", interval: "day", y: "n", fill: "policy", tip: true,
          title: (d) => `${d.day} · ${d.policy}: ${d.n}` }))],
      })} />
      <Legend items={keys.map((k) => [k, policyColor(p, k)])} />
    </>
  );
}

export default function Overview() {
  const q = useOverview();
  if (q.isPending) return <><Loading h={110} /><Loading h={100} /></>;
  if (q.error) return <ErrorState error={q.error} />;
  const o = q.data;
  const m = o.model;
  const g = o.gate;
  const essShare = g.ess != null && g.min_ess ? Math.min(g.ess / (g.min_ess * 1.5), 1) : 0;
  return (
    <>
      <GateBanner g={g} />
      <div className="tiles">
        <div className="tile">
          <span className="eyebrow">Open invoices</span>
          <span className="big">{num(o.open_invoices)}</span>
          <span className="sub"><b>{num(o.queued_retries)}</b> retries queued · <b>{num(o.awaiting_decision)}</b> awaiting a decision</span>
        </div>
        <div className="tile">
          <span className="eyebrow">Recovered, last 30 days</span>
          <span className="big">{pct(o.recovered_30d, 1)}</span>
          <span className="sub">of <b>{num(o.invoices_30d)}</b> invoices whose dunning window closed</span>
        </div>
        <div className="tile">
          <span className="eyebrow">Exploration, last 7 days</span>
          <span className="big">{pct(o.explore_share_7d)}</span>
          <span className="sub"><b>{num(o.decisions_24h)}</b> decisions in the last 24h</span>
          {g.ess != null && g.min_ess != null && (
            <>
              <div className={`meter ${g.ess >= g.min_ess ? "ok" : ""}`} title="Effective sample size vs the gate's floor">
                <i style={{ width: `${essShare * 100}%` }} /><u style={{ left: `${(1 / 1.5) * 100}%` }} />
              </div>
              <span className="sub">gate ESS <b>{num(g.ess)}</b> of {num(g.min_ess)} needed</span>
            </>
          )}
        </div>
        <div className="tile">
          <span className="eyebrow">Model</span>
          {m ? (
            <>
              <span className="big">{m.holdout ? m.holdout.pr_auc.toFixed(3) : "—"}<small>PR-AUC</small></span>
              <span className="sub">base rate <b>{pct(m.holdout?.base_rate, 1)}</b> · P(gone) <b>{m.mean_p_gone.toFixed(2)}</b></span>
              <span className="sub faint">trained {when(m.trained_at)}</span>
            </>
          ) : <span className="sub">No model yet — run <code>recoup-ops retrain</code>.</span>}
        </div>
      </div>
      <div className="grid-2">
        <Card title="Gate over time" aside="each point is one gate run"
          note="The interval must clear zero and ESS its floor before the planner goes live. Exploration under HOLD is what narrows it.">
          <GateTimeline />
        </Card>
        <Card title="Where first retries landed" aside="by logging mode"
          note="The legacy ladder put almost all mass on one bucket. HOLD-mode exploration spreads it to the delays the planner wants — the coverage the gate needs.">
          <CoverageChart />
        </Card>
      </div>
      <Card title="Decisions per day" aside={`data to ${day(o.data_as_of)}`}>
        <DailyChart />
      </Card>
      {m && m.unmapped_codes && Object.keys(m.unmapped_codes).length > 0 && (
        <Card title="Unmapped decline codes" aside="treated as hard declines (fail closed)">
          <p className="small muted" style={{ margin: 0 }}>
            {Object.entries(m.unmapped_codes).map(([k, v]) => `${k} (${v})`).join(" · ")} — confirm each with the Bachs spec and add it to <code>DECLINE_CODE_MAP</code>.
          </p>
        </Card>
      )}
      {o.synthetic && <p className="small faint">All data on this page is synthetic — generated by <code>recoup-ops synth</code>.</p>}
    </>
  );
}
