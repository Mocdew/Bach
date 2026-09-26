import { useEffect, useRef, useState } from "react";
import * as Plot from "@observablehq/plot";
import { useRefreshAll, useScore, useSim, useStep, type Score } from "../api/client";
import { Card, Empty, ErrorState, Legend, Loading, PlotView, plotBase, usePalette } from "../components/ui";
import { day, hours, money, num, pct, signedPct, when } from "../format";
import { MODE_LABEL, modeColor } from "./Overview";

function ScoreChart({ s }: { s: Score }) {
  const p = usePalette();
  const modes = Object.keys(s.by_mode);
  if (!modes.length) return <Empty>Not enough closed invoices per mode yet (30 needed).</Empty>;
  const rows = modes.flatMap((m) => {
    const x = s.by_mode[m];
    return [
      { mode: m, what: "realised", v: x.realised, lo: x.realised - 1.96 * x.se, hi: x.realised + 1.96 * x.se },
      { mode: m, what: "ladder (oracle)", v: x.ladder_oracle, lo: x.ladder_oracle, hi: x.ladder_oracle },
    ];
  });
  const what = ["realised", "ladder (oracle)"];
  return (
    <>
      <PlotView deps={[s]} height={220} build={(w) => Plot.plot({
        ...plotBase(p, w, 220),
        fx: { domain: modes, label: null, tickFormat: (m: string) => MODE_LABEL[m] ?? m },
        x: { axis: null, domain: what },
        y: { label: "net USD per failed invoice", grid: true, tickSize: 0 },
        marks: [
          Plot.barY(rows, { fx: "mode", x: "what", y: "v",
            fill: (d) => (d.what === "realised" ? modeColor(p, d.mode) : p.line), tip: true,
            title: (d) => `${MODE_LABEL[d.mode] ?? d.mode}\n${d.what}: ${money(d.v)}` }),
          Plot.ruleX(rows.filter((r) => r.what === "realised"), { fx: "mode", x: "what", y1: "lo", y2: "hi", stroke: p.ink, strokeWidth: 1.5 }),
          Plot.ruleY([0], { stroke: p.ink3 }),
        ],
      })} />
      <Legend items={[...modes.map((m): [string, string] => [`realised under ${m}`, modeColor(p, m)]), ["ladder, oracle expectation on the same invoices", p.line]]} />
    </>
  );
}

export default function Simulation() {
  const sim = useSim();
  const step = useStep();
  const refresh = useRefreshAll();
  const [hoursIn, setHours] = useState(24);
  const [explore, setExplore] = useState(0.2);
  const [showScore, setShowScore] = useState(false);
  const score = useScore(showScore && !sim.data?.busy);

  // When a step finishes, everything that reads job output is stale.
  const wasBusy = useRef(false);
  useEffect(() => {
    const busy = !!sim.data?.busy;
    if (wasBusy.current && !busy) refresh();
    wasBusy.current = busy;
  }, [sim.data?.busy, refresh]);

  if (sim.isPending) return <Loading h={200} />;
  if (sim.error) return <ErrorState error={sim.error} />;
  const s = sim.data;
  if (!s.synthetic) {
    return <Empty>This dataset is real: its clock is the wall clock, and the plan job runs on its systemd timer.
      The simulation controls only exist for datasets written by <code>recoup-ops synth</code>.</Empty>;
  }
  const liveDays = s.as_of_h != null && s.cutover_h != null ? (s.as_of_h - s.cutover_h) / 24 : 0;
  const left = s.horizon_h != null && s.as_of_h != null ? s.horizon_h - s.as_of_h : null;
  const err = step.error instanceof Error ? step.error.message : s.last_error;
  return (
    <>
      <div className="tiles">
        <div className="tile"><span className="eyebrow">Simulated clock</span>
          <span className="big" style={{ fontSize: 20 }}>{when(s.as_of)}</span>
          <span className="sub">cutover {day(s.cutover)} · <b>{liveDays.toFixed(1)}</b> days live</span></div>
        <div className="tile"><span className="eyebrow">World left</span>
          <span className="big">{hours(left)}</span>
          <span className="sub"><b>{num(s.pending_events)}</b> events waiting to be released</span></div>
        <div className="tile" style={{ gridColumn: "span 2" }}><span className="eyebrow">Last step</span>
          {s.last_step ? (
            <span className="sub">
              to <b>{when(s.last_step.as_of)}</b> in {s.last_step.seconds}s · <b>{num(s.last_step.decisions)}</b> decisions ·{" "}
              <b>{num(s.last_step.released_failures)}</b> new failures · <b>{num(s.last_step.retries)}</b> retries executed,{" "}
              <b>{num(s.last_step.recovered)}</b> recovered{s.last_step.stale ? ` · ${s.last_step.stale} stale` : ""}
            </span>
          ) : <span className="sub">No step run from the console yet.</span>}
        </div>
      </div>

      <Card title="Advance the world"
        note="Each 12-hour slice runs the plan job (decisions + propensities to the audit log, retries to the queue) and then executes due retries against the simulator's ground truth. Retrain and gate are separate jobs: run recoup-ops retrain / gate to refresh them.">
        <div className="row">
          <label className="small muted">by{" "}
            <select value={hoursIn} onChange={(e) => setHours(+e.target.value)} disabled={s.busy}>
              {[12, 24, 72, 168].map((h) => <option key={h} value={h}>{hours(h)}</option>)}
            </select>
          </label>
          <label className="small muted">exploration under HOLD{" "}
            <select value={explore} onChange={(e) => setExplore(+e.target.value)} disabled={s.busy}>
              {[0, 0.1, 0.2, 0.3].map((x) => <option key={x} value={x}>{pct(x)}</option>)}
            </select>
          </label>
          <button className="primary" disabled={s.busy || step.isPending || (left != null && left <= 0)}
            onClick={() => step.mutate({ hours: hoursIn, plan: true, explore_rate: explore })}>
            {s.busy ? "Running…" : "Plan and advance"}
          </button>
          {s.busy && s.progress && (
            <div style={{ flex: "1 1 200px" }} aria-live="polite">
              <div className="progress"><i style={{ width: `${(s.progress.done / s.progress.total) * 100}%` }} /></div>
              <span className="small muted">slice {s.progress.done} of {s.progress.total}</span>
            </div>
          )}
        </div>
        {err && <div className="state error" style={{ marginTop: 10 }} role="alert">{err}</div>}
      </Card>

      <Card title="What the live policy earned" aside={score.data?.window?.length ? `${day(score.data.window[0])} – ${day(score.data.window[1])}` : undefined}
        note="Realised net value per failed invoice since cutover (invoices whose dunning window has closed), against the simulator's exact expectation for the incumbent ladder on the same invoices. Realised values are noisy — the bars show ±1.96·SE.">
        {!showScore ? <button onClick={() => setShowScore(true)}>Compute score</button>
          : score.isPending ? <Loading h={200} /> : score.error ? <ErrorState error={score.error} />
          : score.data.n_invoices === 0 ? <Empty>{score.data.note ?? "No closed invoices yet."}</Empty> : (
            <>
              <div className="row" style={{ marginBottom: 10 }}>
                <span>realised <b className="mono">{money(score.data.realised_per_invoice)}</b> ± {money((score.data.realised_se ?? 0) * 1.96)}</span>
                <span className="muted">ladder oracle <b className="mono">{money(score.data.ladder_oracle_per_invoice)}</b></span>
                <span className="muted">lift <b className="mono">{signedPct(score.data.lift_vs_ladder)}</b></span>
                <span className="muted">recovery <b className="mono">{pct(score.data.recovery_rate, 1)}</b> · {score.data.attempts_per_invoice?.toFixed(2)} tries/invoice · n={num(score.data.n_invoices)}</span>
              </div>
              <ScoreChart s={score.data} />
            </>
          )}
      </Card>
    </>
  );
}
