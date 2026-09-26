import { useMemo, useState } from "react";
import * as Plot from "@observablehq/plot";
import { useCoefficients, useModel, useModelVersions } from "../api/client";
import { Card, Empty, ErrorState, Loading, PlotView, Tag, plotBase, usePalette } from "../components/ui";
import { num, pct, when } from "../format";

function Coefficients() {
  const q = useCoefficients();
  const p = usePalette();
  const [part, setPart] = useState<"hazard" | "cure">("hazard");
  const [filter, setFilter] = useState("");
  const rows = useMemo(() => (q.data ?? [])
    .filter((c) => c.part === part && !c.term.startsWith("mch:") && c.term.toLowerCase().includes(filter.toLowerCase()))
    .map((c) => ({ ...c, lo: c.coef - 1.96 * c.se, hi: c.coef + 1.96 * c.se })), [q.data, part, filter]);
  if (q.isPending) return <Loading h={300} />;
  if (q.error) return <ErrorState error={q.error} />;
  const merchants = q.data.filter((c) => c.part === part && c.term.startsWith("mch:"));
  return (
    <>
      <div className="filters">
        <label className="small muted">component{" "}
          <select value={part} onChange={(e) => setPart(e.target.value as "hazard" | "cure")}>
            <option value="hazard">hazard — P(success | not gone)</option>
            <option value="cure">cure — P(gone)</option>
          </select>
        </label>
        <input type="search" placeholder="term, e.g. d2p or rail" aria-label="Filter terms" value={filter} onChange={(e) => setFilter(e.target.value)} />
      </div>
      {rows.length === 0 ? <Empty>No terms match.</Empty> : (
        <PlotView deps={[rows]} height={Math.max(160, rows.length * 17 + 40)} build={(w) => Plot.plot({
          ...plotBase(p, w, Math.max(160, rows.length * 17 + 40)),
          marginLeft: 130,
          y: { domain: rows.map((r) => r.term), label: null, tickSize: 0 },
          x: { label: "coefficient (standardised), ±1.96·SE", grid: true, tickSize: 0 },
          marks: [
            Plot.ruleX([0], { stroke: p.ink3 }),
            Plot.ruleY(rows, { y: "term", x1: "lo", x2: "hi", stroke: p.line, strokeWidth: 2 }),
            Plot.dot(rows, { y: "term", x: "coef", r: 3.5,
              fill: (r) => (r.lo > 0 ? p.go : r.hi < 0 ? p.crit : p.ink3),
              tip: true, title: (r) => `${r.term}\ncoef ${r.coef.toFixed(3)} ± ${(1.96 * r.se).toFixed(3)}` }),
          ],
        })} />
      )}
      <p className="note small muted">
        {merchants.length} merchant offsets hidden — ridge-pooled toward zero; a merchant with little data runs the population curve.
        Green / red: interval excludes zero. Laplace standard errors are approximate.
      </p>
    </>
  );
}

export default function Model() {
  const m = useModel();
  const v = useModelVersions();
  if (m.isPending) return <Loading h={200} />;
  if (m.error) return <ErrorState error={m.error} />;
  if (!m.data) return <Empty>No model yet. Run <code>recoup-ops retrain</code>.</Empty>;
  const s = m.data;
  return (
    <>
      <div className="tiles">
        <div className="tile"><span className="eyebrow">Holdout PR-AUC</span>
          <span className="big">{s.holdout?.pr_auc.toFixed(3) ?? "—"}</span>
          <span className="sub">base rate <b>{pct(s.holdout?.base_rate, 1)}</b> · ROC-AUC <b>{s.holdout?.roc_auc.toFixed(3) ?? "—"}</b></span></div>
        <div className="tile"><span className="eyebrow">Calibration (ECE)</span>
          <span className="big">{s.holdout?.ece.toFixed(3) ?? "—"}</span>
          <span className="sub">log-loss <b>{s.holdout?.log_loss.toFixed(3) ?? "—"}</b> on {num(s.holdout?.n_test_attempts)} attempts</span></div>
        <div className="tile"><span className="eyebrow">Mean P(gone)</span>
          <span className="big">{s.mean_p_gone.toFixed(3)}</span>
          <span className="sub">cure labels <b>{s.cure_labels?.used ? "used" : "off"}</b></span></div>
        <div className="tile"><span className="eyebrow">Training data</span>
          <span className="big">{num(s.n_invoices)}<small>invoices</small></span>
          <span className="sub"><b>{num(s.n_attempts)}</b> attempts · <b>{s.n_merchants}</b> merchants · to {when(s.data_as_of)}</span></div>
      </div>
      {(s.fit_warnings.length > 0 || !s.converged) && (
        <div className="state error" role="alert">
          {!s.converged && <div>The fit did not converge.</div>}
          {s.fit_warnings.map((w, i) => <div key={i} className="small">{w}</div>)}
        </div>
      )}
      <div className="grid-3">
        <Card title="What the model learned" note="The hazard is P(retry succeeds | customer not gone); the cure part is P(gone). d2p = days to the market's payday; le = log elapsed time; k = attempt number.">
          <Coefficients />
        </Card>
        <Card title="Versions" aside="newest first">
          {v.isPending ? <Loading /> : v.error ? <ErrorState error={v.error} /> : (
            <div className="table-wrap"><table>
              <thead><tr><th>Version</th><th className="r">PR-AUC</th><th className="r">P(gone)</th></tr></thead>
              <tbody>{v.data.map((x) => (
                <tr key={x.version}>
                  <td className="mono">{x.version} {x.current && <Tag kind="ok">current</Tag>}</td>
                  <td className="r mono">{x.pr_auc?.toFixed(3) ?? "—"}</td>
                  <td className="r mono">{x.mean_p_gone?.toFixed(3) ?? "—"}</td>
                </tr>))}
              </tbody>
            </table></div>
          )}
          <p className="note">Dispute base rates: {Object.entries(s.dispute_base).map(([r, b]) => `${r.replace("_", " ")} ${(b * 100).toFixed(2)}%`).join(" · ")}</p>
          <p className="note">Incumbent ladder: {s.ladder_h.map((h) => `${h}h`).join(" → ")}</p>
        </Card>
      </div>
    </>
  );
}
