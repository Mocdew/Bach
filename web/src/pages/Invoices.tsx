import { useEffect, useState } from "react";
import * as Plot from "@observablehq/plot";
import { useInvoice, useInvoices, type Explanation, type InvoiceDetail, type InvoiceQuery, type InvoiceRow } from "../api/client";
import { Card, Empty, ErrorState, Legend, Loading, Pager, PlotView, Tag, plotBase, usePalette } from "../components/ui";
import { DataTable, type Col } from "../components/DataTable";
import { hours, label, money, num, pct, prob, when } from "../format";
import { go } from "../router";

const LIMIT = 50;

const columns: Col<InvoiceRow>[] = [
  { header: "Invoice", accessorKey: "invoice_id", cell: (c) => <span className="mono">{c.getValue()}</span> },
  { header: "Failed", accessorKey: "failed_at", cell: (c) => <span className="mono">{when(c.getValue())}</span> },
  { header: "Status", accessorKey: "status", cell: (c) => <Tag kind={c.getValue()} /> },
  { header: "Reason", accessorKey: "decline_reason", cell: (c) => label(c.getValue()) },
  { header: "Rail", accessorKey: "rail", cell: (c) => label(c.getValue()) },
  { header: "Mkt", accessorKey: "market" },
  { header: "Amount", accessorKey: "amount_usd", meta: { align: "r" }, cell: (c) => <span className="mono">{money(c.getValue())}</span> },
  { header: "Tries", accessorKey: "attempts", meta: { align: "r" }, cell: (c) => <span className="mono">{c.getValue()}</span> },
  { header: "Next", id: "next", cell: ({ row: { original: r } }) =>
      r.next_action ? <span><Tag kind={r.next_policy ?? r.next_action}>{label(r.next_action)}</Tag>{" "}
        {r.next_action === "retry" && r.next_execute_at_h != null &&
          <span className="mono small muted">+{hours(r.next_execute_at_h - r.fail_time_h)}</span>}</span>
        : r.status === "open" ? <span className="small faint">awaiting plan</span> : null },
];

// ---------------------------------------------------------------------------

function ExplainChart({ ex }: { ex: Explanation }) {
  const p = usePalette();
  const viable = ex.curve.filter((c) => c.viable);
  if (!viable.length) return <Empty>No delay passes the constraints for this invoice.</Empty>;
  const best = ex.best_bucket;
  const order = ex.curve.map((c) => c.label);
  return (
    <>
      <PlotView deps={[ex]} height={200} build={(w) => Plot.plot({
        ...plotBase(p, w, 200),
        x: { domain: order, label: "first retry at…", tickSize: 0, padding: 0.2 },
        y: { label: "schedule value, USD", grid: true, tickSize: 0 },
        marks: [
          Plot.barY(viable, { x: "label", y: "value_usd",
            fill: (d) => (d.bucket === best ? p.accent : p.line),
            tip: true, title: (d) => `${d.label} (${hours(d.delay_h)})\nvalue ${money(d.value_usd)}\nThompson share ${pct(d.propensity, 1)}` }),
          Plot.ruleY([0], { stroke: p.ink3 }),
        ],
      })} />
      <PlotView deps={[ex]} height={110} build={(w) => Plot.plot({
        ...plotBase(p, w, 110),
        x: { domain: order, label: null, tickSize: 0, padding: 0.2 },
        y: { label: "P(chosen)", domain: [0, 1], tickFormat: "%", grid: true, ticks: 3, tickSize: 0 },
        marks: [Plot.barY(ex.curve, { x: "label", y: "propensity", fill: p.explore, fillOpacity: 0.8 })],
      })} />
      <Legend items={[["best first bucket", p.accent], ["other viable buckets", p.line], ["posterior (Thompson) probability", p.explore]]} />
    </>
  );
}

function Timeline({ d }: { d: InvoiceDetail }) {
  const p = usePalette();
  const planned = d.explanation?.as_of === "now" ? d.explanation.schedule_h : [];
  const pts = [
    ...d.attempts.map((a) => ({ h: a.delay_hours, kind: a.success ? "succeeded" : "failed", y: "attempts" })),
    ...planned.map((h) => ({ h, kind: "planned", y: "attempts" })),
  ];
  if (!pts.length) return <p className="small muted" style={{ margin: 0 }}>No retries yet.</p>;
  const color = (k: string) => (k === "succeeded" ? p.go : k === "failed" ? p.crit : p.accent);
  return (
    <>
      <PlotView deps={[d]} height={70} build={(w) => Plot.plot({
        ...plotBase(p, w, 70), marginLeft: 12, marginBottom: 26,
        x: { domain: [0, 336], label: null, ticks: [0, 24, 72, 168, 336], tickFormat: (h: number) => `${h}h`, tickSize: 0 },
        y: { axis: null },
        marks: [
          Plot.ruleY(["attempts"], { stroke: p.line }),
          Plot.dot(pts, { x: "h", y: "y", r: 6, fill: (q) => (q.kind === "planned" ? p.surface : color(q.kind)),
            stroke: (q) => color(q.kind), strokeWidth: 2, tip: true, title: (q) => `${q.kind} at +${hours(q.h)}` }),
        ],
      })} />
      <Legend items={[["failed", p.crit], ["succeeded", p.go], ["planned (not yet run)", p.accent]]} />
    </>
  );
}

function Drawer({ id, onClose }: { id: string; onClose: () => void }) {
  const q = useInvoice(id);
  useEffect(() => {
    const on = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    addEventListener("keydown", on);
    return () => removeEventListener("keydown", on);
  }, [onClose]);
  return (
    <>
      <div className="scrim" onClick={onClose} />
      <aside className="drawer" role="dialog" aria-label={`Invoice ${id}`}>
        <div className="head">
          <h2 className="mono">{id}</h2>
          {q.data && <Tag kind={q.data.invoice.status} />}
          <button onClick={onClose} aria-label="Close">Close</button>
        </div>
        {q.isPending ? <Loading h={400} /> : q.error ? <ErrorState error={q.error} /> : <Detail d={q.data} />}
      </aside>
    </>
  );
}

function Detail({ d }: { d: InvoiceDetail }) {
  const i = d.invoice;
  const ex = d.explanation;
  return (
    <>
      <dl className="facts">
        <div><dt>Amount</dt><dd className="mono">{money(i.amount_usd)}</dd></div>
        <div><dt>Decline</dt><dd>{label(i.decline_reason)} <span className="faint small">({i.reason_class})</span></dd></div>
        <div><dt>Issuer advice</dt><dd>{label(i.network_advice)}</dd></div>
        <div><dt>Rail · market</dt><dd>{label(i.rail)} · {i.market}</dd></div>
        <div><dt>Failed</dt><dd className="mono">{when(i.failed_at)}</dd></div>
        <div><dt>Customer</dt><dd className="mono">{i.customer_id}</dd></div>
        <div><dt>Tenure</dt><dd>{d.customer_tenure_days != null ? `${num(d.customer_tenure_days)} days` : "—"}</dd></div>
        <div><dt>Prior payments</dt><dd>{d.prior_successful_payments} · {d.active_in_window ? "active" : "inactive"}</dd></div>
      </dl>

      <Card title="Retries" aside={`${d.attempts.length} made`}><Timeline d={d} /></Card>

      <Card title={ex?.as_of === "now" ? "What the planner would do now" : "What the planner would have done at failure"}
        aside={ex ? `model ${ex.model_version}` : undefined}
        note={ex ? "Each bar is the value of committing to that first delay and then planning the rest optimally. The lower strip is how often a posterior draw picks each bucket — the exploration distribution." : undefined}>
        {d.explanation_error ? <Empty>{d.explanation_error}</Empty> : !ex ? <Empty>No explanation.</Empty> : (
          <div style={{ display: "grid", gap: 12 }}>
            <div className="row">
              <Tag kind={ex.action}>{label(ex.action)}</Tag>
              <span className="muted">{ex.rationale}</span>
            </div>
            {ex.action === "retry" && ex.explore_mass > 0.005 && (
              <p className="small muted" style={{ margin: 0 }}>
                The logged action is drawn from the posterior, so {pct(ex.explore_mass)} of the time it
                is <em>not</em> this best plan — that is exploration, and the logged decisions below show what was actually drawn.
              </p>
            )}
            <dl className="facts">
              <div><dt>P(gone)</dt><dd className="mono">{prob(ex.p_gone)}</dd></div>
              <div><dt>P(success), next try</dt><dd className="mono">{prob(ex.p_success)}</dd></div>
              <div><dt>Expected value</dt><dd className="mono">{money(ex.expected_value_usd)}</dd></div>
              <div><dt>Attempt</dt><dd className="mono">{ex.attempt_index + 1}</dd></div>
            </dl>
            {ex.schedule_h.length > 0 && (
              <div className="schedule" aria-label="Planned schedule">
                <span className="small muted">plan:</span>
                {ex.schedule_h.map((h, k) => (
                  <span key={k} className="row" style={{ gap: 6 }}>
                    {k > 0 && <span className="arrow">→</span>}
                    <span className={`step ${k === 0 ? "first" : ""}`}>+{hours(h)}</span>
                  </span>
                ))}
              </div>
            )}
            {ex.curve.length > 0 && <ExplainChart ex={ex} />}
          </div>
        )}
      </Card>

      <Card title="Logged decisions" aside={`${d.decisions.length}`}>
        {d.decisions.length === 0 ? <p className="small muted" style={{ margin: 0 }}>None logged for this invoice.</p> : (
          <div className="table-wrap"><table>
            <thead><tr><th>#</th><th>Decided</th><th>Action</th><th>Policy</th><th className="r">Delay</th><th className="r">Propensity</th><th>Mode</th></tr></thead>
            <tbody>{d.decisions.map((x) => (
              <tr key={x.decision_id}>
                <td className="mono">{x.attempt_index + 1}</td><td className="mono">{when(x.decided_at)}</td>
                <td><Tag kind={x.action} /></td><td><Tag kind={x.policy} /></td>
                <td className="r mono">{hours(x.delay_hours)}</td><td className="r mono">{prob(x.propensity)}</td>
                <td><Tag kind={x.mode} /></td>
              </tr>))}
            </tbody>
          </table></div>
        )}
      </Card>
    </>
  );
}

// ---------------------------------------------------------------------------

export default function Invoices({ selected }: { selected: string | null }) {
  const [q, setQ] = useState<InvoiceQuery>({ status: "open", limit: LIMIT, offset: 0 });
  const r = useInvoices(q);
  const set = (patch: Partial<InvoiceQuery>) => setQ((s) => ({ ...s, ...patch, offset: 0 }));
  return (
    <>
      <Card title="Invoices" aside={r.data ? `${num(r.data.total)} match` : undefined}
        note="Open = dunning window still running. Routed = hard decline or issuer advice forbids a retry; the customer is asked for a new payment method.">
        <div className="filters">
          <label className="small muted">status{" "}
            <select value={q.status ?? ""} onChange={(e) => set({ status: e.target.value || undefined })}>
              <option value="">all</option>{["open", "recovered", "routed", "closed"].map((s) => <option key={s}>{s}</option>)}
            </select>
          </label>
          <label className="small muted">decline class{" "}
            <select value={q.reason_class ?? ""} onChange={(e) => set({ reason_class: e.target.value || undefined })}>
              <option value="">all</option>{["balance", "infra", "opaque", "hard"].map((s) => <option key={s}>{s}</option>)}
            </select>
          </label>
          <label className="small muted">rail{" "}
            <select value={q.rail ?? ""} onChange={(e) => set({ rail: e.target.value || undefined })}>
              <option value="">all</option>{["card", "mobile_money", "bank_transfer", "stablecoin"].map((s) => <option key={s} value={s}>{label(s)}</option>)}
            </select>
          </label>
          <input type="search" placeholder="invoice or customer id…" aria-label="Search invoices"
            value={q.q ?? ""} onChange={(e) => set({ q: e.target.value || undefined })} />
        </div>
        {r.isPending ? <Loading h={300} /> : r.error ? <ErrorState error={r.error} /> : (
          <>
            <DataTable data={r.data.rows} columns={columns} rowKey={(x) => x.invoice_id}
              onRowClick={(x) => go("invoices", x.invoice_id)} />
            <Pager total={r.data.total} offset={q.offset ?? 0} limit={LIMIT} onChange={(offset) => setQ((s) => ({ ...s, offset }))} />
          </>
        )}
      </Card>
      {selected && <Drawer id={selected} onClose={() => go("invoices")} />}
    </>
  );
}
