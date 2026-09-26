import { useState } from "react";
import { useDecisions, type DecisionQuery, type DecisionRow } from "../api/client";
import { Card, ErrorState, Loading, Pager, Tag } from "../components/ui";
import { DataTable, type Col } from "../components/DataTable";
import { hours, label, money, num, prob, when } from "../format";
import { go } from "../router";

const LIMIT = 50;

const columns: Col<DecisionRow>[] = [
  { header: "Decided", accessorKey: "decided_at", cell: (c) => <span className="mono">{when(c.getValue())}</span> },
  { header: "Invoice", accessorKey: "invoice_id", cell: (c) => <span className="mono">{c.getValue()}</span> },
  { header: "#", accessorKey: "attempt_index", meta: { align: "r" }, cell: (c) => <span className="mono">{c.getValue() + 1}</span> },
  { header: "Action", accessorKey: "action", cell: (c) => <Tag kind={c.getValue()} /> },
  { header: "Policy", accessorKey: "policy", cell: (c) => <Tag kind={c.getValue()} /> },
  { header: "Delay", accessorKey: "delay_hours", meta: { align: "r" }, cell: (c) => <span className="mono">{hours(c.getValue())}</span> },
  { header: "Propensity", accessorKey: "propensity", meta: { align: "r" }, cell: (c) => <span className="mono">{prob(c.getValue())}</span> },
  { header: "P(gone)", accessorKey: "p_gone", meta: { align: "r" }, cell: (c) => <span className="mono">{prob(c.getValue())}</span> },
  { header: "EV", accessorKey: "expected_value_usd", meta: { align: "r" }, cell: (c) => <span className="mono">{money(c.getValue())}</span> },
  { header: "Mode", accessorKey: "mode", cell: (c) => <Tag kind={c.getValue()} /> },
  { header: "Why", accessorKey: "rationale", meta: { wrap: true }, cell: (c) => <span className="small muted">{c.getValue() ?? ""}</span> },
];

function Facet({ name, value, counts, onChange }: {
  name: string; value?: string | null; counts?: Record<string, number>; onChange: (v: string | undefined) => void;
}) {
  const opts = Object.entries(counts ?? {}).sort((a, b) => b[1] - a[1]);
  if (value && !opts.some(([k]) => k === value)) opts.unshift([value, 0]);
  return (
    <label className="small muted">
      {name}{" "}
      <select value={value ?? ""} onChange={(e) => onChange(e.target.value || undefined)}>
        <option value="">all</option>
        {opts.map(([k, n]) => <option key={k} value={k}>{label(k)} ({num(n)})</option>)}
      </select>
    </label>
  );
}

export default function Decisions() {
  const [q, setQ] = useState<DecisionQuery>({ limit: LIMIT, offset: 0 });
  const r = useDecisions(q);
  const set = (patch: Partial<DecisionQuery>) => setQ((s) => ({ ...s, ...patch, offset: patch.offset ?? 0 }));
  return (
    <Card title="Decision audit log"
      aside={r.data ? `${num(r.data.total)} decisions` : undefined}
      note="Every decision the plan job made, with the exact probability it was drawn with. The gate's off-policy estimate reads these propensities — a row here is evidence, not just a log line.">
      <div className="filters">
        <Facet name="mode" value={q.mode} counts={r.data?.facets.mode} onChange={(v) => set({ mode: v })} />
        <Facet name="policy" value={q.policy} counts={r.data?.facets.policy} onChange={(v) => set({ policy: v })} />
        <Facet name="action" value={q.action} counts={r.data?.facets.action} onChange={(v) => set({ action: v })} />
        <label className="small muted">attempt{" "}
          <select value={q.attempt_index ?? ""} onChange={(e) => set({ attempt_index: e.target.value === "" ? undefined : +e.target.value })}>
            <option value="">any</option>{[0, 1, 2, 3].map((k) => <option key={k} value={k}>{k + 1}</option>)}
          </select>
        </label>
        <input type="search" placeholder="invoice id…" aria-label="Filter by invoice id"
          value={q.invoice_id ?? ""} onChange={(e) => set({ invoice_id: e.target.value || undefined })} />
      </div>
      {r.isPending ? <Loading h={300} /> : r.error ? <ErrorState error={r.error} /> : (
        <>
          <DataTable data={r.data.rows} columns={columns} rowKey={(d) => d.decision_id}
            onRowClick={(d) => go("invoices", d.invoice_id)} />
          <Pager total={r.data.total} offset={q.offset ?? 0} limit={LIMIT} onChange={(offset) => setQ((s) => ({ ...s, offset }))} />
        </>
      )}
    </Card>
  );
}
