"""Read models for the console, over the files the batch jobs write.

This is the ML side of the boundary: everything that computes lives here and
returns plain Pydantic models (``schemas.py``). ``app.py`` only routes.

Nothing here writes model state. The one mutating path is the simulation
step (synthetic datasets only), which runs the same ``recoup-ops plan`` and
``advance`` the systemd units and the demo run, under a lock, in a
background thread.

Caching is by file modification time: a request after a plan run, a retrain
or a gate run sees the new files; a request in between costs a dict lookup.
The audit log is read with pandas. If it ever outgrows memory, ``_audit`` is
the one function to point at DuckDB (``read_csv_auto('decisions-*.csv')``).
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
import zlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..domain import (DT_BUCKET_EDGES_H, HARD_DECLINES, MAX_HORIZON_H, N_DT_BUCKETS,
                      NO_RETRY_ADVICE, DeclineReason, NetworkAdvice, dt_bucket)
from ..features import _reason_class, customer_history
from ..policy import RetryPolicy
from ..store import Dataset, ModelBundle, load_bundle, load_dataset, read_audit, resolve_bundle
from ..synthetic import TRUTH_DIR, advance_world, read_manifest
from . import schemas as S

log = logging.getLogger("recoup.api")

VERSION = "0.3.0"


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Paths:
    data: Path
    models: Path
    gate: Path
    audit: Path
    queue: Path

    @classmethod
    def from_env(cls, workdir: str | os.PathLike | None = None) -> "Paths":
        """Same defaults as ``recoup-ops``; ``workdir`` is the demo's layout
        (``<workdir>/{data,state,log}``)."""
        if workdir is not None:
            w = Path(workdir)
            data, state, logs = w / "data", w / "state", w / "log"
        else:
            data = Path(os.environ.get("RECOUP_DATA") or "data/synthetic")
            state = Path(os.environ.get("RECOUP_STATE") or "var/state")
            logs = Path(os.environ.get("RECOUP_LOG") or "var/log")
        return cls(data=data, models=state / "models", gate=state / "gate" / "decision.json",
                   audit=logs / "decisions", queue=state / "queue")

    def schema(self) -> S.Paths:
        return S.Paths(**{k: str(v) for k, v in self.__dict__.items()})


class NotReady(Exception):
    """A resource the request needs does not exist yet (no model, no data)."""


class Conflict(Exception):
    """The request cannot run now (e.g. a simulation step is in progress)."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _clean(v: Any) -> Any:
    """NaN/NaT/numpy scalars -> JSON-safe Python values."""
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, str):
        return v
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def bucket_infos() -> list[S.BucketInfo]:
    """Hours up to two days, days after: 0–2h, 2–6h, 6–24h, 24–48h, 2–3d, ..."""
    e = DT_BUCKET_EDGES_H
    out = []
    for b in range(N_DT_BUCKETS):
        lo, hi = e[b], e[b + 1]
        label = f"{lo:g}–{hi:g}h" if hi <= 48 else f"{lo / 24:g}–{hi / 24:g}d"
        out.append(S.BucketInfo(bucket=b, label=label, lo_h=lo, hi_h=hi))
    return out


BUCKETS = bucket_infos()


def _blocked(reason: str, advice: str) -> bool:
    try:
        return (DeclineReason(reason) in HARD_DECLINES
                or NetworkAdvice(advice) in NO_RETRY_ADVICE)
    except ValueError:
        return True


def _mtime_key(paths: list[Path]) -> tuple:
    key = []
    for p in paths:
        try:
            st = p.stat()
            key.append((str(p), st.st_mtime_ns, st.st_size))
        except FileNotFoundError:
            key.append((str(p), None, None))
    return tuple(key)


# ---------------------------------------------------------------------------
# the service
# ---------------------------------------------------------------------------


class Workspace:
    def __init__(self, paths: Paths):
        self.paths = paths
        self._cache: dict[str, tuple[tuple, Any]] = {}
        self._cache_lock = threading.Lock()
        self._sim_lock = threading.Lock()
        self._sim_state: dict[str, Any] = dict(busy=False, progress=None, last=None, error=None)

    # -- caching ---------------------------------------------------------------

    def _cached(self, name: str, key: tuple, build: Callable[[], Any]) -> Any:
        with self._cache_lock:
            hit = self._cache.get(name)
            if hit is not None and hit[0] == key:
                return hit[1]
        value = build()
        with self._cache_lock:
            self._cache[name] = (key, value)
        return value

    def _data_key(self) -> tuple:
        d = self.paths.data
        return _mtime_key([d / "manifest.json", d / "payments.jsonl", d / "disputes.csv",
                           d / "customer_events.csv", d / "customers.csv",
                           d / "legacy_decisions.csv"])

    def dataset(self) -> tuple[Dataset, pd.DataFrame]:
        """(dataset, invoices with customer-history columns). ~2s cold."""
        if not (self.paths.data / "payments.jsonl").exists():
            raise NotReady(f"no dataset at {self.paths.data} -- run `recoup-ops synth`")

        def build():
            ds = load_dataset(self.paths.data)
            return ds, customer_history(ds.invoices, ds.attempts)

        return self._cached("dataset", self._data_key(), build)

    def _audit_files(self) -> list[Path]:
        a = self.paths.audit
        return sorted(a.glob("decisions-*.csv")) if a.exists() else []

    def _audit_rows(self) -> int:
        """Data rows across the plan runs' audit files (header excluded)."""
        n = 0
        for f in self._audit_files():
            with open(f, "rb") as fh:
                n += max(sum(1 for _ in fh) - 1, 0)
        return n

    def audit(self) -> pd.DataFrame:
        ds, _ = self.dataset()
        key = self._data_key() + _mtime_key(self._audit_files())

        def build():
            df = read_audit(self.paths.audit, ds)
            if len(df):
                df = df.sort_values("decided_at_h", ascending=False, kind="stable")
                df["policy_group"] = np.where(df["mode"] == "LEGACY", "legacy",
                                              df["policy"].astype(str))
            return df.reset_index(drop=True)

        return self._cached("audit", key, build)

    def bundle(self) -> ModelBundle | None:
        try:
            d = resolve_bundle(self.paths.models / "current")
        except FileNotFoundError:
            return None
        return self._cached("bundle", _mtime_key([d / "bundle.pkl"]), lambda: load_bundle(d))

    def manifest(self) -> dict:
        return read_manifest(self.paths.data)

    @property
    def synthetic(self) -> bool:
        return self.manifest().get("kind") == "recoup-synthetic"

    # -- health / gate / model --------------------------------------------------

    def health(self) -> S.Health:
        m = self.manifest()
        return S.Health(ok=True, version=VERSION, synthetic=m.get("kind") == "recoup-synthetic",
                        data_as_of=m.get("as_of"), has_model=self.bundle() is not None,
                        has_gate=self.paths.gate.exists(), paths=self.paths.schema())

    def gate(self) -> S.GateStatus:
        p = self.paths.gate
        if not p.exists():
            return S.GateStatus(decision="HOLD", reason="no gate run on file yet",
                                on_file=False)
        g = json.loads(p.read_text(encoding="utf-8"))
        fields = {k: _clean(g.get(k)) for k in S.GateStatus.model_fields if k in g}
        fields["logged_by_mode"] = {k: int(v) for k, v in (g.get("logged_by_mode") or {}).items()}
        if fields.get("decision") not in ("HOLD", "SWITCH"):
            fields["decision"] = "HOLD"
        return S.GateStatus(on_file=True, **fields)

    def gate_history(self) -> list[S.GateHistoryPoint]:
        p = self.paths.gate.parent / "history.jsonl"
        if not p.exists():
            return []
        out = []
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            g = json.loads(line)
            out.append(S.GateHistoryPoint(**{k: _clean(g.get(k)) for k in
                                             S.GateHistoryPoint.model_fields if k in g}))
        return out

    def model(self) -> S.ModelSummary | None:
        b = self.bundle()
        if b is None:
            return None
        m = b.meta
        h = m.get("holdout")
        return S.ModelSummary(
            version=b.version, trained_at=m.get("trained_at", ""),
            data_as_of=m.get("data_as_of", ""), n_invoices=int(m.get("n_invoices", 0)),
            n_attempts=int(m.get("n_attempts", 0)), n_merchants=int(m.get("n_merchants", 0)),
            success_rate=float(m.get("success_rate", 0.0)),
            mean_p_gone=float(m.get("mean_p_gone", 0.0)), converged=bool(m.get("converged")),
            holdout=S.Holdout(**{k: h[k] for k in S.Holdout.model_fields}) if h else None,
            cure_labels=m.get("cure_labels") or {}, fit_warnings=m.get("fit_warnings") or [],
            dispute_base=m.get("dispute_base") or {}, ladder_h=list(b.ladder_h),
            unmapped_codes=m.get("unmapped_codes") or {})

    def model_versions(self) -> list[S.ModelVersion]:
        root = self.paths.models
        if not root.exists():
            return []
        b = self.bundle()
        cur = b.version if b else None
        out = []
        for d in sorted((p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")),
                        reverse=True):
            meta = {}
            if (d / "meta.json").exists():
                meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
            out.append(S.ModelVersion(
                version=d.name, current=d.name == cur, trained_at=meta.get("trained_at"),
                data_as_of=meta.get("data_as_of"),
                pr_auc=(meta.get("holdout") or {}).get("pr_auc"),
                mean_p_gone=meta.get("mean_p_gone")))
        return out

    def coefficients(self) -> list[S.Coefficient]:
        b = self.bundle()
        if b is None:
            raise NotReady("no model bundle -- run `recoup-ops retrain`")
        t = b.model.coef_table()
        return [S.Coefficient(part=r.part, term=r.term, coef=float(r.coef),
                              se=float(r.se) if np.isfinite(r.se) else 0.0)
                for r in t.itertuples(index=False)]

    # -- invoices ----------------------------------------------------------------

    def _invoice_frame(self) -> pd.DataFrame:
        """One row per invoice with its current status and pending action."""
        ds, inv = self.dataset()
        audit = self.audit()
        key = self._data_key() + _mtime_key(self._audit_files())

        def build():
            now = ds.as_of_h
            att = ds.attempts
            n_att = att.groupby("invoice_id").size()
            rec = att.groupby("invoice_id")["success"].max()
            f = inv.copy()
            f["attempts"] = f["invoice_id"].map(n_att).fillna(0).astype(int)
            f["recovered"] = f["invoice_id"].map(rec).fillna(0).astype(bool)
            f["elapsed_h"] = now - f["fail_time_h"]
            blocked = np.array([_blocked(r, a) for r, a in
                                zip(f["decline_reason"], f["network_advice"])])
            f["status"] = np.select(
                [f["recovered"].to_numpy(), blocked, (f["elapsed_h"] < MAX_HORIZON_H).to_numpy()],
                ["recovered", "routed", "open"], default="closed")
            f["reason_class"] = f["decline_reason"].map(_reason_class)
            # The decision for each invoice's *next* attempt, if one is logged.
            f["next_action"] = None
            f["next_execute_at_h"] = np.nan
            f["next_policy"] = None
            if len(audit):
                latest = audit.drop_duplicates(["invoice_id", "attempt_index"])
                latest = latest.set_index(["invoice_id", "attempt_index"])
                keys = list(zip(f["invoice_id"], f["attempts"]))
                hit = latest.reindex(keys)
                f["next_action"] = hit["action"].to_numpy()
                f["next_execute_at_h"] = hit["execute_at_h"].to_numpy(dtype=float)
                f["next_policy"] = hit["policy"].to_numpy()
            return f.sort_values("fail_time_h", ascending=False).reset_index(drop=True)

        return self._cached("invoices", key, build)

    def _invoice_row(self, r, ds: Dataset) -> S.InvoiceRow:
        from ..ops import _iso
        return S.InvoiceRow(
            invoice_id=r.invoice_id, merchant_id=str(r.merchant_id),
            customer_id=str(r.customer_id), market=r.market, rail=r.rail,
            decline_reason=r.decline_reason, reason_class=r.reason_class,
            network_advice=r.network_advice, amount_usd=float(r.amount_usd),
            fail_time_h=float(r.fail_time_h), failed_at=_iso(ds, r.fail_time_h),
            elapsed_h=float(r.elapsed_h), attempts=int(r.attempts), recovered=bool(r.recovered),
            status=r.status, next_action=_clean(r.next_action),
            next_execute_at_h=_clean(r.next_execute_at_h), next_policy=_clean(r.next_policy))

    def invoices(self, *, status: str | None = None, reason_class: str | None = None,
                 rail: str | None = None, q: str | None = None, offset: int = 0,
                 limit: int = 50) -> S.InvoicePage:
        ds, _ = self.dataset()
        f = self._invoice_frame()
        if status:
            f = f[f["status"] == status]
        if reason_class:
            f = f[f["reason_class"] == reason_class]
        if rail:
            f = f[f["rail"] == rail]
        if q:
            ql = q.lower()
            f = f[f["invoice_id"].str.lower().str.contains(ql, regex=False)
                  | f["customer_id"].astype(str).str.lower().str.contains(ql, regex=False)]
        page = f.iloc[offset: offset + limit]
        return S.InvoicePage(total=int(len(f)), offset=offset, limit=limit,
                             rows=[self._invoice_row(r, ds) for r in page.itertuples(index=False)])

    def invoice(self, invoice_id: str, explain: bool = True) -> S.InvoiceDetail:
        from ..ops import _iso
        ds, inv = self.dataset()
        f = self._invoice_frame()
        hit = f[f["invoice_id"] == invoice_id]
        if hit.empty:
            raise KeyError(invoice_id)
        r = next(hit.itertuples(index=False))
        att = ds.attempts[ds.attempts["invoice_id"] == invoice_id].sort_values("attempt_index")
        audit = self.audit()
        dec = audit[audit["invoice_id"] == invoice_id].sort_values(
            ["attempt_index", "decided_at_h"])
        detail = S.InvoiceDetail(
            invoice=self._invoice_row(r, ds),
            customer_tenure_days=_clean(r.customer_tenure_days),
            prior_successful_payments=int(r.prior_successful_payments),
            active_in_window=bool(r.active_in_window), plan_tier=str(r.plan_tier),
            attempts=[S.AttemptRow(attempt_index=int(a.attempt_index),
                                   attempted_at=_iso(ds, a.attempt_time_h),
                                   delay_hours=float(a.delay_hours), dt_bucket=int(a.dt_bucket),
                                   success=bool(a.success), disputed=bool(a.disputed))
                      for a in att.itertuples(index=False)],
            decisions=[self._decision_row(d) for d in dec.to_dict("records")])
        if explain:
            try:
                detail.explanation = self._explain(r, att, ds, inv)
            except NotReady as e:
                detail.explanation_error = str(e)
        return detail

    def _explain(self, r, att: pd.DataFrame, ds: Dataset, inv: pd.DataFrame) -> S.Explanation:
        """Re-run the planner for this invoice: now, if its window is open, else
        as it looked the moment it failed. Seeded per invoice, so the curve is
        stable across page loads."""
        b = self.bundle()
        if b is None:
            raise NotReady("no model bundle -- run `recoup-ops retrain`")
        one = inv[inv["invoice_id"] == r.invoice_id]
        cfg = replace(b.policy_cfg, n_propensity_samples=64)
        rng = np.random.default_rng(zlib.crc32(r.invoice_id.encode()))
        pol = RetryPolicy(b.model, one, b.support, cfg, b.rules, b.dispute, rng)
        live = r.status == "open"
        if live:
            past = tuple(att["delay_hours"].astype(float))
            d = pol.decide(r.invoice_id, len(past), past_delays=past,
                           now_elapsed_h=float(r.elapsed_h))
            k = len(past)
        else:
            d = pol.decide(r.invoice_id, 0)
            k = 0
        curve = []
        best_b, best_v = None, -np.inf
        if d.curve is not None:
            for c in d.curve.itertuples(index=False):
                viable = bool(c.propensity > 0)
                curve.append(S.CurvePoint(bucket=int(c.bucket), label=BUCKETS[int(c.bucket)].label,
                                          delay_h=float(c.delay_h),
                                          value_usd=float(c.value) if viable else None,
                                          propensity=float(c.propensity), viable=viable))
                if viable and c.value > best_v:
                    best_b, best_v = int(c.bucket), float(c.value)
        if d.action == "retry" and best_b is not None:
            # Report the planner's best plan, not the exploratory draw: the
            # draw is what the audit log already records.
            sched = d.schedules_by_bucket.get(best_b, d.schedule_h)
            delay = float(d.curve["delay_h"].iloc[best_b])
            p_first = float(d.p_success)
            ev = best_v
            rationale = (f"best of {len(sched)}-attempt schedule; P(gone)={d.p_gone:.2f}")
            explore_mass = 1.0 - float(d.curve["propensity"].iloc[best_b])
        else:
            sched, delay, p_first, ev = d.schedule_h, d.delay_hours, d.p_success, d.expected_value_usd
            rationale, explore_mass = d.rationale, 0.0
        return S.Explanation(
            as_of="now" if live else "at_failure", attempt_index=k, action=d.action,
            rationale=rationale, p_gone=float(d.p_gone), p_success=float(p_first),
            expected_value_usd=float(ev), delay_hours=_clean(delay), best_bucket=best_b,
            schedule_h=[float(h) for h in sched], explore_mass=explore_mass, curve=curve,
            model_version=b.version)

    # -- decisions ----------------------------------------------------------------

    @staticmethod
    def _decision_row(d: dict) -> S.DecisionRow:
        g = {k: _clean(d.get(k)) for k in S.DecisionRow.model_fields}
        g["attempt_index"] = int(g["attempt_index"])
        if g.get("dt_bucket") is not None:
            g["dt_bucket"] = int(g["dt_bucket"])
        for k in ("decision_id", "decided_at", "invoice_id", "action", "mode", "policy"):
            g[k] = "" if g[k] is None else str(g[k])
        if d.get("mode") == "LEGACY":
            g["policy"] = "legacy"
        return S.DecisionRow(**g)

    def decisions(self, *, mode: str | None = None, policy: str | None = None,
                  action: str | None = None, invoice_id: str | None = None,
                  attempt_index: int | None = None, offset: int = 0,
                  limit: int = 50) -> S.DecisionPage:
        a = self.audit()
        if len(a):
            if mode:
                a = a[a["mode"] == mode]
            if policy:
                a = a[a["policy_group"] == policy]
            if action:
                a = a[a["action"] == action]
            if invoice_id:
                a = a[a["invoice_id"].str.contains(invoice_id, regex=False)]
            if attempt_index is not None:
                a = a[a["attempt_index"] == attempt_index]
        facets = {c: {str(k): int(v) for k, v in a[src].value_counts().items()}
                  for c, src in (("mode", "mode"), ("policy", "policy_group"), ("action", "action"))
                  } if len(a) else {"mode": {}, "policy": {}, "action": {}}
        page = a.iloc[offset: offset + limit]
        return S.DecisionPage(total=int(len(a)), offset=offset, limit=limit,
                              rows=[self._decision_row(d) for d in page.to_dict("records")],
                              facets=facets)

    def daily(self, days: int = 60) -> list[S.DailyCount]:
        ds, _ = self.dataset()
        a = self.audit()
        if not len(a):
            return []
        a = a[a["decided_at_h"] >= ds.as_of_h - days * 24.0]
        day = pd.to_datetime(a["decided_at"], utc=True).dt.strftime("%Y-%m-%d")
        g = a.assign(day=day).groupby(["day", "policy_group"]).size()
        return [S.DailyCount(day=d, policy=p, n=int(n)) for (d, p), n in g.items()]

    def coverage(self) -> S.Coverage:
        a = self.audit()
        b = self.bundle()
        by_mode, n_by_mode = {}, {}
        if len(a):
            first = a[(a["attempt_index"] == 0) & (a["action"] == "retry") & a["dt_bucket"].notna()]
            for m, g in first.groupby("mode"):
                counts = np.bincount(g["dt_bucket"].astype(int), minlength=N_DT_BUCKETS)
                by_mode[str(m)] = (counts / max(counts.sum(), 1)).tolist()
                n_by_mode[str(m)] = int(counts.sum())
        ladder = int(dt_bucket(b.ladder_h[0])) if b is not None and b.ladder_h else None
        return S.Coverage(buckets=BUCKETS, ladder_bucket=ladder, by_mode=by_mode,
                          n_by_mode=n_by_mode)

    # -- overview -------------------------------------------------------------------

    def overview(self) -> S.Overview:
        from ..ops import _iso
        ds, _ = self.dataset()
        now = ds.as_of_h
        f = self._invoice_frame()
        a = self.audit()
        open_ = f[f["status"] == "open"]
        awaiting = open_[open_["next_action"].isna()]
        queued = open_[(open_["next_action"] == "retry") & (open_["next_execute_at_h"] > now)]
        recent = a[a["decided_at_h"] > now - 24.0] if len(a) else a
        week = a[(a["decided_at_h"] > now - 7 * 24.0) & (a["action"] == "retry")] if len(a) else a
        closed30 = f[(f["fail_time_h"] + MAX_HORIZON_H <= now)
                     & (f["fail_time_h"] + MAX_HORIZON_H > now - 30 * 24.0)]
        return S.Overview(
            data_as_of=_iso(ds, now), synthetic=self.synthetic, gate=self.gate(),
            model=self.model(), open_invoices=int(len(open_)),
            awaiting_decision=int(len(awaiting)), queued_retries=int(len(queued)),
            decisions_24h=int(len(recent)),
            explore_share_7d=(float((week["policy"] == "explore").mean()) if len(week) else None),
            invoices_30d=int(len(closed30)),
            recovered_30d=float(closed30["recovered"].mean()) if len(closed30) else 0.0)

    # -- simulation -----------------------------------------------------------------

    def sim_status(self) -> S.SimStatus:
        m = self.manifest()
        st = self._sim_state
        base = dict(busy=st["busy"], progress=st["progress"], last_step=st["last"],
                    last_error=st["error"])
        if m.get("kind") != "recoup-synthetic":
            return S.SimStatus(synthetic=False, **base)
        pending = self.paths.data / TRUTH_DIR / "pending.jsonl"
        n_pending = 0
        if pending.exists():
            with open(pending, "rb") as fh:
                n_pending = sum(1 for _ in fh)
        cut = float(m.get("synth", {}).get("history_days", 0)) * 24.0
        from ..synthetic import _iso as siso
        return S.SimStatus(synthetic=True, as_of=m.get("as_of"), as_of_h=m.get("as_of_h"),
                           cutover=siso(cut), cutover_h=cut, horizon_h=m.get("horizon_h"),
                           pending_events=n_pending, **base)

    def sim_step(self, req: S.StepRequest) -> None:
        """Start plan+advance in 12h slices on a background thread."""
        if not self.synthetic:
            raise Conflict("not a synthetic dataset: the clock is the wall clock")
        if self.bundle() is None:
            raise NotReady("no model bundle -- run `recoup-ops retrain` first")
        if not self._sim_lock.acquire(blocking=False):
            raise Conflict("a simulation step is already running")
        steps = max(1, int(math.ceil(req.hours / 12.0)))
        self._sim_state.update(busy=True, progress=S.SimProgress(done=0, total=steps),
                               error=None)

        def run():
            from ..ops import main as ops_main
            t0 = time.time()
            tot = dict(decisions=0, released_failures=0, retries=0, recovered=0, stale=0)
            try:
                p = self.paths
                for i in range(steps):
                    hours = req.hours / steps
                    if req.plan:
                        before = self._audit_rows()
                        rc = ops_main(["plan", "--data", str(p.data), "--model",
                                       str(p.models / "current"), "--gate", str(p.gate),
                                       "--audit", str(p.audit), "--queue", str(p.queue),
                                       "--explore-rate", str(req.explore_rate),
                                       "--propensity-samples", "32"])
                        if rc:
                            raise RuntimeError(f"plan exited {rc}")
                        tot["decisions"] += self._audit_rows() - before
                    s = advance_world(p.data, p.queue, hours)
                    for k in ("released_failures", "retries", "recovered", "stale"):
                        tot[k] += int(s[k])
                    self._sim_state["progress"] = S.SimProgress(done=i + 1, total=steps)
                self._sim_state["last"] = S.StepResult(
                    as_of=self.manifest().get("as_of", ""), steps=steps,
                    seconds=round(time.time() - t0, 1), **tot)
            except Exception as e:  # surfaced to the console, not swallowed
                log.exception("simulation step failed")
                self._sim_state["error"] = f"{type(e).__name__}: {e}"
            finally:
                self._sim_state["busy"] = False
                self._sim_lock.release()

        threading.Thread(target=run, name="recoup-sim-step", daemon=True).start()

    def score(self) -> S.Score:
        if not self.synthetic:
            raise Conflict("scoring needs the simulator's ground truth; this dataset has none")
        from ..ops import score_world
        m = self.manifest()
        cut = float(m.get("synth", {}).get("history_days", 0)) * 24.0
        key = self._data_key() + _mtime_key(self._audit_files())

        def build():
            s = score_world(self.paths.data, self.paths.audit, cut)
            s["by_mode"] = {k: S.PhaseScore(**v) for k, v in (s.get("by_mode") or {}).items()}
            return S.Score(**{k: _clean(v) if not isinstance(v, (dict, list)) else v
                              for k, v in s.items()})

        return self._cached("score", key, build)
