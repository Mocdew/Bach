"""Synthetic Bachs-shaped datasets, and a world that moves forward in time.

``simulator.simulate`` produces the package's own modelling tables. That is
enough to benchmark the model, and not enough to test a deployment: nothing
upstream of those tables -- the adapter, the currency conversion, the decline
code mapping, the customer join, the cure labels, the audit log -- is ever
exercised. This module renders the same ground-truth world as the *raw data a
merchant would export from Bachs*, and then lets the operations layer run
against it exactly as it would against the real thing.

A dataset directory looks like this::

    manifest.json          epoch, as_of, FX rates, generator settings
    payments.jsonl         one Bachs payment object per line (FIELD_MAP shape)
    customers.csv          customer_id, merchant_id, market, created_at, plan_tier
    disputes.csv           payment_id, created_at
    customer_events.csv    customer_id, type, created_at
    legacy_decisions.csv   the incumbent retry system's decision log, with the
                           propensities it drew from (omit with
                           ``legacy_propensities=False`` to mimic a system
                           that never recorded them)
    _truth/                the simulator's hidden state. Read ONLY by
                           ``advance_world`` and the demo scorer -- never by
                           retrain / plan / gate.

The world is split at ``as_of``. Everything before it is history, generated
under the simulator's epsilon-randomised ladder. Everything after it --
first failures, renewals, cancellations -- waits in ``_truth/pending.jsonl``
and is released by ``advance_world`` as simulated time moves. Retries of
post-cutover invoices are *not* pre-generated: ``recoup-ops plan`` decides
them, and ``advance_world`` executes the queue against the ground-truth
hazard. That closes the loop the README's rollout plan depends on --
exploration logs accumulate, the gate reads them, and whether it ever says
SWITCH is an empirical question rather than an assertion.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .domain import (DEFAULT_EPOCH, HARD_DECLINES, MARKETS, MAX_HORIZON_H, DeclineReason,
                     NetworkAdvice, Rail, dt_bucket)
from .simulator import (SimConfig, Truth, logging_policy_propensities, simulate,
                        true_dispute_prob, true_success_prob)

# Local currency per USD. Round, plausible, and deliberately not live: the
# point is that the adapter converts, not what the rate was on a given day.
FX_PER_USD: dict[str, float] = {
    "NGN": 1550.0, "KES": 129.0, "GHS": 15.5, "ZAR": 18.2, "GBP": 0.78, "USD": 1.0,
}

# Raw decline codes as a PSP would spell them. Several reasons have more than
# one spelling, which is the normal state of a real code table.
_RAW_DECLINE: dict[DeclineReason, tuple[str, ...]] = {
    DeclineReason.INSUFFICIENT_FUNDS: ("insufficient_funds", "insufficient_balance"),
    DeclineReason.WALLET_EMPTY: ("wallet_balance_low",),
    DeclineReason.LIMIT_EXCEEDED: ("limit_exceeded",),
    DeclineReason.ISSUER_UNAVAILABLE: ("issuer_unavailable",),
    DeclineReason.RAIL_TIMEOUT: ("timeout",),
    DeclineReason.PROCESSING_ERROR: ("processing_error",),
    DeclineReason.DO_NOT_HONOR: ("do_not_honor",),
    DeclineReason.STOLEN_CARD: ("stolen_card", "lost_card"),
    DeclineReason.ACCOUNT_CLOSED: ("account_closed",),
    DeclineReason.REVOKED_AUTHORIZATION: ("revoked_authorization",),
    DeclineReason.INVALID_ACCOUNT: ("invalid_account", "invalid_card"),
}
# Codes the adapter has never heard of. Only ever attached to *hard* declines,
# so failing closed on them is also the right answer -- the dataset exercises
# the unmapped-code report without making the ground truth disagree with it.
_UNMAPPED_HARD_CODES = ("card_blocked_by_issuer", "pickup_card", "security_violation")

_RAW_ADVICE: dict[NetworkAdvice, tuple[str | None, ...]] = {
    NetworkAdvice.NONE: (None,),
    NetworkAdvice.TRY_AGAIN_LATER: ("02", "visa_cat_2", "visa_cat_3"),
    NetworkAdvice.DO_NOT_RETRY: ("03", "visa_cat_1"),
    NetworkAdvice.CANCELLED_RECURRING: ("21",),
    NetworkAdvice.UPDATED_INFO_REQUIRED: ("01",),
}

_RAW_RAIL: dict[str, tuple[str, ...]] = {
    Rail.CARD.value: ("card",), Rail.MOBILE_MONEY.value: ("mobile_money",),
    Rail.BANK_TRANSFER.value: ("bank_transfer",),
    Rail.STABLECOIN.value: ("stablecoin", "crypto"),
}

TRUTH_DIR = "_truth"


@dataclass
class SynthConfig:
    """What to generate. ``sim`` owns the ground truth; the rest is rendering."""

    sim: SimConfig = field(default_factory=lambda: SimConfig(n_invoices=9000, days=240))
    # History ends here; the remainder of the simulated horizon is the future
    # that ``advance_world`` releases.
    history_days: float = 180.0
    renewal_period_days: float = 30.0
    # Share of churned customers whose leaving is observed as an event. The
    # rest just stop paying -- which is the case the cure model exists for.
    cancel_observed_share: float = 0.6
    cancel_lag_days: tuple[float, float] = (0.0, 21.0)
    deleted_share: float = 0.1            # of observed leavers: customer.deleted
    detach_noise_share: float = 0.08      # live customers who swap a card
    unmapped_hard_code_share: float = 0.15
    dispute_lag_days: tuple[float, float] = (5.0, 45.0)
    legacy_propensities: bool = True
    seed: int = 20260926


def _iso(t_h: float, epoch=DEFAULT_EPOCH) -> str:
    # Round, never truncate: floor-to-the-second turns a 24h ladder retry into
    # 23.9997h, which is the bucket below -- the ladder rungs sit exactly on
    # bucket edges.
    return (epoch + timedelta(seconds=round(float(t_h) * 3600.0))).strftime("%Y-%m-%dT%H:%M:%SZ")


def _snap(t_h):
    """Whole seconds, the resolution of an ISO timestamp."""
    return np.round(np.asarray(t_h, dtype=float) * 3600.0) / 3600.0


def _pick(rng: np.random.Generator, options: tuple):
    return options[int(rng.integers(0, len(options)))]


def _payment(*, pid: str, merchant: str, customer: str, amount_usd: float, currency: str,
             status: str, t_h: float, rail_raw: str, ref: str, code: str | None = None,
             advice: str | None = None, active: bool = False) -> dict:
    """One payment object in the ``bachs.FIELD_MAP`` shape."""
    return {
        "id": pid, "account_id": merchant, "customer": {"id": customer},
        "amount": int(round(amount_usd * FX_PER_USD[currency] * 100)),
        "currency": currency, "status": status, "created_at": _iso(t_h),
        "payment_method": {"type": rail_raw}, "failure_reason": code,
        "reference": ref, "network_advice_code": advice,
        "metadata": {"customer_active_7d": bool(active)},
    }


def _write_jsonl(path: Path, rows, mode: str = "w") -> None:
    with open(path, mode, encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def generate_dataset(out_dir: str | os.PathLike, cfg: SynthConfig | None = None) -> dict:
    """Write a synthetic Bachs export to ``out_dir``; return the manifest."""
    cfg = cfg or SynthConfig()
    out = Path(out_dir)
    (out / TRUTH_DIR).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    sim = cfg.sim
    horizon_h = sim.days * 24.0
    as_of_h = float(cfg.history_days) * 24.0
    if not 0 < as_of_h < horizon_h - MAX_HORIZON_H:
        raise ValueError(f"history_days must leave room for at least one dunning window "
                         f"before day {sim.days}: got {cfg.history_days}")

    invoices, attempts, customers = simulate(sim, return_customers=True)
    # Snap to the timestamp resolution *before* rendering, so the delays the
    # adapter reads back from ISO strings equal the ones the logging policy
    # drew -- bit for bit, including the ones sitting on bucket edges.
    invoices["fail_time_h"] = _snap(invoices["fail_time_h"])
    fail_of = invoices.set_index("invoice_id")["fail_time_h"]
    attempts["delay_hours"] = _snap(attempts["delay_hours"])
    attempts["attempt_time_h"] = (fail_of.reindex(attempts["invoice_id"]).to_numpy()
                                  + attempts["delay_hours"].to_numpy())
    customers["rail"] = customers["rail"].map(lambda r: r.value if isinstance(r, Rail) else r)

    # --- per-invoice raw codes, fixed once so retries repeat them ----------
    raw_code, raw_advice, raw_rail = {}, {}, {}
    for inv in invoices.itertuples(index=False):
        reason = DeclineReason(inv.decline_reason)
        if reason in HARD_DECLINES and rng.random() < cfg.unmapped_hard_code_share:
            raw_code[inv.invoice_id] = _pick(rng, _UNMAPPED_HARD_CODES)
        else:
            raw_code[inv.invoice_id] = _pick(rng, _RAW_DECLINE[reason])
        raw_advice[inv.invoice_id] = _pick(rng, _RAW_ADVICE[NetworkAdvice(inv.network_advice)])
        raw_rail[inv.invoice_id] = _pick(rng, _RAW_RAIL[inv.rail])

    released, pending = [], []          # (t_h, payment) / {"t_h", "kind", "obj"}

    def emit(t_h: float, obj: dict, kind: str = "payment") -> None:
        if t_h <= as_of_h:
            released.append((t_h, kind, obj))
        else:
            pending.append({"t_h": float(t_h), "kind": kind, "obj": obj})

    # --- failures and (historic) retries -----------------------------------
    att_by_inv = {k: g.sort_values("attempt_index") for k, g in attempts.groupby("invoice_id")}
    legacy, disputes = [], []
    for inv in invoices.itertuples(index=False):
        iid, ft = inv.invoice_id, float(inv.fail_time_h)
        common = dict(merchant=inv.merchant_id, customer=inv.customer_id,
                      amount_usd=float(inv.amount_usd), currency=inv.currency,
                      rail_raw=raw_rail[iid], ref=iid)
        emit(ft, _payment(pid=f"pay_{iid}_0", status="failed", t_h=ft, code=raw_code[iid],
                          advice=raw_advice[iid], active=bool(inv.active_in_window), **common))
        if ft > as_of_h:
            continue    # the future: retries are for the live policy to decide
        prev_t = ft
        for a in (att_by_inv[iid].itertuples(index=False) if iid in att_by_inv else ()):
            t = float(a.attempt_time_h)
            if t > as_of_h:
                break   # the incumbent had queued this; the cutover replaces it
            ok = bool(a.success)
            pid = f"pay_{iid}_{int(a.attempt_index) + 1}"
            emit(t, _payment(pid=pid, status="succeeded" if ok else "failed", t_h=t,
                             code=None if ok else raw_code[iid],
                             advice=None if ok else raw_advice[iid],
                             active=bool(inv.active_in_window), **common))
            if a.disputed:
                td = t + 24.0 * rng.uniform(*cfg.dispute_lag_days)
                emit(td, {"payment_id": pid, "created_at": _iso(td)}, kind="dispute")
            probs = logging_policy_propensities(sim, int(a.attempt_index))
            legacy.append(dict(
                decision_id=f"legacy-{iid}-{int(a.attempt_index)}", run_id="legacy",
                decided_at_h=prev_t, decided_at=_iso(prev_t), invoice_id=iid,
                attempt_index=int(a.attempt_index), action="retry",
                delay_hours=float(a.delay_hours), execute_at_h=t,
                dt_bucket=int(a.dt_bucket), propensity=float(a.propensity),
                propensities=json.dumps([round(float(p), 6) for p in probs]),
                mode="LEGACY", policy=f"ladder{tuple(int(x) for x in sim.ladder_h)}+eps{sim.epsilon}",
                model_version="", p_gone=np.nan, p_success=np.nan,
                expected_value_usd=np.nan, rationale="incumbent epsilon-randomised ladder",
            ))
            prev_t = t

    # --- renewals: the successful payments that make "present" labels -----
    inv_by_cust = invoices.groupby("customer_id")
    fail_times = {c: np.sort(g["fail_time_h"].to_numpy()) for c, g in inv_by_cust}
    typical_amt = inv_by_cust["amount_usd"].median().to_dict()
    period = cfg.renewal_period_days * 24.0
    cust_rows, events = [], []
    for cu in customers.itertuples(index=False):
        cid = cu.customer_id
        created_h = -24.0 * float(cu.tenure_days)
        mkt = MARKETS[cu.market]
        cust_rows.append(dict(customer_id=cid, merchant_id=cu.merchant_id, market=cu.market,
                              created_at=_iso(created_h), plan_tier=cu.plan_tier))
        end = min(float(cu.churn_time_h), horizon_h)
        ft = fail_times.get(cid, np.array([]))
        amt = float(typical_amt.get(cid, np.exp(rng.normal(3.1, 0.85))))
        rail_raw = _pick(rng, _RAW_RAIL[cu.rail])
        t, j = float(rng.uniform(0, period)), 0
        while t < end:
            # A renewal that lands on a failure *is* that failure.
            if not (ft.size and np.min(np.abs(ft - t)) < 72.0):
                emit(t, _payment(pid=f"pay_ren_{cid}_{j}", merchant=cu.merchant_id,
                                 customer=cid, amount_usd=amt, currency=mkt.currency,
                                 status="succeeded", t_h=t, rail_raw=rail_raw,
                                 ref=f"ren_{cid}_{j}", active=rng.random() < 0.6))
            t += period * float(rng.uniform(0.97, 1.03))
            j += 1
        # Leaving, observed for some. The rest simply stop paying.
        if np.isfinite(cu.churn_time_h) and rng.random() < cfg.cancel_observed_share:
            te = float(cu.churn_time_h) + 24.0 * rng.uniform(*cfg.cancel_lag_days)
            kind = "customer.deleted" if rng.random() < cfg.deleted_share else "subscription.cancelled"
            emit(te, {"customer_id": cid, "type": kind, "created_at": _iso(te)}, kind="event")
        elif rng.random() < cfg.detach_noise_share:
            te = float(rng.uniform(0, end if np.isfinite(end) else horizon_h))
            emit(te, {"customer_id": cid, "type": "payment_method.detached",
                      "created_at": _iso(te)}, kind="event")

    # --- write ----------------------------------------------------------------
    released.sort(key=lambda r: r[0])
    _write_jsonl(out / "payments.jsonl", (o for _, k, o in released if k == "payment"))
    pd.DataFrame(cust_rows).to_csv(out / "customers.csv", index=False)
    pd.DataFrame([o for _, k, o in released if k == "dispute"],
                 columns=["payment_id", "created_at"]).to_csv(out / "disputes.csv", index=False)
    pd.DataFrame([o for _, k, o in released if k == "event"],
                 columns=["customer_id", "type", "created_at"]).to_csv(
        out / "customer_events.csv", index=False)
    if cfg.legacy_propensities:
        pd.DataFrame(legacy).to_csv(out / "legacy_decisions.csv", index=False)
    elif (out / "legacy_decisions.csv").exists():
        (out / "legacy_decisions.csv").unlink()

    pending.sort(key=lambda r: r["t_h"])
    _write_jsonl(out / TRUTH_DIR / "pending.jsonl", pending)
    truth_inv = invoices.rename(columns={"_quality": "quality", "_churned": "churned"})
    truth_inv["raw_code"] = truth_inv["invoice_id"].map(raw_code)
    truth_inv["raw_advice"] = truth_inv["invoice_id"].map(raw_advice)
    truth_inv["raw_rail"] = truth_inv["invoice_id"].map(raw_rail)
    truth_inv.to_csv(out / TRUTH_DIR / "invoices.csv", index=False)
    customers.to_csv(out / TRUTH_DIR / "customers.csv", index=False)
    # Execution state of every released invoice, so ``advance_world`` can
    # validate queued retries without re-reading the payment stream.
    state = {}
    for inv in invoices.itertuples(index=False):
        if inv.fail_time_h > as_of_h:
            continue
        g = att_by_inv.get(inv.invoice_id)
        seen = g[g["attempt_time_h"] <= as_of_h] if g is not None else None
        state[inv.invoice_id] = dict(
            n=int(len(seen)) if seen is not None else 0,
            recovered=bool(seen["success"].any()) if seen is not None else False)
    _atomic_write_text(out / TRUTH_DIR / "state.json", json.dumps(state))
    (out / TRUTH_DIR / "README.txt").write_text(
        "Hidden simulator state. Nothing in retrain/plan/gate may read this directory;\n"
        "it exists so advance_world can execute retries against the ground truth and\n"
        "the demo can score what the live policy actually earned.\n", encoding="utf-8")

    manifest = dict(
        kind="recoup-synthetic", epoch=_iso(0.0), as_of=_iso(as_of_h), as_of_h=as_of_h,
        horizon_h=horizon_h, fx_to_usd={k: 1.0 / v for k, v in FX_PER_USD.items()},
        seed=cfg.seed, sim=_sim_to_dict(sim),
        synth={k: v for k, v in asdict(cfg).items() if k != "sim"},
        counts=dict(
            payments=sum(1 for _, k, _ in released if k == "payment"),
            customers=len(cust_rows),
            invoices_history=int((invoices["fail_time_h"] <= as_of_h).sum()),
            invoices_future=int((invoices["fail_time_h"] > as_of_h).sum()),
            legacy_decisions=len(legacy) if cfg.legacy_propensities else 0,
            pending_events=len(pending),
        ),
    )
    _atomic_write_text(out / "manifest.json", json.dumps(manifest, indent=1))
    return manifest


def _sim_to_dict(sim: SimConfig) -> dict:
    d = asdict(sim)
    d["ladder_h"] = list(sim.ladder_h)
    return d


def _truth_from_manifest(manifest: dict) -> Truth:
    return Truth(**manifest["sim"]["truth"])


def read_manifest(data_dir: str | os.PathLike) -> dict:
    p = Path(data_dir) / "manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def advance_world(data_dir: str | os.PathLike, queue_dir: str | os.PathLike | None,
                  hours: float) -> dict:
    """Move a synthetic dataset forward by ``hours``.

    Releases every pending event in (as_of, as_of + hours] and executes every
    queued retry due in that interval against the ground-truth hazard, in time
    order. A queued retry is executed only if it is still the invoice's next
    attempt, the invoice is unrecovered, and the dunning window is open --
    otherwise it is counted as stale and dropped, as a real executor would.
    Returns a summary of what happened.
    """
    data = Path(data_dir)
    manifest = read_manifest(data)
    if manifest.get("kind") != "recoup-synthetic":
        raise ValueError(f"{data} is not a synthetic dataset; refusing to invent outcomes")
    t0 = float(manifest["as_of_h"])
    t1 = t0 + float(hours)
    truth = _truth_from_manifest(manifest)
    tdir = data / TRUTH_DIR
    rng = np.random.default_rng([int(manifest["seed"]), int(round(t1 * 60))])

    inv = pd.read_csv(tdir / "invoices.csv").set_index("invoice_id")
    state = json.loads((tdir / "state.json").read_text(encoding="utf-8"))
    pending = _read_jsonl(tdir / "pending.jsonl")

    queue = []
    if queue_dir is not None and Path(queue_dir).exists():
        for f in sorted(Path(queue_dir).glob("*.jsonl")):
            queue.extend(q for q in _read_jsonl(f)
                         if q.get("action") == "retry" and t0 < float(q["execute_at_h"]) <= t1)

    # Interleave releases and executions on one clock: a first failure must
    # exist before a retry of it can run, and a retry's dispute lands later.
    timeline = [(float(e["t_h"]), 0, e) for e in pending if float(e["t_h"]) <= t1]
    timeline += [(float(q["execute_at_h"]), 1, q) for q in queue]
    timeline.sort(key=lambda r: (r[0], r[1]))
    keep_pending = [e for e in pending if float(e["t_h"]) > t1]

    new_payments, new_disputes, new_events, executed = [], [], [], []
    summary = dict(released_failures=0, released_renewals=0, released_events=0,
                   released_disputes=0, retries=0, recovered=0, stale=0)
    for t, kind, item in timeline:
        if kind == 0:
            k, obj = item["kind"], item["obj"]
            if k == "payment":
                new_payments.append(obj)
                if obj["status"] == "failed":
                    state[obj["reference"]] = dict(n=0, recovered=False)
                    summary["released_failures"] += 1
                else:
                    summary["released_renewals"] += 1
            elif k == "dispute":
                new_disputes.append(obj); summary["released_disputes"] += 1
            else:
                new_events.append(obj); summary["released_events"] += 1
            continue

        iid, k = str(item["invoice_id"]), int(item["attempt_index"])
        st = state.get(iid)
        if (st is None or st["recovered"] or st["n"] != k or iid not in inv.index
                or t - float(inv.at[iid, "fail_time_h"]) > MAX_HORIZON_H):
            summary["stale"] += 1
            continue
        row = inv.loc[iid]
        p = true_success_prob(
            reason=DeclineReason(row["decline_reason"]), rail=Rail(row["rail"]),
            market_code=row["market"], fail_time_h=float(row["fail_time_h"]),
            attempt_time_h=t, attempt_index=k, customer_quality=float(row["quality"]),
            amount_usd=float(row["amount_usd"]), churned=bool(row["churned"]),
            advice=NetworkAdvice(row["network_advice"]), truth=truth)
        ok = bool(rng.random() < p)
        advice = row["raw_advice"] if isinstance(row["raw_advice"], str) else None
        pid = f"pay_{iid}_{k + 1}"
        new_payments.append(_payment(
            pid=pid, merchant=row["merchant_id"], customer=row["customer_id"],
            amount_usd=float(row["amount_usd"]), currency=row["currency"],
            status="succeeded" if ok else "failed", t_h=t, rail_raw=row["raw_rail"],
            ref=iid, code=None if ok else row["raw_code"], advice=None if ok else advice,
            active=bool(row["active_in_window"])))
        disputed = ok and rng.random() < true_dispute_prob(
            attempt_index=k, elapsed_h=t - float(row["fail_time_h"]), rail=Rail(row["rail"]))
        if disputed:
            td = t + 24.0 * rng.uniform(5.0, 45.0)
            keep_pending.append({"t_h": td, "kind": "dispute",
                                 "obj": {"payment_id": pid, "created_at": _iso(td)}})
        st["n"] = k + 1
        st["recovered"] = ok
        summary["retries"] += 1
        summary["recovered"] += int(ok)
        executed.append(dict(invoice_id=iid, attempt_index=k, t_h=t, success=int(ok),
                             disputed=int(disputed), p_true=p,
                             decision_id=item.get("decision_id")))

    _write_jsonl(data / "payments.jsonl", new_payments, mode="a")
    if new_disputes:
        pd.DataFrame(new_disputes).to_csv(data / "disputes.csv", mode="a", header=False,
                                          index=False)
    if new_events:
        pd.DataFrame(new_events)[["customer_id", "type", "created_at"]].to_csv(
            data / "customer_events.csv", mode="a", header=False, index=False)
    _write_jsonl(tdir / "executed.jsonl", executed, mode="a")
    keep_pending.sort(key=lambda r: r["t_h"])
    _write_jsonl(tdir / "pending.jsonl.tmp", keep_pending)
    os.replace(tdir / "pending.jsonl.tmp", tdir / "pending.jsonl")
    _atomic_write_text(tdir / "state.json", json.dumps(state))
    manifest["as_of_h"] = t1
    manifest["as_of"] = _iso(t1)
    _atomic_write_text(data / "manifest.json", json.dumps(manifest, indent=1))
    summary["as_of"] = manifest["as_of"]
    return summary
