"""Bachs API adapter.

Turns a merchant's own payment history into the tables the rest of the
package expects.

IMPORTANT -- read before trusting this file
-------------------------------------------
The field names below are a *sketch*. They have not been verified against the
live API, and guessing at someone's schema is how integrations rot quietly. The
Bachs OpenAPI spec is published (96 operations) and the sandbox accepts
``sk_sandbox_`` keys, so the correct move is:

    1. pull the spec, and pin the exact field names for payments, charge
       status, payment methods/rails, subscriptions and disputes;
    2. fix the ``FIELD_MAP`` entries below against it;
    3. run ``validate_schema()`` against a sandbox pull before any training.

The decline-reason mapping is the part most likely to differ and the part that
matters most, because the hard/soft split drives a routing decision that no
amount of modelling can recover from if it is wrong. Treat every unmapped
reason as *hard* until a human confirms otherwise -- failing closed costs you a
retry, failing open costs you a network fine.

``recoup.synthetic`` writes raw objects in exactly this shape, so the whole
path from payment objects to a fitted model is exercised before the real
schema is pinned. When it is, only ``FIELD_MAP`` and the code maps change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from .domain import N_DT_BUCKETS, DeclineReason, NetworkAdvice, Rail, dt_bucket

# --- fields to pin against the OpenAPI spec --------------------------------
FIELD_MAP: dict[str, str] = {
    "payment_id": "id",
    "merchant_id": "account_id",
    "customer_id": "customer.id",
    "amount_minor": "amount",
    "currency": "currency",
    "status": "status",
    "created_at": "created_at",
    "rail": "payment_method.type",
    "decline_code": "failure_reason",
    "invoice_ref": "reference",
    # Mastercard Merchant Advice Code / Visa decline category, if the PSP
    # surfaces it. This is the issuer telling you whether and when to retry.
    "advice_code": "network_advice_code",
    # Merchant-supplied metadata: did the customer use the product in the
    # last seven days? Absent -> False. Not a Bachs field; it is whatever the
    # merchant attaches to the charge.
    "active_in_window": "metadata.customer_active_7d",
}

# Fields without which a row is unusable. ``validate_schema`` fails on these;
# the rest only warn.
REQUIRED_FIELDS = ("payment_id", "customer_id", "status", "created_at", "invoice_ref",
                   "amount_minor", "currency")

FAILED_STATUSES = frozenset({"failed", "declined"})
SUCCESS_STATUSES = frozenset({"succeeded", "paid", "completed"})

# Customer events that mean the *person* has left, as opposed to one of their
# instruments dying. These are the only source of "known gone" cure labels
# (see ``features.derive_cure_labels``). ``payment_method.detached`` is
# deliberately absent: people swap cards all the time.
GONE_EVENT_TYPES = frozenset({"subscription.cancelled", "customer.deleted"})

# Unmapped advice is NONE (no instruction), which is safe: the decline reason
# still governs the hard/soft split. Codes that forbid a retry map to advice
# values in NO_RETRY_ADVICE and win regardless of reason.
ADVICE_CODE_MAP: dict[str, NetworkAdvice] = {
    "01": NetworkAdvice.UPDATED_INFO_REQUIRED,   # MAC 01: new account info
    "02": NetworkAdvice.TRY_AGAIN_LATER,         # MAC 02: try again later
    "03": NetworkAdvice.DO_NOT_RETRY,            # MAC 03: do not try again
    "21": NetworkAdvice.CANCELLED_RECURRING,     # MAC 21: recurring cancelled
    "24": NetworkAdvice.DO_NOT_RETRY,            # MAC 24: retry after 1 hour -- treat as blocked for v1 conservatism
    "visa_cat_1": NetworkAdvice.DO_NOT_RETRY,
    "visa_cat_2": NetworkAdvice.TRY_AGAIN_LATER,
    "visa_cat_3": NetworkAdvice.TRY_AGAIN_LATER,
    "visa_cat_4": NetworkAdvice.TRY_AGAIN_LATER,
}

# Unmapped codes fall through to hard, i.e. "do not retry". Fail closed.
DECLINE_CODE_MAP: dict[str, DeclineReason] = {
    "insufficient_funds": DeclineReason.INSUFFICIENT_FUNDS,
    "insufficient_balance": DeclineReason.INSUFFICIENT_FUNDS,
    "wallet_balance_low": DeclineReason.WALLET_EMPTY,
    "limit_exceeded": DeclineReason.LIMIT_EXCEEDED,
    "issuer_unavailable": DeclineReason.ISSUER_UNAVAILABLE,
    "timeout": DeclineReason.RAIL_TIMEOUT,
    "processing_error": DeclineReason.PROCESSING_ERROR,
    "do_not_honor": DeclineReason.DO_NOT_HONOR,
    "stolen_card": DeclineReason.STOLEN_CARD,
    "lost_card": DeclineReason.STOLEN_CARD,
    "account_closed": DeclineReason.ACCOUNT_CLOSED,
    "revoked_authorization": DeclineReason.REVOKED_AUTHORIZATION,
    "invalid_account": DeclineReason.INVALID_ACCOUNT,
    "invalid_card": DeclineReason.INVALID_ACCOUNT,
}

RAIL_MAP: dict[str, Rail] = {
    "card": Rail.CARD,
    "mobile_money": Rail.MOBILE_MONEY,
    "bank_transfer": Rail.BANK_TRANSFER,
    "crypto": Rail.STABLECOIN,
    "stablecoin": Rail.STABLECOIN,
}


class UnmappedDeclineCode(Warning):
    pass


@dataclass
class AdapterReport:
    n_payments: int
    n_failed: int
    unmapped_codes: dict[str, int]
    unmapped_rails: dict[str, int]
    unconverted_currencies: dict[str, int] = field(default_factory=dict)
    n_customers_without_profile: int = 0

    def __str__(self) -> str:
        s = f"{self.n_payments:,} payments, {self.n_failed:,} failed"
        if self.unmapped_codes:
            s += f"\n  UNMAPPED decline codes (treated as hard): {self.unmapped_codes}"
        if self.unmapped_rails:
            s += f"\n  UNMAPPED rails: {self.unmapped_rails}"
        if self.unconverted_currencies:
            s += (f"\n  NO FX RATE for {self.unconverted_currencies}: amounts taken as "
                  "USD minor units, which is wrong for these rows")
        if self.n_customers_without_profile:
            s += (f"\n  {self.n_customers_without_profile:,} customer(s) missing from the "
                  "customers table: tenure unknown, market from fallback")
        return s


@dataclass
class BachsTables:
    """Everything downstream needs, in the package's own vocabulary.

    ``invoices`` / ``attempts`` are the modelling tables. ``successes`` is
    every successful payment by customer and time (cure labels read it: a
    customer who paid for anything after a dunning window closed was not
    gone). ``gone_events`` is the feed ``derive_cure_labels`` wants.
    """

    invoices: pd.DataFrame
    attempts: pd.DataFrame
    successes: pd.DataFrame
    gone_events: pd.DataFrame
    report: AdapterReport

    def __iter__(self):
        # Keeps ``invoices, attempts, report = from_payments(...)`` working.
        return iter((self.invoices, self.attempts, self.report))


def _dig(obj: dict, dotted: str) -> Any:
    cur: Any = obj
    for part in dotted.split("."):
        if cur is None:
            return None
        cur = cur.get(part) if isinstance(cur, dict) else None
    return cur


def map_advice_code(code: str | None) -> NetworkAdvice:
    if code is None:
        return NetworkAdvice.NONE
    return ADVICE_CODE_MAP.get(str(code).lower().strip(), NetworkAdvice.NONE)


def map_decline_code(code: str | None) -> DeclineReason:
    """Unknown codes are treated as hard declines. Fail closed, deliberately."""
    if code is None:
        return DeclineReason.INVALID_ACCOUNT
    return DECLINE_CODE_MAP.get(str(code).lower().strip(), DeclineReason.INVALID_ACCOUNT)


def validate_schema(payments: Iterable[dict], *, min_coverage: float = 0.95) -> list[str]:
    """Check a sample of raw payment objects against ``FIELD_MAP``.

    Returns a list of problems; empty means the sample is usable. A required
    field present on fewer than ``min_coverage`` of objects is an error
    (prefixed ``ERROR``); an optional one that is *never* present is a
    warning, because that usually means the path in ``FIELD_MAP`` is wrong
    rather than that the data is missing. Run this against a sandbox pull
    before the first retrain.
    """
    sample = list(payments)
    if not sample:
        return ["ERROR: no payment objects to validate"]
    problems = []
    for key, path in FIELD_MAP.items():
        present = sum(_dig(p, path) is not None for p in sample) / len(sample)
        if key in REQUIRED_FIELDS and present < min_coverage:
            problems.append(f"ERROR: {key} ({path}) present on {present:.0%} of objects")
        elif key not in REQUIRED_FIELDS and present == 0.0:
            problems.append(f"WARNING: {key} ({path}) never present -- check the path")
    statuses = {str(_dig(p, FIELD_MAP["status"]) or "").lower() for p in sample}
    unknown = statuses - FAILED_STATUSES - SUCCESS_STATUSES - {""}
    if unknown:
        problems.append(f"WARNING: unrecognised statuses {sorted(unknown)} are ignored")
    return problems


def _to_hours(ts, epoch: pd.Timestamp) -> np.ndarray:
    t = pd.to_datetime(pd.Series(ts), utc=True).dt.tz_localize(None)
    return ((t - epoch) / pd.Timedelta(hours=1)).to_numpy(dtype=float)


def normalise_payments(payments: Iterable[dict], *, epoch_iso: str = "2026-01-01T00:00:00Z",
                       fx_to_usd: Mapping[str, float] | None = None
                       ) -> tuple[pd.DataFrame, AdapterReport]:
    """Flatten raw payment objects into one typed row each, sorted by time.

    ``fx_to_usd`` maps a currency code to USD per unit of that currency. Every
    economic quantity downstream (attempt cost, dispute fee, EV) is in USD, so
    an unconverted NGN amount would make a $3 invoice look like a $4,500 one.
    Rows in a currency with no rate are kept, taken as USD, and reported.
    """
    fx = {k.upper(): float(v) for k, v in (fx_to_usd or {"USD": 1.0}).items()}
    fx.setdefault("USD", 1.0)
    rows, unmapped_c, unmapped_r, no_fx = [], {}, {}, {}
    for p in payments:
        code = _dig(p, FIELD_MAP["decline_code"])
        rail_raw = str(_dig(p, FIELD_MAP["rail"]) or "").lower()
        if rail_raw and rail_raw not in RAIL_MAP:
            unmapped_r[rail_raw] = unmapped_r.get(rail_raw, 0) + 1
        if code and str(code).lower() not in DECLINE_CODE_MAP:
            unmapped_c[str(code)] = unmapped_c.get(str(code), 0) + 1
        cur = str(_dig(p, FIELD_MAP["currency"]) or "USD").upper()
        if cur not in fx:
            no_fx[cur] = no_fx.get(cur, 0) + 1
        status = str(_dig(p, FIELD_MAP["status"]) or "").lower()
        rows.append(dict(
            payment_id=_dig(p, FIELD_MAP["payment_id"]),
            merchant_id=_dig(p, FIELD_MAP["merchant_id"]),
            customer_id=_dig(p, FIELD_MAP["customer_id"]),
            invoice_ref=_dig(p, FIELD_MAP["invoice_ref"]),
            status=status,
            amount_usd=float(_dig(p, FIELD_MAP["amount_minor"]) or 0) / 100.0 * fx.get(cur, 1.0),
            currency=cur,
            rail=RAIL_MAP.get(rail_raw, Rail.CARD).value,
            decline_reason=(map_decline_code(code).value if status in FAILED_STATUSES else None),
            network_advice=map_advice_code(_dig(p, FIELD_MAP["advice_code"])).value,
            active_in_window=bool(_dig(p, FIELD_MAP["active_in_window"]) or False),
            created_at=_dig(p, FIELD_MAP["created_at"]),
        ))
    df = pd.DataFrame(rows)
    if df.empty:
        return df, AdapterReport(0, 0, unmapped_c, unmapped_r, no_fx)
    df["t_h"] = _to_hours(df["created_at"], pd.Timestamp(epoch_iso).tz_localize(None))
    df = df.drop(columns=["created_at"]).sort_values("t_h", kind="stable").reset_index(drop=True)
    report = AdapterReport(len(df), int(df["status"].isin(FAILED_STATUSES).sum()),
                           unmapped_c, unmapped_r, no_fx)
    return df, report


def from_payments(
    payments: Iterable[dict] | pd.DataFrame,
    *,
    market_of_customer: Callable[[str], str] | None = None,
    epoch_iso: str = "2026-01-01T00:00:00Z",
    customers: pd.DataFrame | None = None,
    disputes: pd.DataFrame | None = None,
    customer_events: pd.DataFrame | None = None,
    fx_to_usd: Mapping[str, float] | None = None,
    default_market: str = "NG",
) -> BachsTables:
    """Build the modelling tables from Bachs payment objects.

    A failed payment opens an invoice; subsequent payments carrying the same
    ``invoice_ref`` are its retry attempts, ordered by creation time, up to
    and including the first success.

    Market is where the *customer* is -- payday timing is local -- and comes
    from ``market_of_customer`` if given, else the ``customers`` table's
    ``market`` column, else ``default_market`` (counted in the report).

    ``payments`` is raw objects or the frame ``normalise_payments`` returns.
    Optional side tables (all keyed by id, times as ISO strings):

    ``customers``        customer_id, market, created_at, plan_tier
    ``disputes``         payment_id[, created_at]  -- chargebacks on recoveries
    ``customer_events``  customer_id, type, created_at -- ``GONE_EVENT_TYPES``
                         become the ``gone_events`` table for cure labels

    ``prior_successful_payments`` is counted from the payment stream itself:
    successes by the same customer strictly before the invoice failed. That
    undercounts customers older than the export, which the tenure feature
    partly absorbs.

    Returns ``BachsTables``; unpacking it as ``invoices, attempts, report``
    still works.
    """
    epoch = pd.Timestamp(epoch_iso).tz_localize(None)
    if isinstance(payments, pd.DataFrame):
        df = payments
        report = AdapterReport(len(df), int(df["status"].isin(FAILED_STATUSES).sum())
                               if len(df) else 0, {}, {})
    else:
        df, report = normalise_payments(payments, epoch_iso=epoch_iso, fx_to_usd=fx_to_usd)

    ok = df[df["status"].isin(SUCCESS_STATUSES)] if len(df) else df
    successes = pd.DataFrame({
        "customer_id": ok["customer_id"].astype(str).to_numpy() if len(ok) else [],
        "t_h": ok["t_h"].to_numpy(dtype=float) if len(ok) else [],
    })
    att_cols = ["invoice_id", "merchant_id", "payment_id", "attempt_index", "delay_hours",
                "dt_bucket", "attempt_time_h", "success", "disputed", "propensity"]
    invoices = pd.DataFrame()
    attempts = pd.DataFrame(columns=att_cols)
    missing_profile = 0

    if len(df):
        # Only references whose *first* payment failed can be invoices; the
        # rest (renewals that went through) are most of the stream. Stable
        # sort keeps same-timestamp payments in stream order.
        g = df.sort_values(["invoice_ref", "t_h"], kind="stable").reset_index(drop=True)
        g["_k"] = g.groupby("invoice_ref", sort=False).cumcount()
        heads = g[g["_k"] == 0]
        refs = heads.loc[heads["status"].isin(FAILED_STATUSES), "invoice_ref"]
        g = g[g["invoice_ref"].isin(set(refs))]
        head = g[g["_k"] == 0].set_index("invoice_ref")
        tail = g[g["_k"] > 0].copy()

        cid = head["customer_id"].astype(str)
        ft = head["t_h"].astype(float)
        if market_of_customer is not None:
            market = cid.map(market_of_customer)
        elif customers is not None and len(customers) and "market" in customers.columns:
            market = cid.map(customers.drop_duplicates("customer_id")
                             .set_index("customer_id")["market"])
        else:
            market = pd.Series(np.nan, index=head.index)
        missing_profile = int(cid[market.isna()].nunique())
        market = market.fillna(default_market)
        created = pd.Series(np.nan, index=head.index)
        tier = pd.Series("unknown", index=head.index)
        if customers is not None and len(customers):
            c = customers.drop_duplicates("customer_id").set_index("customer_id")
            if "created_at" in c.columns:
                created = cid.map(pd.Series(_to_hours(c["created_at"], epoch), index=c.index))
            if "plan_tier" in c.columns:
                tier = cid.map(c["plan_tier"]).fillna("unknown")

        # Successes by the same customer strictly before the failure.
        succ = successes.sort_values("t_h", kind="stable")
        by_cust = {k: v.to_numpy(dtype=float) for k, v in succ.groupby("customer_id")["t_h"]}
        empty = np.empty(0)
        prior = [int(np.searchsorted(by_cust.get(c_, empty), t_, side="left"))
                 for c_, t_ in zip(cid.to_numpy(), ft.to_numpy())]

        invoices = pd.DataFrame({
            "invoice_id": head.index.astype(str), "merchant_id": head["merchant_id"].to_numpy(),
            "customer_id": cid.to_numpy(), "market": market.to_numpy(),
            "currency": head["currency"].to_numpy(), "rail": head["rail"].to_numpy(),
            "decline_reason": head["decline_reason"].to_numpy(),
            "network_advice": head["network_advice"].to_numpy(),
            "fail_time_h": ft.to_numpy(), "amount_usd": head["amount_usd"].to_numpy(dtype=float),
            "customer_tenure_days": ((ft - created) / 24.0).to_numpy(dtype=float),
            "prior_successful_payments": np.asarray(prior, dtype=int),
            "plan_tier": tier.to_numpy(),
            "active_in_window": head["active_in_window"].astype(bool).to_numpy(),
        }).sort_values("fail_time_h", kind="stable").reset_index(drop=True)

        if len(tail):
            tail["success"] = tail["status"].isin(SUCCESS_STATUSES).astype(int)
            # Keep attempts up to and including the first success.
            before = (tail.groupby("invoice_ref", sort=False)["success"].cumsum()
                      - tail["success"])
            tail = tail[before == 0]
            # Round to the timestamp resolution. The difference of two
            # hours-since-epoch floats comes out as 23.99999999999 for a 24h
            # retry, and 24h is a bucket edge: the incumbent ladder's rungs
            # would all be filed one bucket early.
            delay = np.round((tail["t_h"].to_numpy(dtype=float)
                              - ft.reindex(tail["invoice_ref"]).to_numpy()) * 3600.0) / 3600.0
            disputed_ids = (set(disputes["payment_id"].astype(str))
                            if disputes is not None and len(disputes) else set())
            attempts = pd.DataFrame({
                "invoice_id": tail["invoice_ref"].astype(str).to_numpy(),
                "merchant_id": tail["merchant_id"].to_numpy(),
                "payment_id": tail["payment_id"].to_numpy(),
                "attempt_index": (tail["_k"] - 1).to_numpy(dtype=int),
                "delay_hours": delay, "dt_bucket": dt_bucket(delay).astype(int),
                "attempt_time_h": tail["t_h"].to_numpy(dtype=float),
                "success": tail["success"].to_numpy(dtype=int),
                "disputed": (tail["success"].astype(bool)
                             & tail["payment_id"].astype(str).isin(disputed_ids)
                             ).to_numpy(dtype=int),
                # No logged propensity here: the payment stream does not know
                # which policy chose the delay. Propensities live in the
                # decision audit log (``recoup-ops plan`` writes it) and the
                # gate joins them back. Legacy history without one is the
                # case where off-policy estimates are suggestive at best --
                # see README.
                "propensity": np.nan,
            }, columns=att_cols).sort_values(["attempt_time_h"], kind="stable"
                                             ).reset_index(drop=True)

    gone = pd.DataFrame({"customer_id": pd.Series(dtype=str),
                         "event_time_h": pd.Series(dtype=float)})
    if customer_events is not None and len(customer_events):
        ev = customer_events[customer_events["type"].isin(GONE_EVENT_TYPES)]
        if len(ev):
            gone = pd.DataFrame({"customer_id": ev["customer_id"].astype(str).to_numpy(),
                                 "event_time_h": _to_hours(ev["created_at"], epoch)})

    report.n_customers_without_profile = missing_profile
    return BachsTables(invoices, attempts, successes, gone, report)


def assume_propensities(attempts: pd.DataFrame, ladder_h: tuple[float, ...],
                        epsilon: float = 0.02) -> pd.DataFrame:
    """Backfill propensities for pre-existing history under a stated ladder.

    Use only when you know what ladder was running. The result is an assumption,
    not a measurement, and every estimate downstream inherits that. Prefer
    running the real epsilon-randomised policy for a few weeks instead.
    """
    out = attempts.copy()
    p = np.full(len(out), epsilon / N_DT_BUCKETS)
    nominal = np.array([int(dt_bucket(ladder_h[min(int(k), len(ladder_h) - 1)]))
                        for k in out["attempt_index"]])
    actual = dt_bucket(out["delay_hours"].to_numpy())
    p = np.where(actual == nominal, p + (1 - epsilon), p)
    out["propensity"] = p
    out.attrs["propensities_are_assumed"] = True
    return out
