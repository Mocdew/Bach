"""Feature construction for the retry-outcome table.

One function serves both training and inference. That is deliberate: the most
common way a retry-timing model breaks in production is a skew between the
features computed offline and the ones computed when scoring candidate delays.
Sharing the code path removes the class of bug entirely.

A row is *one (invoice, candidate delay, attempt index) triple*, not one
invoice. The label is whether an attempt at that delay succeeded.

What is deliberately NOT here
-----------------------------
There is no ``payday_window`` feature. An earlier version shipped
``exp(-0.5 * (days_to_payday / 3.2)**2)``, which is the simulator's own
funding function copied verbatim -- the model was being handed the generative
variable and the benchmark lift measured nothing. Calendar inputs are raw
(``days_to_payday``, ``local_dom``, ``local_hour``, day of week); any shape on
top of them must be learned.

This module does not import the simulator. The time origin comes from
``domain.DEFAULT_EPOCH`` or the ``epoch`` argument.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from .domain import (
    BALANCE_DRIVEN,
    DEFAULT_EPOCH,
    HARD_DECLINES,
    INFRA_DRIVEN,
    MARKETS,
    DeclineReason,
    NetworkAdvice,
    dt_bucket,
)

CATEGORICAL = ["rail", "decline_reason", "market", "plan_tier", "reason_class",
               "network_advice"]

NUMERIC = [
    "delay_hours", "log_delay", "dt_bucket", "attempt_index",
    "local_hour", "local_dom", "local_dow", "days_to_payday",
    "is_weekend", "log_amount", "tenure_days", "prior_successful_payments",
    "active_in_window",
    # customer history across invoices (0 when unknown / first invoice)
    "n_prior_failures", "n_prior_recoveries", "prev_recovery_delay_h",
    "has_prior_recovery",
]

FEATURES = CATEGORICAL + NUMERIC

HISTORY_COLS = ["n_prior_failures", "n_prior_recoveries",
                "prev_recovery_delay_h", "has_prior_recovery"]


def _reason_class(reason: str) -> str:
    r = DeclineReason(reason)
    if r in HARD_DECLINES:
        return "hard"
    if r in BALANCE_DRIVEN:
        return "balance"
    if r in INFRA_DRIVEN:
        return "infra"
    return "opaque"


def _calendar(abs_h: np.ndarray, markets: pd.Series,
              epoch: datetime = DEFAULT_EPOCH) -> pd.DataFrame:
    """Local-time calendar features.

    Computed in the *customer's* timezone. Doing this in UTC is the quiet bug
    that makes payday features look useless.
    """
    offsets = markets.map(lambda m: MARKETS[m].utc_offset_h).to_numpy(dtype=float)
    local_h = np.asarray(abs_h, dtype=float) + offsets

    ts = pd.to_datetime(epoch) + pd.to_timedelta(local_h, unit="h")
    dom = ts.day.to_numpy()
    hour = ts.hour.to_numpy() + ts.minute.to_numpy() / 60.0
    dow = ts.dayofweek.to_numpy()
    dim = ts.days_in_month.to_numpy()

    paydays = markets.map(lambda m: MARKETS[m].payday_dom).to_list()
    d2p = np.empty(len(dom), dtype=float)
    for i, pds in enumerate(paydays):
        raw = [abs(int(dom[i]) - p) for p in pds]
        circ = [min(r, int(dim[i]) - r) for r in raw]
        d2p[i] = float(min(circ))

    return pd.DataFrame(
        {
            "local_hour": hour,
            "local_dom": dom.astype(float),
            "local_dow": dow.astype(float),
            "days_to_payday": d2p,
            "is_weekend": (dow >= 5).astype(float),
        }
    )


def customer_history(invoices: pd.DataFrame, attempts: pd.DataFrame | None) -> pd.DataFrame:
    """Per-invoice summary of the *same customer's* earlier invoices.

    Only outcomes observed before this invoice's failure time are used, so the
    features are leak-free under a temporal split. Returns ``invoices`` with the
    ``HISTORY_COLS`` attached. Invoices without a ``customer_id`` get zeros.
    """
    out = invoices.copy()
    for c in HISTORY_COLS:
        out[c] = 0.0
    if attempts is None or "customer_id" not in out.columns:
        return out

    att = attempts.merge(out[["invoice_id", "customer_id"]], on="invoice_id", how="inner")
    # Outcome of each invoice: recovered? at what delay? when was it known?
    outcome = (
        att.sort_values(["invoice_id", "attempt_index"])
        .groupby("invoice_id")
        .agg(recovered=("success", "max"),
             rec_delay=("delay_hours", lambda s: float(s.iloc[-1])),
             known_at=("attempt_time_h", "max"),
             customer_id=("customer_id", "first"))
        .reset_index()
    )
    fail_times = out.set_index("invoice_id")["fail_time_h"]

    by_cust: dict = {}
    for r in outcome.itertuples(index=False):
        by_cust.setdefault(r.customer_id, []).append(
            (float(r.known_at), int(r.recovered), float(r.rec_delay)))
    for k in by_cust:
        by_cust[k].sort()

    rows = []
    for inv_id, cust, ft in zip(out["invoice_id"], out["customer_id"], out["fail_time_h"]):
        hist = [h for h in by_cust.get(cust, []) if h[0] < ft]
        n_fail = len(hist)
        n_rec = sum(h[1] for h in hist)
        prev = [h[2] for h in hist if h[1]]
        rows.append((n_fail, n_rec, prev[-1] if prev else 0.0, 1.0 if prev else 0.0))
    out[HISTORY_COLS] = pd.DataFrame(rows, index=out.index, columns=HISTORY_COLS).astype(float)
    return out


def build_features(invoices: pd.DataFrame, triples: pd.DataFrame,
                   epoch: datetime = DEFAULT_EPOCH) -> pd.DataFrame:
    """Join invoice state to candidate delays and derive features.

    ``triples`` needs columns: invoice_id, attempt_index, delay_hours.
    """
    inv_cols = [
        "invoice_id", "merchant_id", "market", "rail", "decline_reason",
        "fail_time_h", "amount_usd", "customer_tenure_days",
        "prior_successful_payments", "plan_tier", "active_in_window",
    ]
    optional = ["customer_id", "network_advice"] + HISTORY_COLS
    inv_cols += [c for c in optional if c in invoices.columns]

    df = triples.merge(invoices[inv_cols], on="invoice_id", how="left", suffixes=("", "_inv"))
    if "merchant_id_inv" in df.columns:
        df["merchant_id"] = df["merchant_id_inv"]
        df = df.drop(columns=["merchant_id_inv"])

    df["attempt_time_h"] = df["fail_time_h"] + df["delay_hours"]
    cal = _calendar(df["attempt_time_h"].to_numpy(), df["market"], epoch)
    df = pd.concat([df.reset_index(drop=True), cal], axis=1)

    df["log_delay"] = np.log1p(df["delay_hours"])
    df["dt_bucket"] = dt_bucket(df["delay_hours"].to_numpy()).astype(float)
    df["log_amount"] = np.log1p(df["amount_usd"])
    df["tenure_days"] = df["customer_tenure_days"].fillna(0.0)
    df["active_in_window"] = df["active_in_window"].astype(float)
    df["reason_class"] = df["decline_reason"].map(_reason_class)
    if "network_advice" not in df.columns:
        df["network_advice"] = NetworkAdvice.NONE.value
    df["network_advice"] = df["network_advice"].fillna(NetworkAdvice.NONE.value)
    for c in HISTORY_COLS:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = df[c].fillna(0.0).astype(float)

    for c in CATEGORICAL:
        df[c] = df[c].astype("category")

    return df
