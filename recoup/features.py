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
    MAX_HORIZON_H,
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


def derive_cure_labels(invoices: pd.DataFrame, attempts: pd.DataFrame,
                       gone_events: pd.DataFrame | None = None,
                       window_h: float = MAX_HORIZON_H,
                       successes: pd.DataFrame | None = None,
                       gone_within_h: float | None = None) -> pd.Series:
    """Partial labels for the cure component: 1 = gone, 0 = present, NaN = unknown.

    The cure fraction ``pi`` is the hardest thing in the model to identify. All
    the unsupervised likelihood has to work with is the curvature in repeated
    failures, and with at most four attempts per invoice that signal is thin --
    ``pi`` and a uniformly low hazard explain the same data almost equally
    well, which is why the fitted cure fraction sits well under the truth.
    Every observation you can label directly is worth more than another
    covariate.

    **Present (0)** -- the customer made a *successful* payment on some other
    invoice after this one's dunning window closed. "Gone" means gone for
    good, so a later success is a direct contradiction. Verified exact against
    simulator ground truth: 0.000 true churn rate among labelled-present
    invoices, versus a 0.090 base rate.

    **Gone (1)** -- requires ``gone_events``: a frame with ``customer_id`` and
    ``event_time_h`` recording an observed instrument death -- subscription
    cancelled, card deleted, account closed by the customer. An invoice is
    labelled gone when such an event lands after its failure and no success
    follows. This is the half you have to supply from real data; wire it to
    the Bachs subscription/payment-method endpoints when the adapter in
    ``bachs.py`` is pinned.

    .. warning::
       **Do not pass present labels alone.** Labelling here is
       outcome-dependent -- an invoice is labelled present precisely *because*
       the customer came back -- so it is a positive-unlabelled problem, not a
       random subsample. The unlabelled likelihood term needs a
       ``(1 - c(x))`` factor for the probability a live customer went
       unlabelled, and without it the live component is overstated and ``pi``
       collapses. Measured on the simulator: unlabelled ``pi`` 0.124 against a
       true 0.275, present-labels-only ``pi`` 0.016 -- three times worse.
       ``CureHazardModel.fit`` warns when it receives one-sided labels.

    An earlier version of this function derived gone labels from a later hard
    decline or no-retry issuer advice on the same customer. That was measured
    against simulator ground truth and carries no signal at all (0.088 true
    churn rate among labelled, against a 0.090 base), so it was removed rather
    than left as a plausible-looking default. A dead card is evidence about
    the *instrument*, not about whether the human intends to keep paying.

    ``successes`` (``customer_id``, ``t_h``) widens the present side to *any*
    successful payment -- a regular renewal that went through is the most
    common way a customer proves they are still there, and the retry table
    alone never sees it. ``from_payments`` returns this frame.

    ``gone_within_h`` bounds how long after the failure a gone event may land
    and still label the invoice. A customer who cancels five months later
    was not necessarily gone when this invoice failed; with churn modelled as
    an onset time, an unbounded window labels every earlier invoice of a
    future canceller as gone. An invoice that was itself recovered is never
    labelled gone -- the label would contradict the outcome, and the model
    would throw it away with a warning anyway.

    These are *labels*, not features: they look at data after the invoice's
    failure time and are consumed only by ``CureHazardModel.fit``. Nothing in
    ``build_features`` reads them, so they cannot reach inference. Derive them
    from the same time-truncated ``attempts`` frame you train on and the
    temporal split still holds.
    """
    idx = invoices.index
    if attempts is None or "customer_id" not in invoices.columns:
        return pd.Series(np.nan, index=idx, dtype=float)

    att = attempts.merge(invoices[["invoice_id", "customer_id"]], on="invoice_id", how="inner")
    wins = att[att["success"] == 1]
    win_times = [pd.DataFrame({"customer_id": wins["customer_id"].to_numpy(),
                               "t_h": wins["attempt_time_h"].to_numpy(dtype=float)})]
    if successes is not None and len(successes):
        win_times.append(successes[["customer_id", "t_h"]])
    all_wins = pd.concat(win_times, ignore_index=True)
    last_win = (all_wins.groupby("customer_id")["t_h"].max().astype(float).to_dict()
                if len(all_wins) else {})
    recovered = set(wins["invoice_id"])
    horizon = np.inf if gone_within_h is None else float(gone_within_h)

    first_dead: dict = {}
    if gone_events is not None and len(gone_events):
        for cust, t in zip(gone_events["customer_id"].to_numpy(),
                           gone_events["event_time_h"].to_numpy(dtype=float)):
            if cust not in first_dead or t < first_dead[cust]:
                first_dead[cust] = float(t)

    out = np.full(len(invoices), np.nan)
    for i, (inv, cust, ft) in enumerate(zip(invoices["invoice_id"].to_numpy(),
                                            invoices["customer_id"].to_numpy(),
                                            invoices["fail_time_h"].to_numpy(dtype=float))):
        close = ft + window_h
        came_back = cust in last_win and last_win[cust] > close
        if came_back:
            out[i] = 0.0
        elif (cust in first_dead and ft < first_dead[cust] <= ft + horizon
              and inv not in recovered):
            out[i] = 1.0
    return pd.Series(out, index=idx, dtype=float)

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
