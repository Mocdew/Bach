"""Synthetic generator for failed invoices and retry attempts.

Why this exists
---------------
You cannot develop a retry-timing policy without retry data, and you cannot get
retry data without a policy. The simulator breaks that loop. It also gives you
something logged data never can: a *ground-truth* hazard function, so you can
check whether your off-policy estimator is telling the truth before you trust it
on real revenue.

The generative story
--------------------
A customer belongs to a merchant and generates several invoices over the
horizon. An invoice fails for a reason drawn from a rail-conditional
distribution. Hard declines are terminal. Soft declines recover according to a
latent process:

  * balance-driven declines track the customer's payday rhythm;
  * infrastructure declines decay with a half-life of hours;
  * opaque issuer declines sit in between and fatigue with attempt count.

Some customers are silently gone (churned, card in a drawer) and never recover
regardless of timing. Churn is a *customer* state with an onset time, so a
customer's later invoices are informative about their earlier ones -- which is
what the customer-history features and the cure model are built to exploit.

Card declines carry a network advice code (MAC / Visa category) that is
correlated with, but not determined by, the underlying reason.

The logging policy is a fixed ladder with epsilon-randomisation, and it records
its own propensities.

Misspecification
----------------
Every structural constant lives in ``Truth``. The benchmark re-runs under
perturbed truths (payday effect removed, shifted, widened) so a policy's lift
is reported as a range across worlds rather than one number from the world it
was tuned on.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .domain import (
    BALANCE_DRIVEN,
    DEFAULT_EPOCH,
    HARD_DECLINES,
    INFRA_DRIVEN,
    MARKETS,
    MAX_HORIZON_H,
    N_DT_BUCKETS,
    DeclineReason,
    NetworkAdvice,
    Rail,
    dt_bucket,
)

SIM_START = DEFAULT_EPOCH

# Per-rail decline mix. Mobile money times out far more than cards; cards carry
# most of the hard-decline mass.
_DECLINE_MIX: dict[Rail, dict[DeclineReason, float]] = {
    Rail.CARD: {
        DeclineReason.INSUFFICIENT_FUNDS: 0.38,
        DeclineReason.DO_NOT_HONOR: 0.20,
        DeclineReason.LIMIT_EXCEEDED: 0.06,
        DeclineReason.ISSUER_UNAVAILABLE: 0.10,
        DeclineReason.PROCESSING_ERROR: 0.05,
        DeclineReason.RAIL_TIMEOUT: 0.03,
        DeclineReason.STOLEN_CARD: 0.03,
        DeclineReason.ACCOUNT_CLOSED: 0.06,
        DeclineReason.REVOKED_AUTHORIZATION: 0.05,
        DeclineReason.INVALID_ACCOUNT: 0.04,
    },
    Rail.MOBILE_MONEY: {
        DeclineReason.WALLET_EMPTY: 0.42,
        DeclineReason.RAIL_TIMEOUT: 0.25,
        DeclineReason.ISSUER_UNAVAILABLE: 0.12,
        DeclineReason.LIMIT_EXCEEDED: 0.08,
        DeclineReason.PROCESSING_ERROR: 0.06,
        DeclineReason.INVALID_ACCOUNT: 0.04,
        DeclineReason.ACCOUNT_CLOSED: 0.03,
    },
    Rail.BANK_TRANSFER: {
        DeclineReason.INSUFFICIENT_FUNDS: 0.45,
        DeclineReason.ISSUER_UNAVAILABLE: 0.18,
        DeclineReason.PROCESSING_ERROR: 0.12,
        DeclineReason.RAIL_TIMEOUT: 0.10,
        DeclineReason.ACCOUNT_CLOSED: 0.08,
        DeclineReason.INVALID_ACCOUNT: 0.07,
    },
    Rail.STABLECOIN: {
        DeclineReason.WALLET_EMPTY: 0.55,
        DeclineReason.RAIL_TIMEOUT: 0.20,
        DeclineReason.PROCESSING_ERROR: 0.15,
        DeclineReason.INVALID_ACCOUNT: 0.10,
    },
}


@dataclass(frozen=True)
class Truth:
    """Structural constants of the ground-truth process. Perturb to test
    whether the learned policy's lift depends on the world looking exactly
    like the one it was developed in."""

    payday_width_days: float = 3.2     # spread of the salary spike
    payday_shift_days: float = 0.0     # settlement lag after nominal payday
    payday_strength: float = 0.78      # share of balance recovery driven by payday
    balance_scale: float = 0.62
    infra_tau_scale: float = 1.0
    staleness_scale: float = 1.0
    advice_boost: float = 1.20         # TRY_AGAIN_LATER really does mean that


DEFAULT_TRUTH = Truth()


@dataclass
class SimConfig:
    n_merchants: int = 12
    n_invoices: int = 9000
    days: int = 240
    invoices_per_customer: float = 2.5
    # Share of customers who churn at some point in the horizon.
    churned_share: float = 0.28
    # Logging policy: fixed ladder in hours, plus exploration rate.
    ladder_h: tuple[float, ...] = (24.0, 72.0, 168.0)
    epsilon: float = 0.25
    max_attempts: int = 3
    seed: int = 7
    truth: Truth = field(default_factory=Truth)


# ---------------------------------------------------------------------------
# Ground truth hazard
# ---------------------------------------------------------------------------


def _hours_to_local_dom_hour(abs_h: float, utc_offset_h: float) -> tuple[int, float, int]:
    """Return (day_of_month, hour_of_day, days_in_month) in local time."""
    ts = SIM_START + timedelta(hours=float(abs_h) + utc_offset_h)
    nxt = (ts.replace(day=28) + timedelta(days=4)).replace(day=1)
    days_in_month = (nxt - timedelta(days=1)).day
    return ts.day, ts.hour + ts.minute / 60.0, days_in_month


def _payday_funding(abs_h: float, market_code: str, rail: Rail,
                    truth: Truth = DEFAULT_TRUTH) -> float:
    """Probability mass that the customer has money available right now."""
    mkt = MARKETS[market_code]
    dom, _, dim = _hours_to_local_dom_hour(abs_h, mkt.utc_offset_h)

    dists = [min(abs(dom - (p + truth.payday_shift_days)),
                 dim - abs(dom - (p + truth.payday_shift_days))) for p in mkt.payday_dom]
    d = float(min(dists))
    monthly = float(np.exp(-0.5 * (d / truth.payday_width_days) ** 2))

    weekly = 0.0
    if rail in (Rail.MOBILE_MONEY, Rail.STABLECOIN):
        ts = SIM_START + timedelta(hours=float(abs_h) + mkt.utc_offset_h)
        dow = ts.weekday()  # 0 Mon
        weekly = float(np.exp(-0.5 * ((min(abs(dow - 4), 7 - abs(dow - 4))) / 1.5) ** 2))

    w = mkt.weekly_topup_share if rail in (Rail.MOBILE_MONEY, Rail.STABLECOIN) else 0.0
    return (1.0 - w) * monthly + w * weekly


def true_success_prob(
    *,
    reason: DeclineReason,
    rail: Rail,
    market_code: str,
    fail_time_h: float,
    attempt_time_h: float,
    attempt_index: int,
    customer_quality: float,
    amount_usd: float,
    churned: bool,
    advice: NetworkAdvice = NetworkAdvice.NONE,
    truth: Truth = DEFAULT_TRUTH,
) -> float:
    """Ground-truth hazard. The simulator's oracle; never visible to models."""
    if churned or reason in HARD_DECLINES or advice in (
            NetworkAdvice.DO_NOT_RETRY, NetworkAdvice.CANCELLED_RECURRING):
        return 0.0

    elapsed = max(attempt_time_h - fail_time_h, 0.0)
    if elapsed > MAX_HORIZON_H:
        return 0.0

    mkt = MARKETS[market_code]
    _, local_hour, _ = _hours_to_local_dom_hour(attempt_time_h, mkt.utc_offset_h)

    if reason in INFRA_DRIVEN:
        tau = (2.5 if rail in (Rail.MOBILE_MONEY, Rail.STABLECOIN) else 5.0) * truth.infra_tau_scale
        base = 0.72 * (1.0 - np.exp(-elapsed / tau))
        staleness = float(np.exp(-elapsed / (60.0 * truth.staleness_scale)))
        if rail is Rail.BANK_TRANSFER:
            base *= 0.55 + 0.45 * float(np.exp(-0.5 * ((local_hour - 12) / 4.5) ** 2))
        fatigue = 0.88 ** attempt_index

    elif reason in BALANCE_DRIVEN:
        funding = _payday_funding(attempt_time_h, market_code, rail, truth)
        drift = 1.0 - np.exp(-elapsed / 96.0)
        s = truth.payday_strength
        base = truth.balance_scale * ((1.0 - s) * drift + s * funding)
        staleness = float(np.exp(-elapsed / (400.0 * truth.staleness_scale)))
        base *= float(np.clip(1.15 - 0.10 * np.log1p(amount_usd / 25.0), 0.45, 1.15))
        fatigue = 0.93 ** attempt_index

    else:  # DO_NOT_HONOR -- opaque issuer risk
        base = 0.30 * (1.0 - np.exp(-elapsed / 40.0))
        base *= 0.75 + 0.25 * _payday_funding(attempt_time_h, market_code, rail, truth)
        staleness = float(np.exp(-elapsed / (220.0 * truth.staleness_scale)))
        fatigue = 0.72 ** attempt_index

    if advice is NetworkAdvice.TRY_AGAIN_LATER:
        base *= truth.advice_boost

    p = base * staleness * fatigue * (0.55 + 0.9 * customer_quality)
    return float(np.clip(p, 0.0, 0.97))


def true_dispute_prob(*, attempt_index: int, elapsed_h: float, rail: Rail) -> float:
    """Chargeback risk on a *recovered* payment."""
    if rail in (Rail.MOBILE_MONEY, Rail.STABLECOIN):
        return 0.0005 * (1 + attempt_index)
    days = elapsed_h / 24.0
    return float(np.clip(0.0022 * (1 + 0.55 * attempt_index) * (1 + 0.16 * days), 0.0, 0.09))


def _draw_advice(rng: np.random.Generator, rail: Rail, reason: DeclineReason) -> NetworkAdvice:
    """Issuer advice is correlated with the reason but not a relabelling of it."""
    if rail is not Rail.CARD:
        return NetworkAdvice.NONE
    u = rng.random()
    if reason in (DeclineReason.STOLEN_CARD, DeclineReason.ACCOUNT_CLOSED):
        return NetworkAdvice.DO_NOT_RETRY if u < 0.85 else NetworkAdvice.NONE
    if reason is DeclineReason.REVOKED_AUTHORIZATION:
        return NetworkAdvice.CANCELLED_RECURRING if u < 0.8 else NetworkAdvice.NONE
    if reason is DeclineReason.INVALID_ACCOUNT:
        return NetworkAdvice.UPDATED_INFO_REQUIRED if u < 0.7 else NetworkAdvice.NONE
    if reason in BALANCE_DRIVEN or reason is DeclineReason.DO_NOT_HONOR:
        return NetworkAdvice.TRY_AGAIN_LATER if u < 0.45 else NetworkAdvice.NONE
    return NetworkAdvice.NONE


# ---------------------------------------------------------------------------
# Logging policy
# ---------------------------------------------------------------------------


def logging_policy_propensities(cfg: SimConfig, attempt_index: int) -> np.ndarray:
    """Distribution over delay buckets used by the incumbent policy."""
    probs = np.full(N_DT_BUCKETS, cfg.epsilon / N_DT_BUCKETS)
    ladder_h = cfg.ladder_h[min(attempt_index, len(cfg.ladder_h) - 1)]
    probs[int(dt_bucket(ladder_h))] += 1.0 - cfg.epsilon
    return probs


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _draw_reason(rng: np.random.Generator, rail: Rail) -> DeclineReason:
    mix = _DECLINE_MIX[rail]
    keys = list(mix.keys())
    p = np.array([mix[k] for k in keys], dtype=float)
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


def _bucket_midpoint_h(b: int, rng: np.random.Generator) -> float:
    from .domain import DT_BUCKET_EDGES_H

    lo, hi = DT_BUCKET_EDGES_H[b], DT_BUCKET_EDGES_H[b + 1]
    return float(rng.uniform(lo, hi))


def simulate(cfg: SimConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate (invoices, attempts).

    ``attempts`` is the training table: one row per retry actually made, with
    the logging propensity attached.
    """
    cfg = cfg or SimConfig()
    truth = cfg.truth
    rng = np.random.default_rng(cfg.seed)

    merchant_ids = [f"mch_{i:03d}" for i in range(cfg.n_merchants)]
    weights = rng.pareto(1.1, size=cfg.n_merchants) + 0.35
    weights = weights / weights.sum()
    merchant_market = {m: rng.choice(list(MARKETS.keys()),
                                     p=[0.30, 0.18, 0.14, 0.12, 0.14, 0.12])
                       for m in merchant_ids}

    horizon_h = cfg.days * 24.0
    n_customers = max(int(cfg.n_invoices / cfg.invoices_per_customer), 1)

    # Customers: fixed merchant, market, quality, tenure, rail, churn onset.
    cust_merchant = rng.choice(merchant_ids, size=n_customers, p=weights)
    customers = []
    for c in range(n_customers):
        mid = str(cust_merchant[c])
        mkt = MARKETS[str(merchant_market[mid])]
        rails = list(mkt.rail_mix.keys())
        rp = np.array([mkt.rail_mix[r] for r in rails], dtype=float)
        quality = float(rng.beta(2.4, 1.8))
        churns = rng.random() < cfg.churned_share * (1.5 - quality)
        customers.append(dict(
            customer_id=f"cus_{c:06d}", merchant_id=mid, market=mkt.code,
            rail=rails[int(rng.choice(len(rails), p=rp / rp.sum()))],
            quality=quality,
            tenure_days=float(rng.gamma(2.0, 95.0)),
            plan_tier=str(rng.choice(["starter", "pro", "enterprise"], p=[0.62, 0.32, 0.06])),
            churn_time_h=float(rng.uniform(0, horizon_h)) if churns else np.inf,
        ))

    inv_rows, att_rows = [], []
    inv_customers = rng.integers(0, n_customers, size=cfg.n_invoices)
    fail_times = np.sort(rng.uniform(0, horizon_h - MAX_HORIZON_H - 1, size=cfg.n_invoices))

    for i in range(cfg.n_invoices):
        cu = customers[int(inv_customers[i])]
        mid, mkt_code, rail = cu["merchant_id"], cu["market"], cu["rail"]
        mkt = MARKETS[mkt_code]
        quality = cu["quality"]
        fail_h = float(fail_times[i])

        reason = _draw_reason(rng, rail)
        advice = _draw_advice(rng, rail, reason)
        amount = float(np.round(np.exp(rng.normal(3.1, 0.85)), 2))  # ~$22 median
        tenure_d = cu["tenure_days"] + fail_h / 24.0
        prior_success = int(rng.poisson(1 + tenure_d / 45.0))
        active_in_window = bool(rng.random() < (0.22 + 0.62 * quality))
        churned = bool(fail_h >= cu["churn_time_h"])

        inv_rows.append(
            dict(
                invoice_id=f"inv_{i:06d}", merchant_id=mid, customer_id=cu["customer_id"],
                market=mkt_code, currency=mkt.currency, rail=rail.value,
                decline_reason=reason.value, network_advice=advice.value,
                fail_time_h=fail_h, amount_usd=amount, customer_tenure_days=tenure_d,
                prior_successful_payments=prior_success, plan_tier=cu["plan_tier"],
                active_in_window=active_in_window,
                _quality=quality, _churned=churned,
            )
        )

        if reason in HARD_DECLINES:
            continue  # routed to new-payment-method flow, never retried

        for k in range(cfg.max_attempts):
            probs = logging_policy_propensities(cfg, k)
            ladder_h = cfg.ladder_h[min(k, len(cfg.ladder_h) - 1)]
            ladder_b = int(dt_bucket(ladder_h))
            b = int(rng.choice(N_DT_BUCKETS, p=probs))
            exploring = rng.random() >= (1.0 - cfg.epsilon) / probs[b] if b == ladder_b else True
            delay = _bucket_midpoint_h(b, rng) if exploring else ladder_h
            att_time = fail_h + delay

            p = true_success_prob(
                reason=reason, rail=rail, market_code=mkt_code, fail_time_h=fail_h,
                attempt_time_h=att_time, attempt_index=k, customer_quality=quality,
                amount_usd=amount, churned=churned, advice=advice, truth=truth,
            )
            success = bool(rng.random() < p)
            disputed = bool(
                success and rng.random() < true_dispute_prob(
                    attempt_index=k, elapsed_h=delay, rail=rail
                )
            )

            att_rows.append(
                dict(
                    invoice_id=f"inv_{i:06d}", merchant_id=mid, attempt_index=k,
                    delay_hours=delay, dt_bucket=b, attempt_time_h=att_time,
                    propensity=float(probs[b]), success=int(success),
                    disputed=int(disputed),
                )
            )
            if success:
                break

    invoices = pd.DataFrame(inv_rows)
    attempts = pd.DataFrame(att_rows)
    return invoices, attempts


PERTURBED_WORLDS: dict[str, Truth] = {
    "baseline": Truth(),
    "no payday effect": Truth(payday_strength=0.0),
    "payday lags 3 days": Truth(payday_shift_days=3.0),
    "payday spread wide": Truth(payday_width_days=6.0),
    "payday spread narrow": Truth(payday_width_days=1.5),
    "slow infra recovery": Truth(infra_tau_scale=3.0),
    "customers drift faster": Truth(staleness_scale=0.5),
}
