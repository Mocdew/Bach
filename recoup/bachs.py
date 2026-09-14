"""Bachs API adapter.

Turns a merchant's own payment history into the two tables the rest of the
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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from .domain import DeclineReason, NetworkAdvice, Rail

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
}

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

    def __str__(self) -> str:
        s = f"{self.n_payments:,} payments, {self.n_failed:,} failed"
        if self.unmapped_codes:
            s += f"\n  UNMAPPED decline codes (treated as hard): {self.unmapped_codes}"
        if self.unmapped_rails:
            s += f"\n  UNMAPPED rails: {self.unmapped_rails}"
        return s


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


def from_payments(
    payments: Iterable[dict],
    *,
    market_of_customer: Callable[[str], str],
    epoch_iso: str = "2026-01-01T00:00:00Z",
) -> tuple[pd.DataFrame, pd.DataFrame, AdapterReport]:
    """Build (invoices, attempts) from a list of Bachs payment objects.

    A failed payment opens an invoice; subsequent payments carrying the same
    ``invoice_ref`` are its retry attempts, ordered by creation time.

    ``market_of_customer`` maps a customer id to a market code -- payday timing
    is local, so this must reflect where the *customer* is, not where the
    merchant settles.
    """
    rows, unmapped_c, unmapped_r = [], {}, {}
    epoch = pd.Timestamp(epoch_iso)

    for p in payments:
        code = _dig(p, FIELD_MAP["decline_code"])
        rail_raw = str(_dig(p, FIELD_MAP["rail"]) or "").lower()
        if rail_raw and rail_raw not in RAIL_MAP:
            unmapped_r[rail_raw] = unmapped_r.get(rail_raw, 0) + 1
        if code and str(code).lower() not in DECLINE_CODE_MAP:
            unmapped_c[str(code)] = unmapped_c.get(str(code), 0) + 1

        ts = pd.Timestamp(_dig(p, FIELD_MAP["created_at"]))
        rows.append(dict(
            payment_id=_dig(p, FIELD_MAP["payment_id"]),
            merchant_id=_dig(p, FIELD_MAP["merchant_id"]),
            customer_id=_dig(p, FIELD_MAP["customer_id"]),
            invoice_ref=_dig(p, FIELD_MAP["invoice_ref"]),
            status=str(_dig(p, FIELD_MAP["status"]) or "").lower(),
            amount_usd=float(_dig(p, FIELD_MAP["amount_minor"]) or 0) / 100.0,
            currency=_dig(p, FIELD_MAP["currency"]),
            rail=RAIL_MAP.get(rail_raw, Rail.CARD).value,
            decline_reason=map_decline_code(code).value,
            network_advice=map_advice_code(_dig(p, FIELD_MAP["advice_code"])).value,
            t_h=(ts.tz_localize(None) - epoch).total_seconds() / 3600.0,
        ))

    df = pd.DataFrame(rows).sort_values("t_h")
    failed = df[df["status"].isin({"failed", "declined"})]

    inv_rows, att_rows = [], []
    for ref, g in df.groupby("invoice_ref", sort=False):
        g = g.sort_values("t_h")
        first = g.iloc[0]
        if first["status"] not in {"failed", "declined"}:
            continue
        inv_rows.append(dict(
            invoice_id=str(ref), merchant_id=first["merchant_id"],
            market=market_of_customer(str(first["customer_id"])),
            currency=first["currency"], rail=first["rail"],
            decline_reason=first["decline_reason"],
            network_advice=first["network_advice"],
            fail_time_h=float(first["t_h"]), amount_usd=float(first["amount_usd"]),
            # TODO: pull these from the customers/subscriptions endpoints.
            customer_tenure_days=np.nan, prior_successful_payments=0,
            plan_tier="unknown", active_in_window=False,
        ))
        for k, (_, r) in enumerate(g.iloc[1:].iterrows()):
            att_rows.append(dict(
                invoice_id=str(ref), merchant_id=r["merchant_id"], attempt_index=k,
                delay_hours=float(r["t_h"] - first["t_h"]),
                attempt_time_h=float(r["t_h"]),
                success=int(r["status"] in {"succeeded", "paid", "completed"}),
                disputed=0,  # join from the disputes endpoint
                # No logged propensity: historical retries came from a policy
                # that did not record one. See README -- this is exactly the
                # case where off-policy estimates must be treated as suggestive
                # and a live holdout is required before trusting the model.
                propensity=np.nan,
            ))

    report = AdapterReport(len(df), len(failed), unmapped_c, unmapped_r)
    return pd.DataFrame(inv_rows), pd.DataFrame(att_rows), report


def assume_propensities(attempts: pd.DataFrame, ladder_h: tuple[float, ...],
                        epsilon: float = 0.02) -> pd.DataFrame:
    """Backfill propensities for pre-existing history under a stated ladder.

    Use only when you know what ladder was running. The result is an assumption,
    not a measurement, and every estimate downstream inherits that. Prefer
    running the real epsilon-randomised policy for a few weeks instead.
    """
    from .domain import N_DT_BUCKETS, dt_bucket

    out = attempts.copy()
    p = np.full(len(out), epsilon / N_DT_BUCKETS)
    nominal = np.array([int(dt_bucket(ladder_h[min(int(k), len(ladder_h) - 1)]))
                        for k in out["attempt_index"]])
    actual = dt_bucket(out["delay_hours"].to_numpy())
    p = np.where(actual == nominal, p + (1 - epsilon), p)
    out["propensity"] = p
    out.attrs["propensities_are_assumed"] = True
    return out
