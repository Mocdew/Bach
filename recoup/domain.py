"""Domain vocabulary for payment recovery.

Everything downstream keys off these enums, so getting them right matters more
than any modelling choice. In particular the hard/soft decline split is a
routing decision, not a timing one: hard declines must never reach the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Sequence

import numpy as np

# Absolute-time origin for every ``*_h`` column in the package. Owned here so
# that production feature code never imports the simulator to find it.
DEFAULT_EPOCH = datetime(2026, 1, 1)


class Rail(str, Enum):
    """Payment rails have near-unrelated failure dynamics."""

    CARD = "card"
    MOBILE_MONEY = "mobile_money"
    BANK_TRANSFER = "bank_transfer"
    STABLECOIN = "stablecoin"


class DeclineReason(str, Enum):
    # --- hard: never retry, route to new payment method -------------------
    STOLEN_CARD = "stolen_card"
    ACCOUNT_CLOSED = "account_closed"
    REVOKED_AUTHORIZATION = "revoked_authorization"
    INVALID_ACCOUNT = "invalid_account"
    # --- soft: balance driven, retry on payday rhythm ---------------------
    INSUFFICIENT_FUNDS = "insufficient_funds"
    WALLET_EMPTY = "wallet_empty"
    LIMIT_EXCEEDED = "limit_exceeded"
    # --- soft: infrastructure driven, retry in minutes/hours --------------
    ISSUER_UNAVAILABLE = "issuer_unavailable"
    RAIL_TIMEOUT = "rail_timeout"
    PROCESSING_ERROR = "processing_error"
    # --- soft: opaque issuer risk decision --------------------------------
    DO_NOT_HONOR = "do_not_honor"


HARD_DECLINES: frozenset[DeclineReason] = frozenset(
    {
        DeclineReason.STOLEN_CARD,
        DeclineReason.ACCOUNT_CLOSED,
        DeclineReason.REVOKED_AUTHORIZATION,
        DeclineReason.INVALID_ACCOUNT,
    }
)

BALANCE_DRIVEN: frozenset[DeclineReason] = frozenset(
    {
        DeclineReason.INSUFFICIENT_FUNDS,
        DeclineReason.WALLET_EMPTY,
        DeclineReason.LIMIT_EXCEEDED,
    }
)

INFRA_DRIVEN: frozenset[DeclineReason] = frozenset(
    {
        DeclineReason.ISSUER_UNAVAILABLE,
        DeclineReason.RAIL_TIMEOUT,
        DeclineReason.PROCESSING_ERROR,
    }
)


def is_retryable(reason: DeclineReason) -> bool:
    return reason not in HARD_DECLINES


# ---------------------------------------------------------------------------
# Network advice
# ---------------------------------------------------------------------------
# Card networks attach an explicit retry instruction to many declines
# (Mastercard Merchant Advice Codes, Visa decline categories). This is the
# single most informative timing signal the issuer gives you, and ignoring it
# is also how you get fined. It is both a *feature* and a *constraint*.


class NetworkAdvice(str, Enum):
    NONE = "none"                          # no advice attached (non-card rails)
    TRY_AGAIN_LATER = "try_again_later"    # MAC 02 / Visa category 2-4
    UPDATED_INFO_REQUIRED = "updated_info" # MAC 01: credentials changed
    DO_NOT_RETRY = "do_not_retry"          # MAC 03 / Visa category 1
    CANCELLED_RECURRING = "cancelled_recurring"  # MAC 21: cardholder cancelled


NO_RETRY_ADVICE: frozenset[NetworkAdvice] = frozenset(
    {NetworkAdvice.DO_NOT_RETRY, NetworkAdvice.CANCELLED_RECURRING,
     NetworkAdvice.UPDATED_INFO_REQUIRED}
)


@dataclass(frozen=True)
class NetworkRules:
    """Hard retry limits. Defaults follow the card-network rules; the per-rail
    override exists because mobile-money retries are bounded by customer
    tolerance (each retry is a push prompt on their phone), not by a scheme.
    """

    max_attempts_per_invoice: int = 4
    # Visa: at most 15 authorisation attempts per card per 30 days.
    max_attempts_per_card_window: int = 15
    card_window_days: int = 30
    per_rail_max_attempts: dict = field(default_factory=lambda: {
        Rail.MOBILE_MONEY: 3, Rail.STABLECOIN: 3,
    })

    def attempt_cap(self, rail: Rail) -> int:
        return int(self.per_rail_max_attempts.get(rail, self.max_attempts_per_invoice))


# ---------------------------------------------------------------------------
# Delay discretisation
# ---------------------------------------------------------------------------
# Log-ish spacing: infra declines resolve in hours, balance declines in days.
# Bucket edges are in hours since the original failure.
DT_BUCKET_EDGES_H: tuple[float, ...] = (
    0.0, 2.0, 6.0, 24.0, 48.0, 72.0, 120.0, 168.0, 240.0, 336.0,
)
N_DT_BUCKETS = len(DT_BUCKET_EDGES_H) - 1
MAX_HORIZON_H = DT_BUCKET_EDGES_H[-1]


def dt_bucket(delay_hours: float | np.ndarray) -> np.ndarray:
    """Map a delay in hours to a bucket index in [0, N_DT_BUCKETS)."""
    idx = np.digitize(delay_hours, DT_BUCKET_EDGES_H[1:-1], right=False)
    return np.clip(idx, 0, N_DT_BUCKETS - 1)


# ---------------------------------------------------------------------------
# Markets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Market:
    """A settlement market. Payday rhythm is the whole point of this class."""

    code: str
    currency: str
    utc_offset_h: float
    # Day-of-month on which salaries predominantly land.
    payday_dom: tuple[int, ...]
    # Fraction of wallet top-ups that follow a weekly rather than monthly cycle.
    weekly_topup_share: float
    rail_mix: dict[Rail, float] = field(default_factory=dict)


MARKETS: dict[str, Market] = {
    "NG": Market("NG", "NGN", 1.0, (25, 26, 27, 28), 0.35,
                 {Rail.CARD: 0.55, Rail.BANK_TRANSFER: 0.30,
                  Rail.MOBILE_MONEY: 0.05, Rail.STABLECOIN: 0.10}),
    "KE": Market("KE", "KES", 3.0, (28, 29, 30), 0.55,
                 {Rail.MOBILE_MONEY: 0.60, Rail.CARD: 0.25,
                  Rail.BANK_TRANSFER: 0.10, Rail.STABLECOIN: 0.05}),
    "GH": Market("GH", "GHS", 0.0, (26, 27, 28), 0.50,
                 {Rail.MOBILE_MONEY: 0.55, Rail.CARD: 0.30,
                  Rail.BANK_TRANSFER: 0.10, Rail.STABLECOIN: 0.05}),
    "ZA": Market("ZA", "ZAR", 2.0, (25, 26), 0.20,
                 {Rail.CARD: 0.75, Rail.BANK_TRANSFER: 0.20,
                  Rail.MOBILE_MONEY: 0.02, Rail.STABLECOIN: 0.03}),
    "GB": Market("GB", "GBP", 0.0, (28, 29, 30, 31), 0.10,
                 {Rail.CARD: 0.88, Rail.BANK_TRANSFER: 0.10,
                  Rail.MOBILE_MONEY: 0.0, Rail.STABLECOIN: 0.02}),
    "US": Market("US", "USD", -5.0, (1, 15, 16, 30), 0.15,
                 {Rail.CARD: 0.90, Rail.BANK_TRANSFER: 0.07,
                  Rail.MOBILE_MONEY: 0.0, Rail.STABLECOIN: 0.03}),
}


def market_codes() -> Sequence[str]:
    return tuple(MARKETS.keys())
