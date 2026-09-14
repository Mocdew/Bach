"""What a single decision looks like.

Run:  python quickstart.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from recoup import (
    CureHazardModel, PolicyConfig, RetryPolicy, SimConfig, SupportMap,
    TableDisputeModel, build_features, customer_history, simulate, temporal_split,
)
from recoup.domain import BALANCE_DRIVEN, INFRA_DRIVEN, DeclineReason, NetworkAdvice, Rail

pd.set_option("display.width", 120)


def main() -> None:
    invoices, attempts = simulate(SimConfig(n_invoices=5000))
    invoices = customer_history(invoices, attempts)
    inv_tr, att_tr, inv_te, _, _ = temporal_split(invoices, attempts, 0.75)

    train = build_features(inv_tr, att_tr[["invoice_id", "attempt_index", "delay_hours"]])
    train["success"] = att_tr["success"].to_numpy()
    model = CureHazardModel().fit(train)

    policy = RetryPolicy(model, inv_te, SupportMap(train), PolicyConfig(),
                         dispute_model=TableDisputeModel.fit(att_tr, inv_tr),
                         rng=np.random.default_rng(3))

    print("Four invoices, four different right answers.\n")
    soft_ok = ~inv_te["network_advice"].isin([a.value for a in
                                              (NetworkAdvice.DO_NOT_RETRY,
                                               NetworkAdvice.CANCELLED_RECURRING,
                                               NetworkAdvice.UPDATED_INFO_REQUIRED)])
    picks = {
        "balance-driven decline on a card": inv_te[soft_ok & inv_te["decline_reason"].isin(
            {r.value for r in BALANCE_DRIVEN}) & (inv_te["rail"] == Rail.CARD.value)],
        "wallet empty on mobile money": inv_te[soft_ok & (inv_te["decline_reason"] == DeclineReason.WALLET_EMPTY.value)],
        "infrastructure decline": inv_te[soft_ok & inv_te["decline_reason"].isin({r.value for r in INFRA_DRIVEN})],
        "issuer says do not retry": inv_te[inv_te["network_advice"] == NetworkAdvice.DO_NOT_RETRY.value],
    }

    for label, cand in picks.items():
        if cand.empty:
            continue
        inv = cand.iloc[0]
        d = policy.decide(inv["invoice_id"])

        print("-" * 74)
        print(f"{label}: {inv['decline_reason']} on {inv['rail']} in {inv['market']}, "
              f"${inv['amount_usd']:.2f}, advice={inv['network_advice']}")
        print(f"  action      {d.action}")
        if d.delay_hours is not None:
            print(f"  retry in    {d.delay_hours:.1f}h  (bucket {d.bucket}, propensity {d.propensity:.2f})")
            print(f"  schedule    {' -> '.join(f'{h:.0f}h' for h in d.schedule_h)}")
            print(f"  P(gone)     {d.p_gone:.2f}    P(success | first attempt) {d.p_success:.2f}")
            print(f"  EV          ${d.expected_value_usd:.2f} for committing to this first bucket")
        print(f"  why         {d.rationale}")
        if d.curve is not None:
            c = d.curve[d.curve["propensity"] > 0].round(3)
            print("\n  value of committing to each first bucket, then planning optimally:")
            print("    " + c.to_string(index=False).replace("\n", "\n    "))
        print()


if __name__ == "__main__":
    main()
