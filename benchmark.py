"""End-to-end benchmark.

Run:  python benchmark.py

Simulates a merchant population, fits the cure-hazard model on a temporal
split, plans retry schedules, and compares against the baselines any reviewer
will ask about -- against the simulator's ground truth so the off-policy
estimates can be audited, and across *perturbed* worlds so the lift is a range
rather than one number from the world the model was developed in.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from recoup import (
    PERTURBED_WORLDS, CureHazardModel, GBMHazard, BetaBinomialHazard, PolicyConfig,
    RetryPlanner, SimConfig, SupportMap, TableDisputeModel, build_features,
    cross_fitted_plan, customer_history, deployment_gate, fixed_ladder_policy,
    ladder_schedules, logged_first_attempts, month_end_policy, off_policy_value,
    oracle_policy_value, oracle_schedule_value, score_model, simulate, temporal_split,
)
from recoup.domain import HARD_DECLINES

pd.set_option("display.width", 130)
HARD = [d.value for d in HARD_DECLINES]
LADDER = (24.0, 72.0, 168.0)


def rule(title: str) -> None:
    print("\n" + "=" * 86)
    print(title)
    print("=" * 86)


def prepare(cfg_sim: SimConfig):
    invoices, attempts = simulate(cfg_sim)
    invoices = customer_history(invoices, attempts)
    inv_tr, att_tr, inv_te, att_te, cutoff = temporal_split(invoices, attempts, 0.70)
    train = build_features(inv_tr, att_tr[["invoice_id", "attempt_index", "delay_hours"]])
    train["success"] = att_tr["success"].to_numpy()
    test = build_features(inv_te, att_te[["invoice_id", "attempt_index", "delay_hours"]])
    test["success"] = att_te["success"].to_numpy()
    return invoices, attempts, inv_tr, att_tr, inv_te, att_te, train, test, cutoff


def main() -> None:
    cfg_sim, cfg_pol = SimConfig(), PolicyConfig()
    t_start = time.time()

    rule("1. SIMULATE")
    invoices, attempts, inv_tr, att_tr, inv_te, att_te, train, test, cutoff = prepare(cfg_sim)
    print(f"invoices             {len(invoices):,}  "
          f"({(~invoices['decline_reason'].isin(HARD)).sum():,} retryable, "
          f"{invoices['decline_reason'].isin(HARD).sum():,} hard declines)")
    print(f"customers            {invoices['customer_id'].nunique():,}  "
          f"(churned at some point: {invoices.groupby('customer_id')['_churned'].max().mean():.1%})")
    print(f"retry attempts       {len(attempts):,}   success {attempts['success'].mean():.3f}")
    print(f"temporal split       cutoff t={cutoff:,.0f}h  ->  train {len(train):,} / test {len(test):,}")

    rule("2. OUTCOME MODEL")
    y_te = test["success"].to_numpy()
    t = time.time(); cure = CureHazardModel().fit(train); t_fit = time.time() - t
    print(f"cure-hazard    {score_model(y_te, cure.predict_proba1(test))}   ({t_fit:.0f}s)")

    # P(gone) is an *invoice*-level quantity, so compare it per invoice. Doing
    # this over attempt rows -- which an earlier version did -- oversamples
    # churned invoices, because a churned invoice never succeeds and so always
    # spends its full attempt budget. That inflates the "true" share to 0.275
    # against a 0.225 invoice-level rate and makes the model look worse than
    # it is, for a reason no model could fix.
    one = test.drop_duplicates("invoice_id")
    pi_inv, _ = cure.predict_components(one)
    truth_inv = inv_te.set_index("invoice_id").loc[one["invoice_id"], "_churned"].to_numpy()
    print(f"               P(gone) per invoice: model {pi_inv.mean():.3f}  vs  true {truth_inv.mean():.3f}")
    # The remaining gap is mostly drift, not miscalibration: churn is absorbing
    # and onset is uniform over the horizon, so prevalence ramps through the
    # simulated period and the test window is simply churnier than the window
    # the model was fitted on. Print both so the two effects stay separable.
    print(f"               churn prevalence: train {inv_tr['_churned'].mean():.3f} "
          f"-> test {inv_te['_churned'].mean():.3f}  (non-stationary by construction)")
    print(f"gbm baseline   {score_model(y_te, GBMHazard().fit(train).predict_proba1(test))}")
    print(f"beta-binomial  {score_model(y_te, BetaBinomialHazard().fit(train).predict_proba1(test))}")
    print("\nNo payday_window feature: the model sees raw days-to-payday and has to learn the shape.")

    coef = cure.coef_table()
    print("\nWhat the model learned about payday (balance-driven declines):")
    print(coef[coef["term"].str.contains("d2p.*:bal")].to_string(index=False))

    rule("3. POLICY COMPARISON  (first retry decision, USD per failed invoice)")
    inv_eval, logged = logged_first_attempts(inv_te, att_te, cfg_pol)
    support = SupportMap(train, min_support=cfg_pol.min_support)
    dispute = TableDisputeModel.fit(att_tr, inv_tr)
    make_planner = lambda m: RetryPlanner(m, cfg_pol, dispute_model=dispute, support=support)

    t = time.time()
    pi_l, dl_l, q_cf = cross_fitted_plan(train, inv_eval, lambda d: CureHazardModel().fit(d), make_planner)
    print(f"(cross-fitted policy + reward model in {time.time() - t:.0f}s)")
    plan_full = make_planner(cure).plan(inv_eval, explore=False)

    policies = {
        "fixed ladder (incumbent)": fixed_ladder_policy(inv_eval, LADDER),
        "month-end payday heuristic": month_end_policy(inv_eval),
        "learned (greedy argmax)": None,
        "learned (planned, Thompson)": (pi_l, dl_l),
    }
    myo = RetryPlanner(cure, PolicyConfig(horizon=1, n_posterior_samples=0),
                       dispute_model=dispute, support=support).plan(inv_eval)
    policies["learned (greedy argmax)"] = (myo.pi, myo.delays)

    rows = []
    for name, (pi, dl) in policies.items():
        oracle = oracle_policy_value(inv_eval, pi, dl, cfg_pol)
        res = off_policy_value(logged, pi, q_cf, cfg_pol, oracle=oracle)
        rows.append(dict(policy=name, true_value=oracle, dr=res.dr, dr_err_pct=100 * (res.dr - oracle) / abs(oracle),
                         ess=res.ess))
    tab = pd.DataFrame(rows).set_index("policy")
    tab["lift_vs_ladder"] = tab["true_value"] / tab.loc["fixed ladder (incumbent)", "true_value"] - 1
    print(tab.round(3).to_string())
    print("\nDR uses a reward model fitted on the *other* half of training data from the")
    print("policy (cross-fitting), so the estimate is not the policy grading its own homework.")

    rule("4. THE WHOLE SCHEDULE, NOT JUST THE FIRST RETRY")
    v_plan = oracle_schedule_value(inv_eval, plan_full.schedule, cfg_pol)
    greedy = make_planner(cure).plan_greedy(inv_eval)
    v_grd = oracle_schedule_value(inv_eval, greedy, cfg_pol)
    v_lad = oracle_schedule_value(inv_eval, ladder_schedules(inv_eval, LADDER, cfg_sim.max_attempts), cfg_pol)
    print(f"true value of full dunning schedule per failed invoice:")
    print(f"  fixed ladder                 ${v_lad:6.2f}")
    print(f"  greedy argmax per attempt    ${v_grd:6.2f}   ({v_grd / v_lad - 1:+.1%})")
    print(f"  planned (exact DP)           ${v_plan:6.2f}   ({v_plan / v_lad - 1:+.1%})")
    lens = pd.Series([len(s) for s in plan_full.schedule])
    print(f"\nplanned schedule length: {lens.value_counts().sort_index().to_dict()}  "
          f"(0 = stop / route to new method)")
    rc = build_features(inv_eval, pd.DataFrame({"invoice_id": inv_eval["invoice_id"],
                                                "attempt_index": 0, "delay_hours": 24.0}))["reason_class"]
    first_delay = pd.Series([s[0] if s else np.nan for s in plan_full.schedule])
    print("\nfirst planned delay by decline class (hours):")
    print(pd.DataFrame({"reason_class": rc.astype(str), "h": first_delay}).dropna()
          .groupby("reason_class")["h"].describe()[["count", "25%", "50%", "75%"]].round(1).to_string())

    rule("5. DEPLOYMENT GATE")
    pi_inc, _ = policies["fixed ladder (incumbent)"]
    print(f"  planned vs incumbent   {deployment_gate(logged, pi_l, pi_inc, q_cf, cfg_pol)}")
    pi_pay, _ = policies["month-end payday heuristic"]
    print(f"  heuristic vs incumbent {deployment_gate(logged, pi_pay, pi_inc, q_cf, cfg_pol)}")

    rule("6. MISSPECIFICATION: DOES THE LIFT SURVIVE A DIFFERENT WORLD?")
    print("Re-simulate under perturbed ground truth, refit, replan. If the lift only exists")
    print("in the baseline world, the model has learned the simulator, not the problem.\n")
    print(f"{'world':<26} {'PR-AUC':>7} {'ladder':>8} {'planned':>8} {'lift':>7}  gate")
    for name, truth in PERTURBED_WORLDS.items():
        sc = SimConfig(n_invoices=6000, seed=5, truth=truth)
        inv_w, att_w, i_tr, a_tr, i_te, a_te, tr, te, _ = prepare(sc)
        m = CureHazardModel().fit(tr)
        rep = score_model(te["success"].to_numpy(), m.predict_proba1(te))
        ie, lg = logged_first_attempts(i_te, a_te, cfg_pol)
        sup = SupportMap(tr, min_support=cfg_pol.min_support)
        mk = lambda mm, sup=sup: RetryPlanner(mm, PolicyConfig(n_posterior_samples=16),
                                              dispute_model=dispute, support=sup)
        pi_w, dl_w, q_w = cross_fitted_plan(tr, ie, lambda d: CureHazardModel().fit(d), mk)
        plan_w = mk(m).plan(ie, explore=False)
        v_l = oracle_schedule_value(ie, ladder_schedules(ie, LADDER, sc.max_attempts), cfg_pol, truth)
        v_p = oracle_schedule_value(ie, plan_w.schedule, cfg_pol, truth)
        pi_f, _ = fixed_ladder_policy(ie, LADDER)
        g = deployment_gate(lg, pi_w, pi_f, q_w, cfg_pol, n_boot=200)
        print(f"{name:<26} {rep.pr_auc:>7.3f} {v_l:>8.2f} {v_p:>8.2f} {v_p / v_l - 1:>+7.1%}  "
              f"{'DEPLOY' if g.deploy else 'HOLD'}")

    rule("7. SENSITIVITY TO TRAINING VOLUME")
    print(f"{'invoices':>9}  {'PR-AUC':>7}  {'ladder':>8}  {'planned':>8}  {'lift':>7}  gate")
    for n in (1500, 3000, 6000, 9000):
        sc = SimConfig(n_invoices=n, seed=5)
        inv_w, att_w, i_tr, a_tr, i_te, a_te, tr, te, _ = prepare(sc)
        m = CureHazardModel().fit(tr)
        rep = score_model(te["success"].to_numpy(), m.predict_proba1(te))
        ie, lg = logged_first_attempts(i_te, a_te, cfg_pol)
        sup = SupportMap(tr, min_support=cfg_pol.min_support)
        mk = lambda mm, sup=sup: RetryPlanner(mm, PolicyConfig(n_posterior_samples=16),
                                              dispute_model=dispute, support=sup)
        pi_w, dl_w, q_w = cross_fitted_plan(tr, ie, lambda d: CureHazardModel().fit(d), mk)
        plan_w = mk(m).plan(ie, explore=False)
        v_l = oracle_schedule_value(ie, ladder_schedules(ie, LADDER, sc.max_attempts), cfg_pol)
        v_p = oracle_schedule_value(ie, plan_w.schedule, cfg_pol)
        pi_f, _ = fixed_ladder_policy(ie, LADDER)
        g = deployment_gate(lg, pi_w, pi_f, q_w, cfg_pol, n_boot=200)
        print(f"{n:>9}  {rep.pr_auc:>7.3f}  {v_l:>8.2f}  {v_p:>8.2f}  {v_p / v_l - 1:>+7.1%}  "
              f"{'DEPLOY' if g.deploy else 'HOLD'}")

    print(f"\ntotal {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
