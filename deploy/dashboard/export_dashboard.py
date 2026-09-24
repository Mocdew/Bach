"""Run the recoup pipeline and write dashboard.json for the operator console.

Usage:  python export_dashboard.py OUT.json

Runs on the simulator until the Bachs adapter (recoup/bachs.py) is pinned
against the live schema; every number it emits is therefore simulated and the
page says so. Swap `simulate()` for `from_payments()` + `customer_history()`
when the FIELD_MAP is verified and nothing else here has to change.
"""
import json, sys, time
from datetime import datetime, timezone
import numpy as np, pandas as pd
from recoup import (CureHazardModel, PolicyConfig, RetryPlanner, RetryPolicy, SimConfig, SupportMap,
                    TableDisputeModel, build_features, cross_fitted_plan, customer_history,
                    deployment_gate, fixed_ladder_policy, ladder_schedules, logged_first_attempts,
                    off_policy_value, oracle_schedule_value, score_model, simulate, temporal_split)
from recoup.domain import HARD_DECLINES, DT_BUCKET_EDGES_H, N_DT_BUCKETS

OUT = sys.argv[1]
now = lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
fitted_at = now()
HARD = [d.value for d in HARD_DECLINES]; LADDER = (24.0, 72.0, 168.0)
cfg_sim, cfg = SimConfig(n_invoices=6000, seed=5), PolicyConfig()
t0 = time.time()
invoices, attempts = simulate(cfg_sim); invoices = customer_history(invoices, attempts)
inv_tr, att_tr, inv_te, att_te, cutoff = temporal_split(invoices, attempts, 0.70)
train = build_features(inv_tr, att_tr[["invoice_id","attempt_index","delay_hours"]]); train["success"] = att_tr["success"].to_numpy()
test  = build_features(inv_te, att_te[["invoice_id","attempt_index","delay_hours"]]); test["success"] = att_te["success"].to_numpy()
cure = CureHazardModel().fit(train)
rep = score_model(test["success"].to_numpy(), cure.predict_proba1(test))
# P(gone) is per invoice, so compare it per invoice. Averaging it over attempt
# rows oversamples churned invoices -- a churned invoice never succeeds and so
# always spends its whole attempt budget -- and makes the model look badly
# miscalibrated for a reason no model could fix.
one = test.drop_duplicates("invoice_id")
pi_inv, _ = cure.predict_components(one)
truth_inv = inv_te.set_index("invoice_id").loc[one["invoice_id"], "_churned"].to_numpy()

inv_eval, logged = logged_first_attempts(inv_te, att_te, cfg)
support = SupportMap(train, min_support=cfg.min_support); dispute = TableDisputeModel.fit(att_tr, inv_tr)
mk = lambda m: RetryPlanner(m, cfg, dispute_model=dispute, support=support)
pi_l, dl_l, q_cf = cross_fitted_plan(train, inv_eval, lambda d: CureHazardModel().fit(d), mk)
plan = mk(cure).plan(inv_eval, explore=False)
pi_inc,_ = fixed_ladder_policy(inv_eval, LADDER)
g = deployment_gate(logged, pi_l, pi_inc, q_cf, cfg)
ope_plan = off_policy_value(logged, pi_l, q_cf, cfg); ope_inc = off_policy_value(logged, pi_inc, q_cf, cfg)
v_lad = oracle_schedule_value(inv_eval, ladder_schedules(inv_eval, LADDER, cfg_sim.max_attempts), cfg)
v_plan = oracle_schedule_value(inv_eval, plan.schedule, cfg)

# coverage: where the log put first attempts vs where the planner wants them
log_mass = np.bincount(logged["dt_bucket"].astype(int), minlength=N_DT_BUCKETS) / len(logged)
plan_mass = pi_l.mean(axis=0)
edges = list(DT_BUCKET_EDGES_H)
labels = [f"{int(a)}–{int(b)}h" for a,b in zip(edges[:-1], edges[1:])]

# weekly realised recovery under the incumbent (logged), test period
wk = ((inv_eval["fail_time_h"] - cutoff) // 168).astype(int)
per_inv = logged.groupby(wk.values)["reward"].agg(["mean","count"])
weekly = [dict(week=int(w), recovered=float(r["mean"]), n=int(r["count"])) for w,r in per_inv.iterrows() if r["count"] >= 30]

# open-invoice table: sample of retryable test invoices with a live decision
policy = RetryPolicy(cure, inv_te, support, cfg, dispute_model=dispute, rng=np.random.default_rng(3))
sample = inv_te[~inv_te["decline_reason"].isin(HARD)].sample(60, random_state=1)
rows = []
for _, inv in sample.iterrows():
    d = policy.decide(inv["invoice_id"])
    rows.append(dict(id=inv["invoice_id"], reason=inv["decline_reason"], rail=inv["rail"], market=inv["market"],
                     amount=round(float(inv["amount_usd"]),2), advice=inv["network_advice"], action=d.action,
                     delay=None if d.delay_hours is None else round(float(d.delay_hours),1),
                     schedule=[round(float(h)) for h in d.schedule_h], p_gone=round(float(d.p_gone),3),
                     p_success=round(float(d.p_success),3), ev=round(float(d.expected_value_usd),2),
                     propensity=round(float(d.propensity),3), rationale=d.rationale))
hard_rows = inv_te[inv_te["decline_reason"].isin(HARD)].sample(6, random_state=1)
for _, inv in hard_rows.iterrows():
    rows.append(dict(id=inv["invoice_id"], reason=inv["decline_reason"], rail=inv["rail"], market=inv["market"],
                     amount=round(float(inv["amount_usd"]),2), advice=inv["network_advice"], action="route_to_update_method",
                     delay=None, schedule=[], p_gone=None, p_success=None, ev=0.0, propensity=None, rationale="hard decline"))

lens = pd.Series([len(s) for s in plan.schedule]).value_counts().sort_index()
actions = pd.Series([r["action"] for r in rows]).value_counts()
out = dict(
    generated_at=now(), source="simulator (SimConfig n_invoices=6000, seed=5) — stand-in for the Bachs audit log",
    gate=dict(deploy=bool(g.deploy), delta=g.delta, lo=g.lo, hi=g.hi, ess=g.ess, min_ess=100.0, reason=g.reason,
              dr_planned=ope_plan.dr, dr_incumbent=ope_inc.dr, n_logged=int(len(logged))),
    model=dict(fitted_at=fitted_at, n_train=int(len(train)), n_test=int(len(test)), pr_auc=rep.pr_auc,
               roc_auc=rep.roc_auc, ece=rep.ece, base_rate=rep.base_rate,
               p_gone_model=float(pi_inv.mean()), p_gone_true=float(truth_inv.mean()),
               churn_train=float(inv_tr["_churned"].mean()), churn_test=float(inv_te["_churned"].mean())),
    value=dict(ladder=v_lad, planned=v_plan, lift=v_plan/v_lad-1),
    coverage=dict(labels=labels, log=log_mass.tolist(), planned=plan_mass.tolist(), ladder_buckets=[int(np.searchsorted(edges, h, side="right")-1) for h in LADDER]),
    weekly=weekly,
    schedule_len={int(k): int(v) for k,v in lens.items()},
    open_invoices=rows,
    counts=dict(open=int(len(sample)+len(hard_rows)), retryable=int((~inv_te["decline_reason"].isin(HARD)).sum()), hard=int(inv_te["decline_reason"].isin(HARD).sum())),
    runtime_s=time.time()-t0,
)
import os; os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
tmp = OUT + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, default=float)
import os; os.replace(tmp, OUT)  # atomic: the web server never serves a half-written file
print(json.dumps({k:v for k,v in out.items() if k not in ("open_invoices","weekly","coverage")}, indent=1, default=str))
