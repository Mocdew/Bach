"""Tests for the invariants that would silently cost money if they broke."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import pytest
except ImportError:  # minimal stand-in so the bundled runner works offline
    import re as _re
    import warnings as _warnings
    from contextlib import contextmanager
    from types import SimpleNamespace

    @contextmanager
    def _warns(category, match=None):
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            yield caught
        hits = [w for w in caught if issubclass(w.category, category)
                and (match is None or _re.search(match, str(w.message)))]
        assert hits, f"expected {category.__name__} matching {match!r}"

    @contextmanager
    def _raises(exc, match=None):
        try:
            yield
        except exc as e:
            assert match is None or _re.search(match, str(e)),                 f"{e!r} does not match {match!r}"
        else:
            raise AssertionError(f"expected {exc.__name__}")

    pytest = SimpleNamespace(fixture=lambda **kw: (lambda f: f),
                             warns=_warns, raises=_raises)

from recoup import (
    CureHazardModel, NetworkRules, PolicyConfig, RetryPlanner, RetryPolicy, SimConfig,
    SupportMap, TableDisputeModel, Truth, build_features, customer_history,
    cross_fitted_plan, deployment_gate, fixed_ladder_policy, ladder_schedules,
    derive_cure_labels, logged_first_attempts, off_policy_value, oracle_policy_value,
    oracle_schedule_value, simulate, temporal_split,
)
from recoup.bachs import DECLINE_CODE_MAP, map_decline_code
from recoup.domain import (
    HARD_DECLINES, MAX_HORIZON_H, N_DT_BUCKETS, NO_RETRY_ADVICE, DeclineReason,
    NetworkAdvice, Rail, dt_bucket, is_retryable,
)
from recoup.evaluate import expected_calibration_error, score_model
from recoup.models import CURE_LABEL, BetaBinomialHazard, GBMHazard
from recoup.simulator import true_dispute_prob, true_success_prob

LADDER = (24.0, 72.0, 168.0)


@pytest.fixture(scope="module")
def sim():
    inv, att = simulate(SimConfig(n_invoices=5000, seed=11))
    return customer_history(inv, att), att


@pytest.fixture(scope="module")
def fitted(sim):
    invoices, attempts = sim
    inv_tr, att_tr, inv_te, att_te, _ = temporal_split(invoices, attempts, 0.7)
    train = build_features(inv_tr, att_tr[["invoice_id", "attempt_index", "delay_hours"]])
    train["success"] = att_tr["success"].to_numpy()
    model = CureHazardModel().fit(train)
    return model, train, inv_te, att_te


def _soft(inv):
    return inv[~inv["decline_reason"].isin([d.value for d in HARD_DECLINES])
               & ~inv["network_advice"].isin([a.value for a in NO_RETRY_ADVICE])]


# --- domain ----------------------------------------------------------------


def test_hard_declines_are_not_retryable():
    for r in DeclineReason:
        assert is_retryable(r) == (r not in HARD_DECLINES)


def test_dt_buckets_cover_the_horizon_and_are_monotone():
    h = np.array([0.0, 1.0, 5.0, 23.9, 24.0, 100.0, MAX_HORIZON_H - 1])
    b = dt_bucket(h)
    assert b.min() >= 0 and b.max() < N_DT_BUCKETS
    assert np.all(np.diff(b) >= 0)


def test_network_rules_cap_wallet_rails_tighter():
    rules = NetworkRules()
    assert rules.attempt_cap(Rail.MOBILE_MONEY) <= rules.attempt_cap(Rail.CARD)


# --- simulator -------------------------------------------------------------


def test_hard_declines_never_recover():
    for r in HARD_DECLINES:
        p = true_success_prob(
            reason=r, rail=Rail.CARD, market_code="NG", fail_time_h=0.0,
            attempt_time_h=48.0, attempt_index=0, customer_quality=1.0,
            amount_usd=50.0, churned=False,
        )
        assert p == 0.0


def test_do_not_retry_advice_never_recovers():
    p = true_success_prob(
        reason=DeclineReason.INSUFFICIENT_FUNDS, rail=Rail.CARD, market_code="NG",
        fail_time_h=0.0, attempt_time_h=48.0, attempt_index=0, customer_quality=1.0,
        amount_usd=10.0, churned=False, advice=NetworkAdvice.DO_NOT_RETRY,
    )
    assert p == 0.0


def test_hazard_is_hump_shaped_not_monotone():
    delays = np.linspace(1, MAX_HORIZON_H - 1, 120)
    ps = [true_success_prob(
        reason=DeclineReason.ISSUER_UNAVAILABLE, rail=Rail.CARD, market_code="NG",
        fail_time_h=0.0, attempt_time_h=float(d), attempt_index=0,
        customer_quality=0.7, amount_usd=40.0, churned=False,
    ) for d in delays]
    peak = int(np.argmax(ps))
    assert 0 < peak < len(ps) - 1
    assert ps[-1] < ps[peak] * 0.5


def test_perturbed_truth_removes_payday_effect():
    kw = dict(reason=DeclineReason.INSUFFICIENT_FUNDS, rail=Rail.CARD, market_code="NG",
              fail_time_h=0.0, attempt_index=0, customer_quality=0.7, amount_usd=40.0,
              churned=False, truth=Truth(payday_strength=0.0))
    # With payday removed the balance curve depends only on elapsed time, so
    # two attempts at the same delay from different failure dates agree.
    a = true_success_prob(attempt_time_h=100.0, **kw)
    kw["fail_time_h"] = 24.0 * 15
    b = true_success_prob(attempt_time_h=24.0 * 15 + 100.0, **kw)
    assert abs(a - b) < 1e-9


def test_customers_recur_and_churn_is_sticky(sim):
    invoices, _ = sim
    per_cust = invoices.groupby("customer_id").size()
    assert per_cust.max() > 1
    # once churned, a customer's later invoices are churned too
    for _, g in invoices.sort_values("fail_time_h").groupby("customer_id"):
        c = g["_churned"].to_numpy()
        assert np.all(np.diff(c.astype(int)) >= 0)


def test_propensities_are_valid(sim):
    _, attempts = sim
    p = attempts["propensity"]
    assert p.between(0, 1).all() and (p > 0).all()


# --- features --------------------------------------------------------------


def test_feature_path_is_identical_for_training_and_scoring(sim):
    invoices, attempts = sim
    a = attempts.iloc[:50][["invoice_id", "attempt_index", "delay_hours"]]
    f1 = build_features(invoices, a)
    f2 = build_features(invoices, a.copy())
    pd.testing.assert_frame_equal(f1, f2)


def test_no_simulator_shaped_feature_ships():
    """The old payday_window was the simulator's funding function verbatim."""
    from recoup.features import FEATURES
    assert "payday_window" not in FEATURES


def test_features_module_does_not_import_simulator():
    import importlib, sys
    sys.modules.pop("recoup.features", None)
    before = set(sys.modules)
    importlib.import_module("recoup.features")
    assert "recoup.simulator" not in (set(sys.modules) - before)


def test_calendar_features_use_local_time(sim):
    invoices, _ = sim
    inv = invoices.iloc[[0]].copy()
    hours = []
    for mkt in ("NG", "US"):
        i = inv.copy()
        i["market"] = mkt
        f = build_features(i, pd.DataFrame({
            "invoice_id": i["invoice_id"], "attempt_index": 0, "delay_hours": 24.0}))
        hours.append(float(f["local_hour"].iloc[0]))
    assert hours[0] != hours[1]


def test_customer_history_is_leak_free(sim):
    invoices, attempts = sim
    # An invoice's history must only count invoices whose outcome was known
    # before this invoice failed.
    known = attempts.groupby("invoice_id")["attempt_time_h"].max()
    inv = invoices.set_index("invoice_id")
    for cust, g in invoices.groupby("customer_id"):
        g = g.sort_values("fail_time_h")
        for _, row in g.iterrows():
            earlier = g[(g["invoice_id"] != row["invoice_id"])]
            earlier = earlier[earlier["invoice_id"].map(known).fillna(np.inf) < row["fail_time_h"]]
            assert row["n_prior_failures"] == len(earlier)
        if len(g) > 3:
            break


def test_no_nans_in_features(sim):
    invoices, attempts = sim
    f = build_features(invoices, attempts[["invoice_id", "attempt_index", "delay_hours"]])
    from recoup.features import NUMERIC
    assert not f[NUMERIC].isna().any().any()


# --- models ----------------------------------------------------------------


def test_cure_model_identifies_churn_fraction(fitted, sim):
    model, train, inv_te, att_te = fitted
    test = build_features(inv_te, att_te[["invoice_id", "attempt_index", "delay_hours"]])
    pi, _ = model.predict_components(test)
    truth = test.merge(inv_te[["invoice_id", "_churned"]], on="invoice_id")["_churned"].mean()
    assert abs(pi.mean() - truth) < 0.12


def test_cure_model_beats_the_base_rate_and_is_calibrated(fitted):
    model, train, inv_te, att_te = fitted
    test = build_features(inv_te, att_te[["invoice_id", "attempt_index", "delay_hours"]])
    y = att_te["success"].to_numpy()
    r = score_model(y, model.predict_proba1(test))
    assert r.pr_auc > y.mean() * 1.3
    assert r.ece < 0.06


def test_cure_model_shrinks_small_merchants_to_the_pool(fitted):
    model, train, _, _ = fitted
    tab = model.coef_table()
    m = tab[tab["term"].str.startswith("mch:")]
    counts = train["merchant_id"].astype(str).value_counts()
    small = [f"mch:{k}" for k, v in counts.items() if v < 60]
    if small:
        assert m[m["term"].isin(small)]["coef"].abs().max() < 0.5


def test_posterior_samples_are_finite_and_centred(fitted):
    model, _, _, _ = fitted
    th = model.sample_params(50, np.random.default_rng(0))
    assert np.isfinite(th).all()
    assert np.abs(th.mean(0) - model.theta_).max() < 1.0


def test_beta_binomial_conditions_on_attempt_index(fitted):
    _, train, _, _ = fitted
    bb = BetaBinomialHazard().fit(train)
    keys = set(bb.level1_.keys())
    assert any(k[1] == 1 for k in keys), "attempt index must be part of the cell key"


def test_gbm_is_monotone_in_attempt_index(fitted):
    _, train, _, _ = fitted
    g = GBMHazard(max_iter=80).fit(train)
    sub = train.head(200).copy()
    p0 = g.predict_proba1(sub.assign(attempt_index=0))
    p2 = g.predict_proba1(sub.assign(attempt_index=2))
    assert np.all(p2 <= p0 + 1e-9)


# --- policy ----------------------------------------------------------------


def test_hard_declines_are_routed_not_retried(fitted):
    model, train, inv_te, _ = fitted
    pol = RetryPolicy(model, inv_te, SupportMap(train), PolicyConfig(n_posterior_samples=0))
    hard = inv_te[inv_te["decline_reason"].isin([d.value for d in HARD_DECLINES])]
    for iid in hard["invoice_id"].head(10):
        d = pol.decide(iid)
        assert d.action == "route_to_update_method" and d.delay_hours is None


def test_network_advice_blocks_retry(fitted):
    model, train, inv_te, _ = fitted
    inv = inv_te.copy()
    soft = _soft(inv)["invoice_id"].head(5)
    inv.loc[inv["invoice_id"].isin(soft), "network_advice"] = NetworkAdvice.DO_NOT_RETRY.value
    pol = RetryPolicy(model, inv, SupportMap(train), PolicyConfig(n_posterior_samples=0))
    for iid in soft:
        assert pol.decide(iid).action == "route_to_update_method"


def test_attempt_cap_is_never_exceeded(fitted):
    model, train, inv_te, _ = fitted
    pol = RetryPolicy(model, inv_te, SupportMap(train), PolicyConfig(max_attempts=3, n_posterior_samples=0))
    assert pol.decide(_soft(inv_te)["invoice_id"].iloc[0], attempt_index=3).action == "stop"


def test_planned_schedules_respect_caps_and_spacing(fitted):
    model, train, inv_te, _ = fitted
    cfg = PolicyConfig(n_posterior_samples=0, min_spacing_h=6.0)
    sub = inv_te.head(150).reset_index(drop=True)
    plan = RetryPlanner(model, cfg, support=SupportMap(train)).plan(sub)
    rules = NetworkRules()
    for (_, inv), s in zip(sub.iterrows(), plan.schedule):
        assert len(s) <= min(cfg.max_attempts, rules.attempt_cap(Rail(inv["rail"])))
        if len(s) > 1:
            assert np.all(np.diff(s) >= 6.0 - 1e-9)


def test_quiet_hours_are_respected(fitted):
    model, train, inv_te, _ = fitted
    cfg = PolicyConfig(quiet_hours_local=(0, 23), n_posterior_samples=0)
    sub = _soft(inv_te).head(40).reset_index(drop=True)
    plan = RetryPlanner(model, cfg, support=SupportMap(train)).plan(sub)
    for i, s in enumerate(plan.schedule):
        for h in s:
            f = build_features(sub.iloc[[i]], pd.DataFrame({
                "invoice_id": [sub["invoice_id"].iloc[i]], "attempt_index": [0], "delay_hours": [h]}))
            assert not (0 <= float(f["local_hour"].iloc[0]) < 23)


def test_support_guard_blocks_unobserved_delays(fitted):
    model, train, inv_te, _ = fitted
    strict = SupportMap(train, min_support=10 ** 9)
    pol = RetryPolicy(model, inv_te, strict, PolicyConfig(n_posterior_samples=0))
    for iid in _soft(inv_te)["invoice_id"].head(5):
        assert pol.decide(iid).action == "stop"


def test_policy_propensities_are_distributions(fitted):
    model, train, inv_te, _ = fitted
    sub = inv_te.head(80).reset_index(drop=True)
    plan = RetryPlanner(model, PolicyConfig(n_posterior_samples=4), support=SupportMap(train)).plan(sub)
    sums = plan.pi.sum(axis=1)
    assert np.all(np.isclose(sums, 1.0) | np.isclose(sums, 0.0))


def test_thompson_keeps_every_viable_action_reachable(fitted):
    model, train, inv_te, _ = fitted
    sub = inv_te.head(80).reset_index(drop=True)
    cfg = PolicyConfig(n_posterior_samples=4, propensity_floor=0.02)
    plan = RetryPlanner(model, cfg, support=SupportMap(train)).plan(sub)
    rows = plan.pi.sum(1) > 0
    live, viable = plan.pi[rows], plan.viable[rows]
    assert np.all(live[viable] > 0)
    multi = live[(live > 0).sum(1) > 1]
    assert len(multi) > 0 and multi.max() < 1.0


def test_dispute_cost_shortens_chosen_delays(fitted):
    """Dispute risk grows with elapsed time, so pricing it must pull the
    *schedule* in. Asserted on the whole schedule, not the first attempt: a
    planner facing an expensive tail can rationally take a slightly later
    first attempt in exchange for a much shorter tail (measured: first delay
    35.7h -> 36.1h while the last attempt moves 134h -> 123h at a $400 fee).
    The first-attempt version of this test passed by luck of the fixture."""
    model, train, inv_te, _ = fitted
    sub = inv_te.head(150).reset_index(drop=True)
    sup = SupportMap(train)
    lo = RetryPlanner(model, PolicyConfig(dispute_fee_usd=0.0, n_posterior_samples=0), support=sup).plan(sub)
    hi = RetryPlanner(model, PolicyConfig(dispute_fee_usd=400.0, n_posterior_samples=0), support=sup).plan(sub)
    live = [i for i in range(len(sub)) if lo.schedule[i] and hi.schedule[i]]
    assert len(live) > 50
    last = lambda p: np.mean([p.schedule[i][-1] for i in live])
    every = lambda p: np.mean([h for i in live for h in p.schedule[i]])
    assert last(hi) < last(lo)
    assert every(hi) < every(lo)


def test_annoyance_cost_shortens_wallet_schedules(fitted):
    model, train, inv_te, _ = fitted
    sub = inv_te[inv_te["rail"] == Rail.MOBILE_MONEY.value].head(120).reset_index(drop=True)
    if sub.empty:
        return
    sup = SupportMap(train)
    cheap = RetryPlanner(model, PolicyConfig(annoyance_cost_usd=(), n_posterior_samples=0), support=sup).plan(sub)
    dear = RetryPlanner(model, PolicyConfig(annoyance_cost_usd=((Rail.MOBILE_MONEY.value, 5.0),),
                                            n_posterior_samples=0), support=sup).plan(sub)
    assert np.mean([len(s) for s in dear.schedule]) <= np.mean([len(s) for s in cheap.schedule])


def test_dispute_model_is_injected(fitted):
    """The policy must run with any callable; it must not reach for the oracle."""
    import recoup.policy as pol
    assert "true_dispute_prob" not in vars(pol)
    model, train, inv_te, _ = fitted
    seen = []
    def dm(k, t, rail):
        seen.append(k); return 0.0
    sub = _soft(inv_te).head(20).reset_index(drop=True)
    RetryPlanner(model, PolicyConfig(n_posterior_samples=0), dispute_model=dm,
                 support=SupportMap(train)).plan(sub)
    assert seen
    assert 0 <= TableDisputeModel()(0, 24.0, Rail.CARD) < 0.1


# --- evaluation ------------------------------------------------------------


def test_temporal_split_does_not_leak(sim):
    invoices, attempts = sim
    inv_tr, att_tr, inv_te, att_te, cutoff = temporal_split(invoices, attempts, 0.7)
    assert inv_tr["fail_time_h"].max() <= cutoff < inv_te["fail_time_h"].min()
    assert not set(att_tr["invoice_id"]) & set(att_te["invoice_id"])


def test_calibration_error_is_zero_for_a_perfect_model():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 20000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.02


def test_dr_recovers_the_logging_policy_value(sim):
    from recoup.simulator import logging_policy_propensities
    cfg = PolicyConfig()
    inv, first = logged_first_attempts(*sim, cfg)
    pi = np.tile(logging_policy_propensities(SimConfig(), 0), (len(inv), 1))
    q = np.zeros((len(inv), N_DT_BUCKETS))
    res = off_policy_value(first, pi, q, cfg)
    assert abs(res.snips - first["reward"].mean()) < 0.5


def test_planner_beats_the_fixed_ladder_on_the_schedule(fitted):
    model, train, inv_te, _ = fitted
    sub = inv_te.head(400).reset_index(drop=True)
    cfg = PolicyConfig(n_posterior_samples=0)
    plan = RetryPlanner(model, cfg, support=SupportMap(train)).plan(sub)
    learned = oracle_schedule_value(sub, plan.schedule, cfg)
    ladder = oracle_schedule_value(sub, ladder_schedules(sub, LADDER, 3), cfg)
    assert learned > ladder


def test_planning_ahead_is_not_worse_than_greedy(fitted):
    model, train, inv_te, _ = fitted
    sub = inv_te.head(300).reset_index(drop=True)
    planner = RetryPlanner(model, PolicyConfig(n_posterior_samples=0), support=SupportMap(train))
    seq = planner.plan(sub).schedule
    grd = planner.plan_greedy(sub)
    cfg = PolicyConfig()
    assert oracle_schedule_value(sub, seq, cfg) >= oracle_schedule_value(sub, grd, cfg) - 0.05


def test_cross_fitted_dr_tracks_the_oracle(fitted):
    model, train, inv_te, att_te = fitted
    cfg = PolicyConfig(n_posterior_samples=4)
    inv, first = logged_first_attempts(inv_te, att_te, cfg)
    inv, first = inv.head(600).reset_index(drop=True), first.head(600).reset_index(drop=True)
    pi, dl, q = cross_fitted_plan(train, inv, lambda d: CureHazardModel().fit(d),
                                  lambda m: RetryPlanner(m, cfg, support=SupportMap(train)))
    res = off_policy_value(first, pi, q, cfg, oracle=oracle_policy_value(inv, pi, dl, cfg))
    # q_hat must be in first-attempt units: the direct-method term alone should
    # sit near the oracle, and DR should not be wildly off despite thin ESS.
    dm = float(np.mean(np.sum(pi * q, axis=1)))
    assert abs(dm - res.oracle) < 0.4 * abs(res.oracle) + 0.5
    assert abs(res.dr - res.oracle) < 0.6 * abs(res.oracle) + 1.0


# --- adapter ---------------------------------------------------------------


def test_unknown_decline_codes_fail_closed():
    assert map_decline_code("some_code_bachs_added_last_week") in HARD_DECLINES
    assert map_decline_code(None) in HARD_DECLINES


def test_known_soft_codes_stay_retryable():
    for code, reason in DECLINE_CODE_MAP.items():
        if reason not in HARD_DECLINES:
            assert is_retryable(map_decline_code(code)), code


# --- deployment gate -------------------------------------------------------


def test_gate_holds_when_the_candidate_is_the_incumbent(sim):
    from recoup.simulator import logging_policy_propensities
    cfg = PolicyConfig()
    inv, first = logged_first_attempts(*sim, cfg)
    pi = np.tile(logging_policy_propensities(SimConfig(), 0), (len(inv), 1))
    q = np.zeros((len(inv), N_DT_BUCKETS))
    g = deployment_gate(first, pi, pi, q, cfg, n_boot=120)
    assert not g.deploy and abs(g.delta) < 1e-9


def test_gate_holds_on_thin_coverage(sim):
    from recoup.simulator import logging_policy_propensities
    cfg = PolicyConfig()
    inv, first = logged_first_attempts(*sim, cfg)
    incumbent = np.tile(logging_policy_propensities(SimConfig(), 0), (len(inv), 1))
    candidate = np.zeros((len(inv), N_DT_BUCKETS))
    candidate[:, N_DT_BUCKETS - 1] = 1.0
    q = np.zeros((len(inv), N_DT_BUCKETS))
    g = deployment_gate(first, candidate, incumbent, q, cfg, min_ess=200, n_boot=120)
    assert not g.deploy and "sample size" in g.reason


def test_gate_interval_brackets_the_point_estimate(sim):
    from recoup.simulator import logging_policy_propensities
    cfg = PolicyConfig()
    inv, first = logged_first_attempts(*sim, cfg)
    incumbent = np.tile(logging_policy_propensities(SimConfig(), 0), (len(inv), 1))
    candidate = np.roll(incumbent, 1, axis=1)
    q = np.zeros((len(inv), N_DT_BUCKETS))
    g = deployment_gate(first, candidate, incumbent, q, cfg, min_ess=0, n_boot=200)
    assert g.lo <= g.delta <= g.hi


# --- semi-supervised cure component ----------------------------------------


def test_cure_labels_from_later_successes_are_exact(sim):
    """A later success proves the customer was not gone. Churn is absorbing in
    the simulator, so this label admits no exceptions -- if it ever does, the
    derivation is looking at the wrong window."""
    invoices, attempts = sim
    lab = derive_cure_labels(invoices, attempts)
    present = lab == 0.0
    assert present.sum() > 100, "expected a usable number of present labels"
    assert invoices.loc[present, "_churned"].mean() == 0.0


def test_cure_labels_need_real_events_to_call_anyone_gone(sim):
    """Without observed instrument-death events nothing is labelled gone. The
    heuristic that used to fill this in -- a later hard decline on the same
    customer -- measured at the base rate, i.e. pure noise, and was removed."""
    invoices, attempts = sim
    assert (derive_cure_labels(invoices, attempts) == 1.0).sum() == 0

    events = invoices.groupby("customer_id")["fail_time_h"].max().reset_index()
    events.columns = ["customer_id", "event_time_h"]
    events["event_time_h"] += 1.0
    lab = derive_cure_labels(invoices, attempts, gone_events=events)
    assert (lab == 1.0).sum() > 0


def test_known_gone_invoices_do_not_shape_the_hazard(fitted):
    """w_live=0 must remove a known-gone invoice from the hazard likelihood.
    If it does not, dead instruments drag the timing curve down."""
    model, train, _, _ = fitted
    recovered = train.groupby("invoice_id")["success"].max()
    # Only a never-recovered invoice can honestly be labelled gone; labelling a
    # recovered one is the contradiction the guard downgrades.
    never = list(recovered[recovered == 0].index)
    ever = list(recovered[recovered > 0].index)
    lab = {i: 1.0 for i in never[: len(never) // 2]}
    lab.update({i: 0.0 for i in never[len(never) // 2:] + ever})
    labelled = train.copy()
    labelled[CURE_LABEL] = labelled["invoice_id"].map(lab)
    m = CureHazardModel().fit(labelled)
    assert m.n_known_gone_ > 0 and m.n_known_live_ > 0
    assert m.n_known_gone_ + m.n_known_live_ == len(lab)
    assert m.n_known_gone_ == len(never) // 2
    pi, h = m.predict_components(train)
    assert np.all(np.isfinite(pi)) and np.all(np.isfinite(h))


def test_contradictory_gone_label_is_refused_not_obeyed(fitted):
    """known_gone=1 on a recovered invoice is a broken label. It must be
    downgraded with a warning, not drive the likelihood to zero."""
    model, train, _, _ = fitted
    recovered = train.groupby("invoice_id")["success"].max()
    rec_ids = set(recovered[recovered > 0].index[:20])
    bad = train.copy()
    bad[CURE_LABEL] = np.where(bad["invoice_id"].isin(rec_ids), 1.0, 0.0)
    with pytest.warns(RuntimeWarning, match="were recovered"):
        m = CureHazardModel().fit(bad)
    assert m.n_known_gone_ == 0
    assert np.all(np.isfinite(m.theta_))


def test_one_sided_cure_labels_are_flagged(fitted):
    """Present-only labels bias the cure fraction down hard (measured 0.124 ->
    0.016 against a 0.225 truth), because labelling is outcome-dependent.
    Silence here would ship that."""
    model, train, _, _ = fitted
    one_sided = train.copy()
    one_sided[CURE_LABEL] = 0.0
    with pytest.warns(RuntimeWarning, match="one-sided"):
        CureHazardModel().fit(one_sided)


def test_unlabelled_fit_is_unchanged(fitted):
    """The semi-supervised path must be an exact no-op with no labels."""
    model, train, _, _ = fitted
    m = CureHazardModel().fit(train.assign(**{CURE_LABEL: np.nan}))
    assert m.n_known_gone_ == 0 and m.n_known_live_ == 0
    np.testing.assert_allclose(m.theta_, model.theta_, rtol=1e-6, atol=1e-8)


# --- cross-fitting ---------------------------------------------------------


def test_cross_fitting_refuses_more_than_two_folds(fitted):
    """With K>2 the leave-one-out models share folds pairwise, so no pair of
    them is disjoint and the guarantee does not hold."""
    model, train, inv_te, att_te = fitted
    with pytest.raises(ValueError, match="n_folds=2"):
        cross_fitted_plan(train, inv_te.head(5), lambda d: model,
                          lambda m: RetryPlanner(m, PolicyConfig()), n_folds=3)


def test_policy_and_reward_model_come_from_different_fits(fitted):
    """The bug this guards: averaging pi and q_hat over both fold-models makes
    every row depend on both again, handing back the optimiser's curse that
    cross-fitting exists to remove. Each row's q_hat must equal exactly one
    model's output -- an average of the two would match neither."""
    model, train, inv_te, att_te = fitted
    cfg = PolicyConfig(n_posterior_samples=0)
    ev = _soft(inv_te).head(40).reset_index(drop=True)

    fitted_models = []

    def fit_model(df):
        m = CureHazardModel().fit(df)
        fitted_models.append((m, len(df)))
        return m

    pi, dl, q = cross_fitted_plan(train, ev, fit_model,
                                  lambda m: RetryPlanner(m, cfg), seed=0)
    assert len(fitted_models) == 2
    # Disjoint halves: together they cover the training rows at most once.
    assert sum(n for _, n in fitted_models) <= len(train) + 1

    plans = [RetryPlanner(m, cfg).plan(ev, explore=False) for m, _ in fitted_models]
    match = [np.isclose(q, p.q_first).all(axis=1) for p in plans]
    assert np.all(match[0] | match[1]), "q_hat is not a single model's output"
    assert match[0].any() and match[1].any(), "both halves should be used"
    # And the row's policy comes from the *other* model than its q_hat.
    pol = [np.isclose(pi, p.pi).all(axis=1) for p in plans]
    for row in range(len(ev)):
        if match[0][row] and not match[1][row]:
            assert pol[1][row] or not pol[0][row]


# --- exploration sampling --------------------------------------------------


def test_logged_propensity_is_the_sampling_distribution(fitted):
    """The propensity written to the audit log must be the probability the
    action was actually drawn with -- it becomes the denominator of the next
    gate run. Exact by construction, not estimated."""
    model, train, inv_te, att_te = fitted
    cfg = PolicyConfig(n_propensity_samples=32)
    pol = RetryPolicy(model, inv_te, SupportMap(train), cfg,
                      rng=np.random.default_rng(4))
    checked = 0
    for iid in _soft(inv_te)["invoice_id"].head(25):
        d = pol.decide(iid)
        if d.action != "retry":
            continue
        row = d.curve[d.curve["bucket"] == d.bucket]
        assert np.isclose(float(row["propensity"].iloc[0]), d.propensity)
        assert np.isclose(d.curve["propensity"].sum(), 1.0)
        assert d.propensity > 0.0
        checked += 1
    assert checked > 5


def test_decide_uses_the_higher_propensity_sample_count(fitted):
    """Production logs one invoice at a time, where extra posterior draws cost
    milliseconds. Batch planning keeps the cheap setting."""
    model, train, inv_te, _ = fitted
    cfg = PolicyConfig(n_posterior_samples=16, n_propensity_samples=96)
    pol = RetryPolicy(model, inv_te, SupportMap(train), cfg)
    assert pol.planner.cfg.n_posterior_samples == 96
    assert cfg.n_posterior_samples == 16


def test_shortened_planners_are_cached(fitted):
    """Rebuilding a planner re-enumerates every candidate schedule; doing that
    per decision is pure waste in an hourly job."""
    model, train, inv_te, _ = fitted
    pol = RetryPolicy(model, inv_te, SupportMap(train),
                      PolicyConfig(n_propensity_samples=8))
    iid = _soft(inv_te)["invoice_id"].iloc[0]
    pol.decide(iid, attempt_index=1)
    first = next(iter(pol._shortened.values()))
    pol.decide(iid, attempt_index=1)
    assert len(pol._shortened) == 1
    assert next(iter(pol._shortened.values())) is first
