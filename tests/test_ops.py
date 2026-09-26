"""Tests for the path from raw Bachs-shaped data to a logged decision.

The modelling invariants live in ``test_recoup.py``. These guard the parts
that sit around the model and fail quietly: the adapter, the synthetic
world's consistency with its own ground truth, continuation decisions, the
propensities written to the audit log, and the jobs' guards.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pytest
except ImportError:  # the bundled runner supplies fixtures itself
    from types import SimpleNamespace
    pytest = SimpleNamespace(fixture=lambda **kw: (lambda f: f))

from recoup import SimConfig, simulate
from recoup.bachs import FIELD_MAP, from_payments, normalise_payments, validate_schema
from recoup.domain import HARD_DECLINES, MAX_HORIZON_H, N_DT_BUCKETS, NetworkAdvice, dt_bucket
from recoup.features import customer_history, derive_cure_labels
from recoup.ops import EXIT_OK, EXIT_REFUSED, _choose, main, training_table
from recoup.policy import PolicyConfig, RetryPolicy
from recoup.simulator import logging_policy_propensities
from recoup.store import load_bundle, load_dataset, read_audit, resolve_bundle
from recoup.synthetic import TRUTH_DIR, SynthConfig, advance_world, generate_dataset

SMALL = dict(n_invoices=2400, days=150, seed=13)
HISTORY_DAYS = 110.0


@pytest.fixture(scope="module")
def world():
    """A small synthetic dataset, a fitted bundle, and one plan run over it."""
    root = Path(tempfile.mkdtemp(prefix="recoup-test-"))
    data, state, logs = root / "data", root / "state", root / "log"
    generate_dataset(data, SynthConfig(sim=SimConfig(**SMALL), history_days=HISTORY_DAYS,
                                       seed=13))
    rc = main(["retrain", "--data", str(data), "--out", str(state / "models"),
               "--min-invoices", "200", "--holdout-frac", "0.15"])
    assert rc == EXIT_OK
    rc = main(["plan", "--data", str(data), "--model", str(state / "models" / "current"),
               "--gate", str(state / "gate.json"), "--audit", str(logs / "decisions"),
               "--queue", str(state / "queue"), "--propensity-samples", "8", "--seed", "1"])
    assert rc == EXIT_OK
    return dict(root=root, data=data, state=state, logs=logs)


def _truth(world) -> pd.DataFrame:
    return pd.read_csv(world["data"] / TRUTH_DIR / "invoices.csv").set_index("invoice_id")


def _payments(world) -> list[dict]:
    with open(world["data"] / "payments.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# --- simulator fix -----------------------------------------------------------


def test_logged_schedules_move_forward_in_time():
    """Attempt k+1 before attempt k is a schedule no system can execute, and
    it corrupts the join between the decision log and a time-ordered payment
    stream. The logging policy must only draw later delays."""
    _, att = simulate(SimConfig(n_invoices=2500, seed=4))
    gaps = att.sort_values(["invoice_id", "attempt_index"]).groupby("invoice_id")[
        "delay_hours"].diff().dropna()
    assert (gaps >= 2.0 - 1e-9).all()


def test_first_attempt_logging_policy_is_unchanged():
    """The fix must not touch attempt 0: the benchmark's first-decision OPE
    and every existing test read that distribution."""
    cfg = SimConfig()
    p = logging_policy_propensities(cfg, 0)
    expected = np.full(N_DT_BUCKETS, cfg.epsilon / N_DT_BUCKETS)
    expected[int(dt_bucket(cfg.ladder_h[0]))] += 1 - cfg.epsilon
    np.testing.assert_allclose(p, expected)


def test_continuation_propensities_exclude_the_past_and_sum_to_one():
    cfg = SimConfig()
    p = logging_policy_propensities(cfg, 1, prev_delay_h=100.0)
    assert np.isclose(p.sum(), 1.0)
    assert p[: int(dt_bucket(100.0))].sum() == 0.0


# --- adapter against the synthetic world's ground truth ----------------------


def test_synthetic_payments_pass_schema_validation(world):
    problems = validate_schema(_payments(world)[:2000])
    assert not [p for p in problems if p.startswith("ERROR")], problems


def test_schema_validation_catches_a_wrong_field_path(world):
    broken = [{k: v for k, v in p.items() if k != "reference"} for p in _payments(world)[:200]]
    assert any("invoice_ref" in p and p.startswith("ERROR") for p in validate_schema(broken))


def test_adapter_recovers_the_invoices_the_world_generated(world):
    ds = load_dataset(world["data"])
    truth = _truth(world)
    inv = ds.invoices.set_index("invoice_id")
    t = truth.loc[inv.index]
    assert len(inv) == int((truth["fail_time_h"] <= ds.as_of_h).sum())
    assert (inv["market"] == t["market"]).all()
    assert (inv["network_advice"] == t["network_advice"]).all()
    # FX round trip: minor units of local currency back to USD.
    assert np.allclose(inv["amount_usd"], t["amount_usd"], atol=0.01)
    # Unmapped raw codes may change *which* hard reason, never hard vs soft.
    hard = {d.value for d in HARD_DECLINES}
    assert (inv["decline_reason"].isin(hard) == t["decline_reason"].isin(hard)).all()
    assert (inv.loc[~t["decline_reason"].isin(hard), "decline_reason"]
            == t.loc[~t["decline_reason"].isin(hard), "decline_reason"]).all()
    assert ds.tables.report.unmapped_codes, "the dataset should exercise the unmapped report"


def test_attempts_line_up_with_the_legacy_decision_log(world):
    """Every logged legacy decision must find its attempt at the same index
    and bucket. This is the join the gate's propensities ride on."""
    ds = load_dataset(world["data"])
    att = ds.attempts.set_index(["invoice_id", "attempt_index"])
    leg = ds.legacy_decisions
    assert len(leg) == len(att)
    got = att.loc[list(zip(leg["invoice_id"], leg["attempt_index"])), "dt_bucket"].to_numpy()
    assert (got == leg["dt_bucket"].to_numpy()).all()


def test_attempts_stop_at_the_first_success(world):
    ds = load_dataset(world["data"])
    after = ds.attempts.sort_values(["invoice_id", "attempt_index"]).groupby("invoice_id")[
        "success"].apply(lambda s: s.cumsum().shift(fill_value=0).max())
    assert (after == 0).all()


def test_missing_fx_rate_is_reported_not_silent():
    p = {"id": "p1", "customer": {"id": "c1"}, "amount": 155000, "currency": "NGN",
         "status": "failed", "created_at": "2026-02-01T00:00:00Z", "reference": "i1"}
    _, report = normalise_payments([p], fx_to_usd={"USD": 1.0})
    assert report.unconverted_currencies == {"NGN": 1}


def test_as_of_hides_the_future(world):
    ds_now = load_dataset(world["data"])
    ds_then = load_dataset(world["data"], as_of_h=ds_now.as_of_h - 30 * 24.0)
    assert ds_then.invoices["fail_time_h"].max() <= ds_then.as_of_h
    assert ds_then.attempts["attempt_time_h"].max() <= ds_then.as_of_h
    assert len(ds_then.invoices) < len(ds_now.invoices)


def test_old_three_tuple_unpacking_still_works(world):
    invoices, attempts, report = from_payments(_payments(world)[:500])
    assert {"invoice_id", "customer_id"} <= set(invoices.columns)


# --- cure labels from the raw feed -------------------------------------------


def test_present_labels_from_renewals_are_exact(world):
    ds = load_dataset(world["data"])
    lab = derive_cure_labels(ds.invoices, ds.attempts, ds.tables.gone_events,
                             successes=ds.tables.successes, gone_within_h=7 * 24.0)
    churned = _truth(world).loc[ds.invoices["invoice_id"], "churned"].to_numpy()
    assert (lab == 0).sum() > 200
    assert churned[lab.to_numpy() == 0].mean() == 0.0


def test_gone_labels_beat_the_base_rate_and_skip_recovered_invoices(world):
    ds = load_dataset(world["data"])
    lab = derive_cure_labels(ds.invoices, ds.attempts, ds.tables.gone_events,
                             successes=ds.tables.successes, gone_within_h=7 * 24.0).to_numpy()
    churned = _truth(world).loc[ds.invoices["invoice_id"], "churned"].to_numpy()
    assert (lab == 1).sum() > 0
    assert churned[lab == 1].mean() > 3 * churned.mean()
    recovered = set(ds.attempts.loc[ds.attempts["success"] == 1, "invoice_id"])
    assert not set(ds.invoices["invoice_id"][lab == 1]) & recovered


# --- continuation decisions --------------------------------------------------


def _open_soft(ds):
    inv = customer_history(ds.invoices, ds.attempts)
    soft = inv[~inv["decline_reason"].isin([d.value for d in HARD_DECLINES])
               & (inv["network_advice"] == NetworkAdvice.NONE.value)]
    return inv, soft


def test_continuation_never_schedules_in_the_past(world):
    bundle = load_bundle(world["state"] / "models" / "current")
    ds = load_dataset(world["data"])
    inv, soft = _open_soft(ds)
    pol = RetryPolicy(bundle.model, inv, bundle.support,
                      PolicyConfig(n_propensity_samples=8), rng=np.random.default_rng(0))
    checked = 0
    for iid in soft["invoice_id"].head(12):
        d = pol.decide(iid, 1, past_delays=(30.0,), now_elapsed_h=40.0)
        if d.action == "retry":
            assert d.delay_hours >= 40.0
            assert all(h >= 40.0 for h in d.schedule_h)
            checked += 1
    assert checked > 0


def test_failures_so_far_raise_p_gone(world):
    """Each observed failure is more likely under 'gone'. A continuation
    decision that ignores them prices the next retry as if nothing had
    happened."""
    bundle = load_bundle(world["state"] / "models" / "current")
    ds = load_dataset(world["data"])
    inv, soft = _open_soft(ds)
    pol = RetryPolicy(bundle.model, inv, bundle.support,
                      PolicyConfig(n_propensity_samples=4), rng=np.random.default_rng(0))
    iid = soft["invoice_id"].iloc[0]
    p0 = pol.planner.plan(pol.invoices.loc[[iid]], explore=False).p_gone[0]
    fresh = pol._planner_for(2, 4).plan(pol.invoices.loc[[iid]], explore=False,
                                        attempt_offset=2).p_gone[0]
    after = pol._planner_for(2, 4).plan(pol.invoices.loc[[iid]], explore=False,
                                        attempt_offset=2, past_delays=[(24.0, 72.0)]).p_gone[0]
    assert np.isclose(fresh, p0)
    assert after > p0


# --- the plan job --------------------------------------------------------------


def test_plan_logs_a_distribution_it_actually_sampled_from(world):
    audit = read_audit(world["logs"] / "decisions")
    retry = audit[audit["action"] == "retry"]
    assert len(retry) > 0
    for _, r in retry.iterrows():
        vec = np.array(json.loads(r["propensities"]))
        assert np.isclose(vec.sum(), 1.0)
        assert np.isclose(vec[int(r["dt_bucket"])], r["propensity"])
        assert r["execute_at_h"] > r["decided_at_h"]


def test_hold_mixture_keeps_the_ladder_dominant():
    """Under HOLD at most eps of the mass may leave the ladder rung."""
    class D:  # a planner decision that disagrees with the ladder entirely
        action, delay_hours, bucket, propensity = "retry", 3.0, 1, 1.0
        p_gone = p_success = expected_value_usd = 0.0
        rationale = ""
        curve = pd.DataFrame({"bucket": range(N_DT_BUCKETS),
                              "delay_h": np.linspace(1, 300, N_DT_BUCKETS),
                              "propensity": np.eye(N_DT_BUCKETS)[1]})

    from recoup.domain import NetworkRules

    class B:
        ladder_h, rules = (24.0, 72.0, 168.0), NetworkRules()

    row = pd.Series(dict(decline_reason="insufficient_funds", network_advice="none",
                         rail="card", fail_time_h=10.0, market="GB"))
    rng = np.random.default_rng(0)
    picks = [_choose("HOLD", D(), row, 0, 1.0, B(), PolicyConfig(), 0.2, rng)
             for _ in range(400)]
    vec = np.array(picks[0]["vector"])
    assert np.isclose(vec.sum(), 1.0) and vec.max() >= 0.8 - 1e-9
    explored = np.mean([p["policy"] == "explore" for p in picks])
    assert 0.12 < explored < 0.28
    for p in picks:
        assert np.isclose(vec[p["bucket"]], p["propensity"])


def test_plan_is_idempotent(world):
    before = read_audit(world["logs"] / "decisions")
    rc = main(["plan", "--data", str(world["data"]),
               "--model", str(world["state"] / "models" / "current"),
               "--gate", str(world["state"] / "gate.json"),
               "--audit", str(world["logs"] / "decisions"),
               "--queue", str(world["state"] / "queue"), "--propensity-samples", "4"])
    assert rc == EXIT_OK
    assert len(read_audit(world["logs"] / "decisions")) == len(before)


def test_hard_declines_are_routed_by_the_plan_job(world):
    audit = read_audit(world["logs"] / "decisions")
    ds = load_dataset(world["data"])
    reason = ds.invoices.set_index("invoice_id")["decline_reason"]
    hard = audit[audit["invoice_id"].map(reason).isin([d.value for d in HARD_DECLINES])]
    assert len(hard) > 0
    assert (hard["action"] == "route_to_update_method").all()


# --- bundles and guards ----------------------------------------------------------


def test_bundle_pointer_resolves(world):
    d = resolve_bundle(world["state"] / "models" / "current")
    assert (d / "bundle.pkl").exists() and (d / "meta.json").exists()
    assert resolve_bundle(world["state"] / "models") == d
    meta = json.loads((d / "meta.json").read_text())
    assert meta["holdout"]["pr_auc"] > meta["holdout"]["base_rate"]


def test_retrain_refuses_on_too_little_data_and_keeps_the_current_model(world):
    before = resolve_bundle(world["state"] / "models" / "current")
    rc = main(["retrain", "--data", str(world["data"]), "--out",
               str(world["state"] / "models"), "--min-invoices", "10000000"])
    assert rc == EXIT_REFUSED
    assert resolve_bundle(world["state"] / "models" / "current") == before


def test_gate_holds_when_evidence_is_thin(world):
    out = world["state"] / "gate-thin.json"
    rc = main(["gate", "--data", str(world["data"]),
               "--model", str(world["state"] / "models" / "current"),
               "--audit", str(world["logs"] / "decisions"), "--out", str(out),
               "--min-decisions", "10000000"])
    assert rc == EXIT_OK
    assert json.loads(out.read_text())["decision"] == "HOLD"


# --- the synthetic world moving forward ------------------------------------------


def test_advance_executes_due_retries_and_drops_stale_ones():
    root = Path(tempfile.mkdtemp(prefix="recoup-adv-"))
    try:
        data, queue = root / "data", root / "queue"
        generate_dataset(data, SynthConfig(sim=SimConfig(n_invoices=600, days=100, seed=2),
                                           history_days=60.0, seed=2))
        ds = load_dataset(data)
        open_inv = ds.invoices[(ds.as_of_h - ds.invoices["fail_time_h"] < 48)
                               & ~ds.invoices["decline_reason"].isin(
                                   [d.value for d in HARD_DECLINES])]
        tried = set(ds.attempts["invoice_id"])
        fresh = open_inv[~open_inv["invoice_id"].isin(tried)].head(3)
        assert len(fresh) > 0
        queue.mkdir()
        items = [dict(invoice_id=i, attempt_index=0, action="retry",
                      execute_at_h=ds.as_of_h + 5.0) for i in fresh["invoice_id"]]
        items.append(dict(invoice_id=fresh["invoice_id"].iloc[0], attempt_index=3,
                          action="retry", execute_at_h=ds.as_of_h + 6.0))
        with open(queue / "q.jsonl", "w") as f:
            f.writelines(json.dumps(i) + "\n" for i in items)
        s = advance_world(data, queue, 12.0)
        assert s["retries"] == len(fresh) and s["stale"] == 1
        after = load_dataset(data)
        assert after.as_of_h == ds.as_of_h + 12.0
        got = after.attempts[after.attempts["invoice_id"].isin(set(fresh["invoice_id"]))]
        assert len(got) == len(fresh)
        assert np.allclose(got["attempt_time_h"], ds.as_of_h + 5.0, atol=1e-3)
        # A second advance over a later window must not re-execute them.
        assert advance_world(data, queue, 12.0)["retries"] == 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_training_table_excludes_open_windows(world):
    ds = load_dataset(world["data"])
    inv = customer_history(ds.invoices, ds.attempts)
    closed, _, feats = training_table(ds, inv, ds.as_of_h)
    assert (closed["fail_time_h"] + MAX_HORIZON_H <= ds.as_of_h).all()
    assert set(feats["invoice_id"]) <= set(closed["invoice_id"])


# --- console API ------------------------------------------------------------------
# Optional extra: skipped when fastapi is not installed.


def _client(world):
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        return None
    from recoup.api.app import create_app
    from recoup.api.service import Paths
    s, lg = world["state"], world["logs"]
    paths = Paths(data=world["data"], models=s / "models", gate=s / "gate.json",
                  audit=lg / "decisions", queue=s / "queue")
    return TestClient(create_app(paths, static_dir=Path("__none__")))


def test_api_contract_matches_the_generated_frontend_types():
    """web/openapi.json is what the TypeScript client was generated from. If
    schemas.py changes without `npm run gen:api`, the frontend is compiled
    against a contract the server no longer honours."""
    try:
        from recoup.api.app import create_app
    except ImportError:
        return
    from recoup.api.service import Paths
    committed = Path(__file__).resolve().parents[1] / "web" / "openapi.json"
    nowhere = Path("__none__")
    live = create_app(Paths(nowhere, nowhere, nowhere, nowhere, nowhere),
                      static_dir=nowhere).openapi()
    assert json.loads(committed.read_text(encoding="utf-8")) == json.loads(json.dumps(live)), \
        "API contract drifted: run `npm run gen:api` in web/"


def test_api_overview_and_pages(world):
    c = _client(world)
    if c is None:
        return
    o = c.get("/api/overview").json()
    assert o["synthetic"] is True and o["gate"]["decision"] == "HOLD"
    assert o["open_invoices"] >= o["queued_retries"]
    page = c.get("/api/decisions", params={"limit": 7, "offset": 3}).json()
    assert len(page["rows"]) == 7 and page["offset"] == 3
    assert sum(page["facets"]["mode"].values()) == page["total"]
    legacy = c.get("/api/decisions", params={"mode": "LEGACY", "limit": 5}).json()
    assert legacy["total"] > 0 and all(r["policy"] == "legacy" for r in legacy["rows"])
    inv = c.get("/api/invoices", params={"status": "open"}).json()
    assert inv["total"] > 0 and all(r["status"] == "open" for r in inv["rows"])
    cov = c.get("/api/decisions/coverage").json()
    for shares in cov["by_mode"].values():
        assert abs(sum(shares) - 1.0) < 1e-9
    assert c.get("/api/invoices/does_not_exist").status_code == 404


def test_api_explanation_is_the_best_plan_not_the_draw(world):
    """The inspector reports the planner's best first bucket and says how
    often exploration departs from it -- not whichever bucket one posterior
    draw happened to pick."""
    c = _client(world)
    if c is None:
        return
    rows = c.get("/api/invoices", params={"status": "open", "reason_class": "balance",
                                          "limit": 20}).json()["rows"]
    checked = 0
    for r in rows:
        ex = c.get(f"/api/invoices/{r['invoice_id']}").json()["explanation"]
        if not ex or ex["action"] != "retry":
            continue
        viable = [p for p in ex["curve"] if p["viable"]]
        best = max(viable, key=lambda p: p["value_usd"])
        assert ex["best_bucket"] == best["bucket"]
        assert abs(ex["expected_value_usd"] - best["value_usd"]) < 1e-9
        assert abs(ex["explore_mass"] - (1 - best["propensity"])) < 1e-9
        assert ex["schedule_h"][0] == best["delay_h"]
        checked += 1
    assert checked > 0


def test_api_refuses_simulation_on_real_data(world):
    """No clock-advancing on a dataset without ground truth: 409, not a crash
    and not invented outcomes."""
    c = _client(world)
    if c is None:
        return
    real = world["root"] / "real"
    if not real.exists():
        shutil.copytree(world["data"], real, ignore=shutil.ignore_patterns(TRUTH_DIR))
        m = json.loads((real / "manifest.json").read_text())
        m["kind"] = "bachs-export"
        (real / "manifest.json").write_text(json.dumps(m))
    from fastapi.testclient import TestClient
    from recoup.api.app import create_app
    from recoup.api.service import Paths
    s = world["state"]
    rc = TestClient(create_app(Paths(real, s / "models", s / "gate.json",
                                     world["logs"] / "decisions", s / "queue"),
                               static_dir=Path("__none__")))
    assert rc.get("/api/sim").json()["synthetic"] is False
    assert rc.post("/api/sim/step", json={"hours": 12}).status_code == 409
    assert rc.get("/api/sim/score").status_code == 409
