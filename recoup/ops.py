"""recoup-ops -- the batch jobs behind deploy/systemd.

    recoup-ops synth    write a synthetic Bachs-shaped dataset
    recoup-ops retrain  fit the cure-hazard model; promote it if it passes its checks
    recoup-ops plan     decide the next retry of every open invoice; log propensities
    recoup-ops gate     cross-fitted OPE of the planner vs the incumbent; HOLD / SWITCH
    recoup-ops advance  (synthetic only) move the world forward, executing the queue
    recoup-ops demo     (synthetic only) the whole rollout, end to end
    recoup-ops score    (synthetic only) what the live policy earned vs the ladder
    recoup-ops serve    the operator console: API + web UI

Paths default from the environment the systemd units set, so the unit files
stay short:

    RECOUP_DATA   dataset directory      (default ./data/synthetic)
    RECOUP_STATE  models/, gate/, queue/ (default ./var/state)
    RECOUP_LOG    decisions/ audit log   (default ./var/log)

The data source is a *directory export* (layout in ``recoup.synthetic``).
The piece that writes that directory from the live Bachs API is the one part
not built: it needs ``bachs.FIELD_MAP`` pinned against the OpenAPI spec first,
and a fetcher written against a guessed schema is worse than none.

Exit codes: 0 ok, 1 error, 2 refused (a guard fired; nothing was changed).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
import warnings
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .domain import (DEFAULT_EPOCH, HARD_DECLINES, MAX_HORIZON_H, N_DT_BUCKETS,
                     NO_RETRY_ADVICE, DeclineReason, NetworkAdvice, NetworkRules, Rail,
                     dt_bucket)
from .evaluate import (cross_fitted_plan, deployment_gate, ladder_schedules, off_policy_value,
                       oracle_schedule_value, realised_reward, score_model)
from .features import _calendar, build_features, customer_history, derive_cure_labels
from .models import CURE_LABEL, CureHazardModel
from .policy import (PolicyConfig, RetryPlanner, RetryPolicy, SupportMap, TableDisputeModel,
                     _in_quiet_hours, fixed_ladder_policy)
from .simulator import SimConfig
from .store import (Dataset, ModelBundle, atomic_write_json, load_bundle, load_dataset,
                    read_audit, save_bundle, write_audit, write_queue)
from .synthetic import TRUTH_DIR, SynthConfig, advance_world, generate_dataset, read_manifest

log = logging.getLogger("recoup")

EXIT_OK, EXIT_ERROR, EXIT_REFUSED = 0, 1, 2
DEFAULT_LADDER = (24.0, 72.0, 168.0)


class Refused(Exception):
    """A guard fired. The job changed nothing and says why."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _env(var: str, default: str) -> Path:
    return Path(os.environ.get(var) or default)


def _data_default() -> Path:
    return _env("RECOUP_DATA", "data/synthetic")


def _state() -> Path:
    return _env("RECOUP_STATE", "var/state")


def _logdir() -> Path:
    return _env("RECOUP_LOG", "var/log")


def _epoch(ds: Dataset) -> datetime:
    iso = ds.manifest.get("epoch")
    return pd.Timestamp(iso).tz_localize(None).to_pydatetime() if iso else DEFAULT_EPOCH


def _iso(ds: Dataset, t_h: float) -> str:
    return (_epoch(ds) + timedelta(hours=float(t_h))).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]


def _parse_as_of(value: str | None, data: Path) -> float | None:
    """--as-of accepts hours since the dataset epoch or an ISO timestamp."""
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        pass
    epoch = read_manifest(data).get("epoch", DEFAULT_EPOCH.strftime("%Y-%m-%dT%H:%M:%SZ"))
    t = pd.Timestamp(value)
    t = t.tz_convert(None) if t.tzinfo else t
    return float((t - pd.Timestamp(epoch).tz_localize(None)) / pd.Timedelta(hours=1))


def _ladder(text: str) -> tuple[float, ...]:
    return tuple(float(x) for x in text.split(",") if x.strip())


def _blocked(reason: str, advice: str) -> bool:
    return DeclineReason(reason) in HARD_DECLINES or NetworkAdvice(advice) in NO_RETRY_ADVICE


def training_table(ds: Dataset, invoices: pd.DataFrame, closed_by_h: float,
                   since_h: float | None = None):
    """Invoices whose dunning window had closed by ``closed_by_h``, their
    attempts, and the feature table with labels. Open windows are left out:
    their outcomes are still arriving."""
    closed = invoices[invoices["fail_time_h"] + MAX_HORIZON_H <= closed_by_h]
    if since_h is not None:
        closed = closed[closed["fail_time_h"] >= since_h]
    closed = closed.reset_index(drop=True)
    att = ds.attempts[ds.attempts["invoice_id"].isin(set(closed["invoice_id"]))]
    att = att[att["attempt_time_h"] <= closed_by_h].reset_index(drop=True)
    feats = build_features(closed, att[["invoice_id", "attempt_index", "delay_hours"]])
    feats["success"] = att["success"].to_numpy()
    return closed, att, feats


def _fit_quietly(df: pd.DataFrame) -> tuple[CureHazardModel, list[str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = CureHazardModel().fit(df)
    msgs = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    for msg in msgs:
        log.warning("model: %s", msg)
    return m, msgs


def read_gate(path: Path | None) -> dict:
    if path is None or not Path(path).exists():
        return {"decision": "HOLD", "reason": "no gate decision on file"}
    try:
        g = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        log.warning("gate file %s unreadable (%s); treating as HOLD", path, e)
        return {"decision": "HOLD", "reason": f"unreadable gate file: {e}"}
    if g.get("decision") not in ("HOLD", "SWITCH"):
        return {"decision": "HOLD", "reason": f"unknown decision {g.get('decision')!r}"}
    return g


# ---------------------------------------------------------------------------
# synth / advance
# ---------------------------------------------------------------------------


def cmd_synth(a) -> int:
    out = Path(a.out)
    if (out / "manifest.json").exists() and not a.force:
        raise Refused(f"{out} already holds a dataset; pass --force to overwrite it")
    if a.force and out.exists():
        import shutil
        shutil.rmtree(out)
    cfg = SynthConfig(
        sim=SimConfig(n_invoices=a.invoices, days=a.days, n_merchants=a.merchants,
                      seed=a.seed),
        history_days=a.history_days, legacy_propensities=not a.no_legacy_log, seed=a.seed)
    t0 = time.time()
    m = generate_dataset(out, cfg)
    log.info("wrote %s in %.1fs: %s", out, time.time() - t0, json.dumps(m["counts"]))
    log.info("history ends %s; %d future invoices wait in %s/pending.jsonl",
             m["as_of"], m["counts"]["invoices_future"], TRUTH_DIR)
    return EXIT_OK


def cmd_advance(a) -> int:
    s = advance_world(a.data, a.queue, a.hours)
    log.info("advanced %.1fh -> %s: %s", a.hours, s.pop("as_of"), json.dumps(s))
    return EXIT_OK


# ---------------------------------------------------------------------------
# retrain
# ---------------------------------------------------------------------------


def cmd_retrain(a) -> int:
    t0 = time.time()
    ds = load_dataset(a.data, _parse_as_of(a.as_of, Path(a.data)))
    log.info("data as of %s: %s", _iso(ds, ds.as_of_h), str(ds.tables.report).replace("\n", ";"))
    inv = customer_history(ds.invoices, ds.attempts)
    since = ds.as_of_h - a.train_days * 24.0 if a.train_days else None
    closed, att, feats = training_table(ds, inv, ds.as_of_h, since)
    n_inv = feats["invoice_id"].nunique()
    if n_inv < a.min_invoices:
        raise Refused(f"only {n_inv} retried invoices with closed dunning windows "
                      f"(need {a.min_invoices}); keeping the current model")

    # Semi-supervised cure labels: opt-in. Measured on the default synthetic
    # dataset (true churn share among retried invoices 0.090): unlabelled fit
    # P(gone) 0.187; with both label sides 0.008; gone-only 0.241. Renewals
    # make "present" labels plentiful (5,266 vs 25 gone), and because present
    # labelling is outcome-dependent the likelihood's missing selection term
    # drags pi to zero -- the planner would then retry zombies forever. The
    # fix is a PU-style (1 - c(x)) factor in CureHazardModel, not a threshold
    # here. Until then the unlabelled fit (biased high, i.e. gives up a little
    # early) is the safer default.
    labels = dict(used=False, n_gone=0, n_present=0)
    if a.cure_labels:
        lab = derive_cure_labels(closed, ds.attempts, ds.tables.gone_events,
                                 successes=ds.tables.successes,
                                 gone_within_h=a.gone_within_days * 24.0)
        n_g, n_p = int((lab == 1.0).sum()), int((lab == 0.0).sum())
        labels.update(n_gone=n_g, n_present=n_p)
        if min(n_g, n_p) >= a.min_labels:
            feats[CURE_LABEL] = feats["invoice_id"].map(
                dict(zip(closed["invoice_id"], lab.to_numpy())))
            labels["used"] = True
        else:
            labels["reason"] = (f"need >= {a.min_labels} of each side; "
                                f"have {n_g} gone, {n_p} present")
    log.info("training on %d invoices / %d attempts; cure labels %s", n_inv, len(feats),
             json.dumps(labels))

    holdout = None
    if a.holdout_frac > 0:
        cut = closed["fail_time_h"].quantile(1.0 - a.holdout_frac)
        early = set(closed.loc[closed["fail_time_h"] <= cut, "invoice_id"])
        tr, te = feats[feats["invoice_id"].isin(early)], feats[~feats["invoice_id"].isin(early)]
        m0, _ = _fit_quietly(tr)
        rep = score_model(te["success"].to_numpy(), m0.predict_proba1(te.drop(
            columns=[CURE_LABEL], errors="ignore")))
        holdout = dict(asdict(rep), n_test_attempts=int(len(te)), converged=m0.converged_)
        log.info("temporal holdout (last %.0f%%): %s", 100 * a.holdout_frac, rep)
        if not a.force and (not m0.converged_ or rep.pr_auc < a.min_pr_auc_lift * rep.base_rate):
            raise Refused(f"holdout check failed (converged={m0.converged_}, PR-AUC "
                          f"{rep.pr_auc:.3f} vs base rate {rep.base_rate:.3f}); "
                          "keeping the current model")

    model, fit_warnings = _fit_quietly(feats)
    if not model.converged_ and not a.force:
        raise Refused("final fit did not converge; keeping the current model")
    cfg, rules = PolicyConfig(), NetworkRules()
    dispute = TableDisputeModel.fit(att, closed)
    support = SupportMap(feats, min_support=cfg.min_support)
    one = feats.drop_duplicates("invoice_id")
    pi_one, _ = model.predict_components(one)

    meta = dict(
        trained_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        data=str(Path(a.data).resolve()), data_as_of=_iso(ds, ds.as_of_h),
        data_as_of_h=ds.as_of_h, n_invoices=int(n_inv), n_attempts=int(len(feats)),
        n_merchants=len(model.merchants_), success_rate=float(feats["success"].mean()),
        mean_p_gone=float(pi_one.mean()), cure_labels=labels, holdout=holdout,
        converged=model.converged_, fit_warnings=fit_warnings,
        dispute_base={r.value: v for r, v in dispute.base.items()},
        ladder_h=list(_ladder(a.ladder)),
        unmapped_codes=ds.tables.report.unmapped_codes, fit_seconds=round(time.time() - t0, 1),
    )
    bundle = ModelBundle(model, dispute, support, cfg, rules, _ladder(a.ladder), meta)
    path = save_bundle(bundle, a.out, keep=a.keep)
    log.info("promoted %s (P(gone) %.3f, %d merchants) in %.0fs", path, meta["mean_p_gone"],
             meta["n_merchants"], time.time() - t0)
    return EXIT_OK


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------


def _incumbent_delay(ladder, k: int, cap: int, earliest: float, fail_h: float, market: str,
                     cfg: PolicyConfig) -> float | None:
    """The ladder's next rung, moved out of the past and out of quiet hours."""
    if k >= len(ladder) or k >= cap:
        return None
    delay = max(float(ladder[k]), earliest)
    for _ in range(24):
        hour = _calendar(np.array([fail_h + delay]), pd.Series([market]))["local_hour"].iloc[0]
        if not _in_quiet_hours(np.array([hour]), cfg.quiet_hours_local)[0]:
            break
        delay += 1.0
    return delay if delay <= MAX_HORIZON_H else None


def _choose(mode: str, d, row, k: int, earliest: float, bundle: ModelBundle,
            cfg: PolicyConfig, explore_rate: float, rng: np.random.Generator) -> dict:
    """Turn the planner's Decision into the action actually taken, and the
    exact probability it was taken with.

    SWITCH  act on the planner's Thompson draw.
    HOLD    the incumbent ladder, except with probability ``explore_rate``
            act on the planner's Thompson distribution instead. The logged
            propensity is the mixture's -- (1-eps)*ladder + eps*thompson --
            which is what lets the gate evaluate the planner later without
            having bet more than eps of retries on it.
    """
    onehot = lambda b: [1.0 if j == b else 0.0 for j in range(N_DT_BUCKETS)]
    thompson = (d.curve["propensity"].to_numpy(dtype=float)
                if d.action == "retry" and d.curve is not None else None)
    base = dict(p_gone=d.p_gone, p_success=d.p_success, expected_value_usd=d.expected_value_usd)

    if _blocked(row["decline_reason"], row["network_advice"]):
        return dict(base, action="route_to_update_method", delay=None, bucket=None,
                    propensity=1.0, vector=None, policy="constraint", rationale=d.rationale)
    if mode == "SWITCH":
        return dict(base, action=d.action, delay=d.delay_hours, bucket=d.bucket,
                    propensity=d.propensity if d.action == "retry" else 1.0,
                    vector=list(thompson) if thompson is not None else None,
                    policy="planner", rationale=d.rationale)

    cap = min(cfg.max_attempts, bundle.rules.attempt_cap(Rail(row["rail"])))
    inc = _incumbent_delay(bundle.ladder_h, k, cap, earliest, float(row["fail_time_h"]),
                           str(row["market"]), cfg)
    if inc is None:
        return dict(base, action="stop", delay=None, bucket=None, propensity=1.0,
                    vector=None, policy="ladder", rationale="incumbent ladder exhausted")
    lb = int(dt_bucket(inc))
    if thompson is None or explore_rate <= 0:
        return dict(base, action="retry", delay=inc, bucket=lb, propensity=1.0,
                    vector=onehot(lb), policy="ladder",
                    rationale="incumbent ladder" + ("" if thompson is not None else
                                                    f" (planner: {d.rationale})"))
    mix = (1.0 - explore_rate) * np.array(onehot(lb)) + explore_rate * thompson
    if rng.random() < explore_rate:
        b = int(rng.choice(N_DT_BUCKETS, p=thompson / thompson.sum()))
        delay = float(d.curve["delay_h"].iloc[b])
        policy, why = "explore", "exploration: planner's Thompson draw"
    else:
        b, delay, policy, why = lb, inc, "ladder", "incumbent ladder"
    return dict(base, action="retry", delay=delay, bucket=b, propensity=float(mix[b]),
                vector=[float(x) for x in mix], policy=policy, rationale=why)


def cmd_plan(a) -> int:
    t0 = time.time()
    bundle = load_bundle(a.model)
    ds = load_dataset(a.data, _parse_as_of(a.as_of, Path(a.data)))
    now = ds.as_of_h
    gate = read_gate(Path(a.gate) if a.gate else None)
    mode = gate["decision"]
    audit = read_audit(a.audit, ds)
    decided = set(zip(audit["invoice_id"], audit["attempt_index"]))

    inv = customer_history(ds.invoices, ds.attempts)
    elapsed = now - inv["fail_time_h"]
    open_inv = inv[(elapsed >= 0) & (elapsed < MAX_HORIZON_H)].reset_index(drop=True)
    att = ds.attempts[ds.attempts["invoice_id"].isin(set(open_inv["invoice_id"]))]
    past = {i: tuple(g.sort_values("attempt_index")["delay_hours"].astype(float))
            for i, g in att.groupby("invoice_id")}
    recovered = set(att.loc[att["success"] == 1, "invoice_id"])

    todo = []
    for r in open_inv.itertuples(index=False):
        if r.invoice_id in recovered:
            continue
        k = len(past.get(r.invoice_id, ()))
        if (r.invoice_id, k) not in decided:
            todo.append((r.invoice_id, k))
    if a.limit:
        todo = todo[: a.limit]
    log.info("as of %s: %d open invoices, %d need a decision; gate %s (%s); model %s",
             _iso(ds, now), len(open_inv), len(todo), mode, gate.get("reason", ""),
             bundle.version)

    cfg = bundle.policy_cfg
    if a.propensity_samples:
        cfg = replace(cfg, n_propensity_samples=a.propensity_samples)
    rng = np.random.default_rng(a.seed)
    policy = RetryPolicy(bundle.model, open_inv, bundle.support, cfg, bundle.rules,
                         bundle.dispute, rng)
    rows_by_id = open_inv.set_index("invoice_id", drop=False)
    run_id = _run_id()
    audit_rows, queue = [], []
    counts: dict[str, int] = {}
    for iid, k in todo:
        row = rows_by_id.loc[iid]
        el = now - float(row["fail_time_h"])
        pd_ = past.get(iid, ())
        d = policy.decide(iid, k, past_delays=pd_, now_elapsed_h=el)
        earliest = max(el + 0.25, (pd_[-1] + cfg.min_spacing_h) if pd_ else 0.0)
        c = _choose(mode, d, row, k, earliest, bundle, cfg, a.explore_rate, rng)
        counts[f"{c['action']}/{c['policy']}"] = counts.get(f"{c['action']}/{c['policy']}", 0) + 1
        did = f"{run_id}-{len(audit_rows):05d}"
        exec_h = float(row["fail_time_h"]) + c["delay"] if c["delay"] is not None else now
        audit_rows.append(dict(
            decision_id=did, run_id=run_id, decided_at_h=now, decided_at=_iso(ds, now),
            invoice_id=iid, attempt_index=k, action=c["action"], delay_hours=c["delay"],
            execute_at_h=exec_h, dt_bucket=c["bucket"], propensity=c["propensity"],
            propensities=json.dumps([round(x, 6) for x in c["vector"]]) if c["vector"] else "",
            mode=mode, policy=c["policy"], model_version=bundle.version,
            p_gone=round(float(c["p_gone"]), 5), p_success=round(float(c["p_success"]), 5),
            expected_value_usd=round(float(c["expected_value_usd"]), 4),
            rationale=c["rationale"]))
        if c["action"] in ("retry", "route_to_update_method"):
            queue.append(dict(decision_id=did, invoice_id=iid, attempt_index=k,
                              action=c["action"], execute_at_h=exec_h,
                              execute_at=_iso(ds, exec_h), delay_hours=c["delay"]))

    if a.dry_run:
        log.info("dry run: %d decisions not written", len(audit_rows))
    else:
        p = write_audit(a.audit, run_id, audit_rows)
        q = write_queue(a.queue, run_id, queue)
        log.info("wrote %s and %s", p, q)
    log.info("decisions %s in %.1fs", json.dumps(dict(sorted(counts.items()))), time.time() - t0)
    return EXIT_OK


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------


def logged_from_audit(ds: Dataset, audit: pd.DataFrame, start_h: float, cfg: PolicyConfig,
                      invoices: pd.DataFrame):
    """First-retry decisions in [start_h, as_of] joined to observed outcomes.

    Returns (inv_eval, logged, stats) in matching row order. The action is the
    *decided* bucket; the reward is what the executed attempt earned.
    """
    first = audit[(audit["attempt_index"] == 0) & (audit["action"] == "retry")
                  & (audit["decided_at_h"] >= start_h) & (audit["decided_at_h"] <= ds.as_of_h)]
    first = first.drop_duplicates("invoice_id", keep="first")
    att0 = ds.attempts[ds.attempts["attempt_index"] == 0].drop_duplicates("invoice_id")
    att0 = att0.set_index("invoice_id")
    stats = dict(decisions=int(len(first)))
    first = first[first["invoice_id"].isin(att0.index)]
    stats["with_outcome"] = int(len(first))
    first = first[pd.to_numeric(first["propensity"], errors="coerce").fillna(0) > 0]
    stats["with_propensity"] = int(len(first))
    inv_idx = invoices.set_index("invoice_id", drop=False)
    first = first[first["invoice_id"].isin(inv_idx.index)]
    inv_eval = inv_idx.loc[first["invoice_id"]].reset_index(drop=True)
    rewards = [realised_reward(att0.loc[i], float(amt), cfg, str(rail))
               for i, amt, rail in zip(inv_eval["invoice_id"], inv_eval["amount_usd"],
                                       inv_eval["rail"])]
    logged = pd.DataFrame({
        "invoice_id": inv_eval["invoice_id"].to_numpy(),
        "dt_bucket": first["dt_bucket"].astype(int).to_numpy(),
        "propensity": first["propensity"].astype(float).to_numpy(),
        "reward": np.asarray(rewards, dtype=float),
        "mode": first["mode"].astype(str).to_numpy(),
    })
    return inv_eval, logged, stats


def cmd_gate(a) -> int:
    t0 = time.time()
    bundle = load_bundle(a.model)
    ds = load_dataset(a.data, _parse_as_of(a.as_of, Path(a.data)))
    prev = read_gate(Path(a.out))
    start = ds.as_of_h - a.eval_days * 24.0
    cfg = replace(bundle.policy_cfg, n_posterior_samples=a.posterior_samples)
    inv = customer_history(ds.invoices, ds.attempts)
    audit = read_audit(a.audit, ds)
    inv_eval, logged, stats = logged_from_audit(ds, audit, start, cfg, inv)
    result = dict(generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                  data_as_of=_iso(ds, ds.as_of_h), window_start=_iso(ds, start),
                  model_version=bundle.version, previous=prev.get("decision"),
                  logged=stats, ladder_h=list(bundle.ladder_h))

    def finish(decision: str, reason: str, **extra) -> int:
        result.update(decision=decision, reason=reason, **extra)
        atomic_write_json(a.out, result)
        # One line per run, for the console's gate timeline.
        with open(Path(a.out).parent / "history.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({k: result.get(k) for k in (
                "generated_at", "data_as_of", "decision", "delta", "lo", "hi", "ess",
                "n_logged")}, default=float) + "\n")
        log.info("gate %s -- %s (%.0fs)", decision, reason, time.time() - t0)
        return EXIT_OK

    if len(logged) < a.min_decisions:
        return finish("HOLD" if prev.get("decision") != "SWITCH" else "SWITCH",
                      f"only {len(logged)} first-retry decisions with outcomes and "
                      f"propensities in the window (need {a.min_decisions}); no change")

    train_inv, _, train = training_table(ds, inv[~inv["invoice_id"].isin(
        set(inv_eval["invoice_id"]))], start)
    if train["invoice_id"].nunique() < a.min_train_invoices:
        return finish("HOLD" if prev.get("decision") != "SWITCH" else "SWITCH",
                      f"only {train['invoice_id'].nunique()} training invoices closed before "
                      "the evaluation window; no change")
    support = SupportMap(train, min_support=cfg.min_support)
    mk = lambda m: RetryPlanner(m, cfg, bundle.rules, bundle.dispute, support,
                                rng=np.random.default_rng(a.seed))
    fit = lambda d: _fit_quietly(d)[0]
    pi_c, _, q_hat = cross_fitted_plan(train, inv_eval, fit, mk, seed=a.seed)
    pi_inc, _ = fixed_ladder_policy(inv_eval, bundle.ladder_h)
    g = deployment_gate(logged, pi_c, pi_inc, q_hat, cfg, min_ess=a.min_ess,
                        n_boot=a.n_boot, seed=a.seed)
    ope_c = off_policy_value(logged, pi_c, q_hat, cfg)
    ope_i = off_policy_value(logged, pi_inc, q_hat, cfg)
    by_mode = logged.groupby("mode").size().to_dict()
    extra = dict(delta=g.delta, lo=g.lo, hi=g.hi, ess=g.ess, min_ess=a.min_ess,
                 dr_planner=ope_c.dr, dr_incumbent=ope_i.dr, ips_planner=ope_c.ips,
                 n_logged=int(len(logged)), logged_by_mode={k: int(v) for k, v in by_mode.items()},
                 n_train_invoices=int(train["invoice_id"].nunique()), gate=str(g))
    log.info("%s", g)

    if g.deploy:
        return finish("SWITCH", g.reason, **extra)
    # Hysteresis: once switched, only go back on evidence the planner is
    # *worse* -- not merely on a week where the interval touches zero. The
    # planner's own logs have little mass on ladder delays, so "cannot prove
    # it is better than the ladder" is the normal state after a switch.
    if prev.get("decision") == "SWITCH" and not a.no_sticky and g.hi >= 0:
        return finish("SWITCH", f"kept: no evidence the planner is worse ({g.reason})", **extra)
    return finish("HOLD", g.reason, **extra)


# ---------------------------------------------------------------------------
# score (synthetic only)
# ---------------------------------------------------------------------------


def score_world(data: Path, audit_dir: Path | None, since_h: float) -> dict:
    """What the live system actually earned per failed invoice, against the
    oracle expectation of the ladder on the same invoices. Reads _truth."""
    ds = load_dataset(data)
    truth = pd.read_csv(data / TRUTH_DIR / "invoices.csv").set_index("invoice_id")
    cfg = PolicyConfig()
    until = ds.as_of_h - MAX_HORIZON_H
    inv = ds.invoices[(ds.invoices["fail_time_h"] >= since_h)
                      & (ds.invoices["fail_time_h"] <= until)].reset_index(drop=True)
    if inv.empty:
        return dict(n_invoices=0, note="no invoice has a closed dunning window yet")
    att = ds.attempts[ds.attempts["invoice_id"].isin(set(inv["invoice_id"]))]
    amt = inv.set_index("invoice_id")["amount_usd"]
    rail = inv.set_index("invoice_id")["rail"]
    realised = att.apply(lambda r: realised_reward(r, float(amt[r["invoice_id"]]), cfg,
                                                   str(rail[r["invoice_id"]])), axis=1)
    per_inv = realised.groupby(att["invoice_id"]).sum().reindex(inv["invoice_id"]).fillna(0.0)

    oracle_inv = inv.copy()
    oracle_inv["_quality"] = truth.loc[inv["invoice_id"], "quality"].to_numpy()
    oracle_inv["_churned"] = truth.loc[inv["invoice_id"], "churned"].to_numpy()
    from .synthetic import _truth_from_manifest
    world = _truth_from_manifest(ds.manifest)
    ladder = tuple(ds.manifest["sim"]["ladder_h"])
    v_ladder = oracle_schedule_value(oracle_inv, ladder_schedules(oracle_inv, ladder, 3), cfg,
                                     world)
    out = dict(n_invoices=int(len(inv)), window=[_iso(ds, since_h), _iso(ds, until)],
               realised_per_invoice=float(per_inv.mean()),
               realised_se=float(per_inv.std(ddof=1) / np.sqrt(len(per_inv))),
               ladder_oracle_per_invoice=float(v_ladder),
               recovery_rate=float(att.groupby("invoice_id")["success"].max().reindex(
                   inv["invoice_id"]).fillna(0).mean()),
               attempts_per_invoice=float(len(att) / len(inv)))
    out["lift_vs_ladder"] = out["realised_per_invoice"] / v_ladder - 1 if v_ladder else None

    audit = read_audit(audit_dir, ds)
    first = audit[audit["attempt_index"] == 0].drop_duplicates("invoice_id")
    mode = first.set_index("invoice_id")["mode"].reindex(inv["invoice_id"]).fillna("none")
    phases = {}
    for m in sorted(mode.unique()):
        sel = (mode == m).to_numpy()
        if sel.sum() < 30:
            continue
        v_l = oracle_schedule_value(oracle_inv[sel], ladder_schedules(oracle_inv[sel], ladder, 3),
                                    cfg, world)
        pv = per_inv.to_numpy()[sel]
        phases[m] = dict(n=int(sel.sum()), realised=float(pv.mean()),
                         se=float(pv.std(ddof=1) / np.sqrt(sel.sum())), ladder_oracle=float(v_l),
                         lift=float(pv.mean() / v_l - 1) if v_l else None)
    out["by_mode"] = phases
    return out


def cmd_score(a) -> int:
    data = Path(a.data)
    since = (_parse_as_of(a.since, data) if a.since is not None
             else float(read_manifest(data).get("synth", {}).get("history_days", 0)) * 24.0)
    s = score_world(data, Path(a.audit) if a.audit else None, since)
    print(json.dumps(s, indent=1, default=float))
    return EXIT_OK


# ---------------------------------------------------------------------------
# demo (synthetic only)
# ---------------------------------------------------------------------------


def cmd_demo(a) -> int:
    """Synthesize -> retrain -> [plan -> advance]* with periodic retrain and
    gate -> score. Every step goes through the same CLI entry points the
    systemd units call."""
    work = Path(a.workdir)
    data, state, logs = work / "data", work / "state", work / "log"
    models, gate, queue, audit = (state / "models", state / "gate" / "decision.json",
                                  state / "queue", logs / "decisions")
    if a.fresh and work.exists():
        import shutil
        shutil.rmtree(work)
    common = ["--data", str(data)]
    if not (data / "manifest.json").exists():
        rc = main(["synth", "--out", str(data), "--invoices", str(a.invoices),
                   "--days", str(a.days_total), "--history-days", str(a.history_days),
                   "--seed", str(a.seed)] + (["--no-legacy-log"] if a.no_legacy_log else []))
        if rc:
            return rc
    start_h = float(read_manifest(data)["as_of_h"])

    def retrain():
        return main(["retrain", *common, "--out", str(models)])

    def run_gate():
        return main(["gate", *common, "--model", str(models / "current"), "--audit",
                     str(audit), "--out", str(gate), "--eval-days", str(a.eval_days)])

    if retrain() not in (EXIT_OK, EXIT_REFUSED):
        return EXIT_ERROR
    run_gate()
    steps = int(round(a.days * 24.0 / a.step_hours))
    t0 = time.time()
    for s in range(steps):
        elapsed_d = s * a.step_hours / 24.0
        if s and a.retrain_every_days and elapsed_d % a.retrain_every_days < a.step_hours / 24.0:
            retrain()
        if s and a.gate_every_days and elapsed_d % a.gate_every_days < a.step_hours / 24.0:
            run_gate()
        rc = main(["plan", *common, "--model", str(models / "current"), "--gate", str(gate),
                   "--audit", str(audit), "--queue", str(queue), "--explore-rate",
                   str(a.explore_rate), "--propensity-samples", str(a.propensity_samples),
                   "--seed", str(a.seed + s)])
        if rc:
            return rc
        main(["advance", *common, "--queue", str(queue), "--hours", str(a.step_hours)])
        log.info("demo: day %.1f / %d done (%.0fs)", elapsed_d + a.step_hours / 24.0, a.days,
                 time.time() - t0)
    run_gate()
    s = score_world(data, audit, start_h)
    s["gate"] = read_gate(gate)
    atomic_write_json(work / "demo_report.json", s)
    print(json.dumps(s, indent=1, default=float))
    return EXIT_OK


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_serve(a) -> int:
    try:
        import uvicorn
        from .api.app import create_app
        from .api.service import Paths
    except ImportError as e:
        raise Refused(f"the console needs the api extra: pip install -e .[api] ({e})")
    paths = Paths.from_env(a.workdir)
    app = create_app(paths)
    if not any(getattr(r, "path", "") == "/{path:path}" for r in app.routes):
        log.warning("no built UI found (set RECOUP_WEB or run `npm run build` in web/); "
                    "serving the API only -- docs at /docs")
    log.info("console on http://%s:%d/  (data %s)", a.host, a.port, paths.data)
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="recoup-ops", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    def data_arg(sp):
        sp.add_argument("--data", default=str(_data_default()),
                        help="dataset directory (env RECOUP_DATA)")
        sp.add_argument("--as-of", help="hours since epoch or ISO time (default: dataset clock)")

    s = sub.add_parser("synth", help="write a synthetic Bachs-shaped dataset")
    s.add_argument("--out", default=str(_data_default()))
    s.add_argument("--invoices", type=int, default=9000)
    s.add_argument("--days", type=int, default=240, help="length of the simulated world")
    s.add_argument("--history-days", type=float, default=180.0,
                   help="released as history; the rest is the future")
    s.add_argument("--merchants", type=int, default=12)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--no-legacy-log", action="store_true",
                   help="omit the incumbent's propensity log, like real Bachs history")
    s.add_argument("--force", action="store_true", help="overwrite an existing dataset")
    s.set_defaults(fn=cmd_synth)

    s = sub.add_parser("retrain", help="fit and promote a model bundle")
    data_arg(s)
    s.add_argument("--out", default=str(_state() / "models"), help="models directory")
    s.add_argument("--keep", type=int, default=5, help="bundles to retain")
    s.add_argument("--ladder", default=",".join(str(int(x)) for x in DEFAULT_LADDER),
                   help="incumbent ladder, hours")
    s.add_argument("--train-days", type=float, default=None, help="only the most recent N days")
    s.add_argument("--min-invoices", type=int, default=300)
    s.add_argument("--holdout-frac", type=float, default=0.15)
    s.add_argument("--min-pr-auc-lift", type=float, default=1.05,
                   help="holdout PR-AUC must beat base rate by this factor")
    s.add_argument("--cure-labels", action="store_true",
                   help="use semi-supervised cure labels (biases P(gone) low; see cmd_retrain)")
    s.add_argument("--gone-within-days", type=float, default=7.0,
                   help="a cancellation labels an invoice gone only this soon after it failed")
    s.add_argument("--min-labels", type=int, default=10)
    s.add_argument("--force", action="store_true", help="promote even if checks fail")
    s.set_defaults(fn=cmd_retrain)

    s = sub.add_parser("plan", help="decide next retries, write audit log + queue")
    data_arg(s)
    s.add_argument("--model", default=str(_state() / "models" / "current"))
    s.add_argument("--gate", default=str(_state() / "gate" / "decision.json"))
    s.add_argument("--audit", default=str(_logdir() / "decisions"))
    s.add_argument("--queue", default=str(_state() / "queue"))
    s.add_argument("--explore-rate", type=float, default=0.2,
                   help="under HOLD, share of retries that follow the planner")
    s.add_argument("--propensity-samples", type=int, default=None,
                   help="posterior draws per decision (default: bundle's, 128)")
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=cmd_plan)

    s = sub.add_parser("gate", help="off-policy evaluation; write HOLD/SWITCH")
    data_arg(s)
    s.add_argument("--model", default=str(_state() / "models" / "current"))
    s.add_argument("--audit", default=str(_logdir() / "decisions"))
    s.add_argument("--out", default=str(_state() / "gate" / "decision.json"))
    s.add_argument("--eval-days", type=float, default=45.0)
    s.add_argument("--min-decisions", type=int, default=200)
    s.add_argument("--min-train-invoices", type=int, default=300)
    s.add_argument("--min-ess", type=float, default=100.0)
    s.add_argument("--n-boot", type=int, default=400)
    s.add_argument("--posterior-samples", type=int, default=16)
    s.add_argument("--no-sticky", action="store_true",
                   help="revert SWITCH whenever the interval includes zero")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_gate)

    s = sub.add_parser("advance", help="(synthetic) release events, execute queued retries")
    s.add_argument("--data", default=str(_data_default()))
    s.add_argument("--queue", default=str(_state() / "queue"))
    s.add_argument("--hours", type=float, default=24.0)
    s.set_defaults(fn=cmd_advance)

    s = sub.add_parser("score", help="(synthetic) realised value vs the ladder's oracle value")
    s.add_argument("--data", default=str(_data_default()))
    s.add_argument("--audit", default=str(_logdir() / "decisions"))
    s.add_argument("--since", default=None, help="default: the dataset's cutover")
    s.set_defaults(fn=cmd_score)

    s = sub.add_parser("demo", help="(synthetic) the whole rollout loop, end to end")
    s.add_argument("--workdir", default="var/demo")
    s.add_argument("--fresh", action="store_true")
    s.add_argument("--invoices", type=int, default=9000)
    s.add_argument("--days-total", type=int, default=240)
    s.add_argument("--history-days", type=float, default=180.0)
    s.add_argument("--days", type=int, default=45, help="days to run the live loop")
    s.add_argument("--step-hours", type=float, default=12.0)
    s.add_argument("--retrain-every-days", type=float, default=7.0)
    s.add_argument("--gate-every-days", type=float, default=7.0)
    s.add_argument("--eval-days", type=float, default=45.0)
    s.add_argument("--explore-rate", type=float, default=0.2)
    s.add_argument("--propensity-samples", type=int, default=32)
    s.add_argument("--no-legacy-log", action="store_true")
    s.add_argument("--seed", type=int, default=7)
    s.set_defaults(fn=cmd_demo)

    s = sub.add_parser("serve", help="the operator console (API + web UI)")
    s.add_argument("--workdir", default=None,
                   help="use <workdir>/{data,state,log} instead of the RECOUP_* paths")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8080)
    s.set_defaults(fn=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO, stream=sys.stderr,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        return args.fn(args)
    except Refused as e:
        log.error("refused: %s", e)
        return EXIT_REFUSED
    except FileNotFoundError as e:
        log.error("%s", e)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
