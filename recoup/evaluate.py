"""Evaluation.

Three things here matter more than the metrics themselves.

**Temporal splits only.** A random split lets the same customer's later attempts
inform the model about their earlier ones, and payday effects leak across the
boundary. The number you get is beautiful and meaningless.

**Off-policy estimation, cross-fitted, checked against truth.** Logged data was
collected under the incumbent ladder, so a naive "average predicted success
rate under the new policy" is circular. We use doubly-robust estimation with a
reward model fitted on a *different* fold from the one that produced the
policy (``cross_fitted_plan``) -- otherwise the direct term is the policy's
optimistic opinion of itself. Because this runs on the simulator, the policy's
*true* value is also computed by oracle and the estimator is audited.

**The sequential objective.** A retry policy is a schedule, not a delay. The
planner is scored on the value of the whole schedule (``oracle_schedule_value``)
as well as on the first decision (``oracle_policy_value``), because the two can
disagree and the first is what a merchant's revenue actually sees.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from .domain import (DT_BUCKET_EDGES_H, HARD_DECLINES, N_DT_BUCKETS, NO_RETRY_ADVICE,
                     DeclineReason, NetworkAdvice, Rail)
from .policy import PolicyConfig


def bucket_midpoints() -> np.ndarray:
    e = np.array(DT_BUCKET_EDGES_H)
    return (e[:-1] + e[1:]) / 2.0


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def temporal_split(invoices: pd.DataFrame, attempts: pd.DataFrame, train_frac: float = 0.7):
    cutoff = invoices["fail_time_h"].quantile(train_frac)
    tr_ids = set(invoices.loc[invoices["fail_time_h"] <= cutoff, "invoice_id"])
    inv_tr = invoices[invoices["invoice_id"].isin(tr_ids)].copy()
    inv_te = invoices[~invoices["invoice_id"].isin(tr_ids)].copy()
    att_tr = attempts[attempts["invoice_id"].isin(tr_ids)].copy()
    att_te = attempts[~attempts["invoice_id"].isin(tr_ids)].copy()
    return inv_tr, att_tr, inv_te, att_te, float(cutoff)


# ---------------------------------------------------------------------------
# Model-level metrics
# ---------------------------------------------------------------------------


@dataclass
class FitReport:
    pr_auc: float
    roc_auc: float
    log_loss: float
    ece: float
    base_rate: float

    def __str__(self) -> str:
        return (f"PR-AUC {self.pr_auc:.4f} | ROC-AUC {self.roc_auc:.4f} | "
                f"logloss {self.log_loss:.4f} | ECE {self.ece:.4f} | "
                f"base rate {self.base_rate:.4f}")


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 12) -> float:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(p, edges[1:-1])
    err = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        err += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(err)


def score_model(y: np.ndarray, p: np.ndarray) -> FitReport:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return FitReport(
        pr_auc=float(average_precision_score(y, p)),
        roc_auc=float(roc_auc_score(y, p)),
        log_loss=float(log_loss(y, p)),
        ece=expected_calibration_error(y, p),
        base_rate=float(y.mean()),
    )


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------


def realised_reward(attempt: pd.Series, amount: float, cfg: PolicyConfig,
                    rail: str | None = None) -> float:
    disc = np.exp(-cfg.daily_discount * float(attempt["delay_hours"]) / 24.0)
    r = attempt["success"] * amount * disc - cfg.attempt_cost_usd - cfg.annoyance(rail or "")
    if attempt["disputed"]:
        r -= amount + cfg.dispute_fee_usd
    return float(r)


def logged_first_attempts(invoices: pd.DataFrame, attempts: pd.DataFrame,
                          cfg: PolicyConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align the first logged retry of each invoice with its invoice row and
    attach the realised reward. Returns (invoices, logged) in the same order."""
    first = attempts[attempts["attempt_index"] == 0].drop_duplicates("invoice_id")
    inv = invoices[invoices["invoice_id"].isin(first["invoice_id"])].reset_index(drop=True)
    first = first.set_index("invoice_id").loc[inv["invoice_id"]].reset_index()
    amt = inv.set_index("invoice_id")["amount_usd"]
    rail = inv.set_index("invoice_id")["rail"]
    first["reward"] = [realised_reward(r, float(amt.loc[r["invoice_id"]]), cfg,
                                      str(rail.loc[r["invoice_id"]]))
                       for _, r in first.iterrows()]
    return inv, first


# ---------------------------------------------------------------------------
# Oracle (simulation only)
# ---------------------------------------------------------------------------


def _oracle_p(inv: pd.Series, delay_h: float, attempt_index: int, truth) -> float:
    from .simulator import true_success_prob
    return true_success_prob(
        reason=DeclineReason(inv["decline_reason"]), rail=Rail(inv["rail"]),
        market_code=inv["market"], fail_time_h=float(inv["fail_time_h"]),
        attempt_time_h=float(inv["fail_time_h"]) + delay_h,
        attempt_index=attempt_index, customer_quality=float(inv["_quality"]),
        amount_usd=float(inv["amount_usd"]), churned=bool(inv["_churned"]),
        advice=NetworkAdvice(inv.get("network_advice", NetworkAdvice.NONE.value)),
        truth=truth,
    )


def oracle_policy_value(invoices: pd.DataFrame, propensities: np.ndarray,
                        delays: np.ndarray, cfg: PolicyConfig,
                        attempt_index: int = 0, truth=None) -> float:
    """True expected value per failed invoice of the *first* decision, using
    the simulator's oracle. Each policy is evaluated at the delays it actually
    chose, not at bucket midpoints. Only available in simulation."""
    from .simulator import DEFAULT_TRUTH, true_dispute_prob
    truth = truth or DEFAULT_TRUTH
    total = 0.0
    for (_, inv), pi, dl in zip(invoices.iterrows(), propensities, delays):
        if pi.sum() <= 0:
            continue
        amount = float(inv["amount_usd"])
        rail = Rail(inv["rail"])
        cost = cfg.attempt_cost_usd + cfg.annoyance(rail.value)
        v = 0.0
        for b in range(N_DT_BUCKETS):
            if pi[b] <= 0:
                continue
            h = float(dl[b])
            p = _oracle_p(inv, h, attempt_index, truth)
            pd_ = true_dispute_prob(attempt_index=attempt_index, elapsed_h=h, rail=rail)
            disc = np.exp(-cfg.daily_discount * h / 24.0)
            ev = p * amount * disc - cost - p * pd_ * (amount + cfg.dispute_fee_usd)
            v += pi[b] * ev
        total += v
    return total / max(len(invoices), 1)


def oracle_schedule_value(invoices: pd.DataFrame, schedules: list, cfg: PolicyConfig,
                          truth=None) -> float:
    """True expected value of a full retry *schedule* per invoice, in closed
    form under the oracle: sum over attempts of P(reach attempt k) * EV_k."""
    from .simulator import DEFAULT_TRUTH, true_dispute_prob
    truth = truth or DEFAULT_TRUTH
    total = 0.0
    for (_, inv), sched in zip(invoices.iterrows(), schedules):
        amount = float(inv["amount_usd"])
        rail = Rail(inv["rail"])
        cost = cfg.attempt_cost_usd + cfg.annoyance(rail.value)
        reach = 1.0
        for k, h in enumerate(sched):
            p = _oracle_p(inv, float(h), k, truth)
            pd_ = true_dispute_prob(attempt_index=k, elapsed_h=float(h), rail=rail)
            disc = np.exp(-cfg.daily_discount * float(h) / 24.0)
            total += reach * (p * amount * disc - cost - p * pd_ * (amount + cfg.dispute_fee_usd))
            reach *= (1.0 - p)
    return total / max(len(invoices), 1)


def ladder_schedules(invoices: pd.DataFrame, ladder_h: tuple[float, ...], max_attempts: int) -> list:
    """The incumbent's schedule, for ``oracle_schedule_value``."""
    from .policy import _blocked
    blocked = _blocked(invoices)
    sched = tuple(ladder_h[min(k, len(ladder_h) - 1)] for k in range(max_attempts))
    return [() if b else sched for b in blocked]


# ---------------------------------------------------------------------------
# Off-policy estimators
# ---------------------------------------------------------------------------


@dataclass
class OPEResult:
    ips: float
    snips: float
    dr: float
    oracle: float | None = None
    ess: float = 0.0

    def __str__(self) -> str:
        s = (f"IPS {self.ips:+.4f} | SNIPS {self.snips:+.4f} | "
             f"DR {self.dr:+.4f} | ESS {self.ess:.0f}")
        if self.oracle is not None:
            s += f" | oracle {self.oracle:+.4f} (DR error {self.dr - self.oracle:+.4f})"
        return s


def off_policy_value(logged: pd.DataFrame, target_propensities: np.ndarray,
                     q_hat: np.ndarray, cfg: PolicyConfig,
                     oracle: float | None = None) -> OPEResult:
    """IPS, self-normalised IPS and doubly-robust value per invoice.

    ``logged``              one row per evaluated logged attempt, with columns
                            dt_bucket, propensity, reward.
    ``target_propensities`` (n, N_DT_BUCKETS) action distribution of the new policy.
    ``q_hat``               (n, N_DT_BUCKETS) reward-model estimates -- fit them
                            on a different fold from the policy (see
                            ``cross_fitted_plan``).
    """
    a = logged["dt_bucket"].to_numpy().astype(int)
    p_log = np.clip(logged["propensity"].to_numpy(dtype=float), 1e-4, 1.0)
    r = logged["reward"].to_numpy(dtype=float)
    rows = np.arange(len(logged))

    pi_a = target_propensities[rows, a]
    w = np.clip(pi_a / p_log, 0.0, 25.0)  # clipped to tame variance

    ips = float(np.mean(w * r))
    snips = float(np.sum(w * r) / max(np.sum(w), 1e-9))
    direct = np.sum(target_propensities * q_hat, axis=1)
    dr = float(np.mean(direct + w * (r - q_hat[rows, a])))
    ess = float(w.sum() ** 2 / max(np.sum(w ** 2), 1e-9))
    return OPEResult(ips=ips, snips=snips, dr=dr, oracle=oracle, ess=ess)


# ---------------------------------------------------------------------------
# Cross-fitting
# ---------------------------------------------------------------------------


def cross_fitted_plan(train: pd.DataFrame, eval_invoices: pd.DataFrame,
                      fit_model, make_planner, n_folds: int = 2, seed: int = 0):
    """Policy and reward model from *disjoint* halves of the training data.

    The doubly-robust estimator's direct term uses a reward model q_hat. If
    q_hat is the same model whose argmax produced the policy, the direct term
    is the policy's optimistic opinion of itself (the optimiser's curse) and DR
    inherits that bias exactly where importance weights are thin.

    Breaking that dependence takes two things, and an earlier version of this
    function only did the first:

    1. Fit two models on disjoint halves of the training invoices.
    2. **Give each evaluated invoice a policy and a reward model from
       different halves.** Averaging both quantities over both models -- which
       is what this used to do -- makes every row's ``pi`` and ``q_hat``
       depend on both fits again, and hands the curse straight back. So the
       *evaluation* set is split too: half its rows are planned under model A
       and valued under model B, the other half the reverse, and the halves
       are concatenated back into the caller's row order.

    Only ``n_folds=2`` is meaningful here. With K>2 the usual
    leave-one-fold-out models share training folds pairwise, so no pairing of
    them is disjoint; K=2 is the only split that gives the guarantee this
    function exists to provide.

    ``fit_model(df) -> model``; ``make_planner(model) -> RetryPlanner``.
    Returns (pi, delays, q_hat) for ``eval_invoices`` in the order given.
    ``q_hat`` is the *first-attempt* expected reward per bucket, in the same
    units as the logged reward -- not the schedule value.

    Note that each model sees half the training data, so the estimate is of a
    policy fitted on n/2. That is mildly pessimistic versus the policy you
    would actually deploy, and pessimism is the safe direction for a gate.

    What the fix does and does not buy (measured, 6000-invoice fixture, DR
    against the first-decision oracle)::

        same model both sides   DR 10.60  oracle 7.65  error +2.95  ESS 76
        old, average both       DR 10.27  oracle 7.73  error +2.55  ESS 83
        this, disjoint pairing  DR 11.09  oracle 7.74  error +3.35  ESS 71

    The disjoint version's error is *larger*, and it is worth being clear why
    rather than quietly shipping the smaller number: averaging two models
    shrinks q_hat toward the middle, which happens to drag the optimistic
    direct term down. That is smoothing, not bias correction, and it came
    bundled with a concrete defect -- the averaged ``delays`` were the
    arithmetic mean of two models' chosen delays, so 45% of invoices were
    logged with a delay *neither* model would have chosen (model A says 3h,
    model B says 5h, the log says 4h). An action no policy would take cannot
    be the action whose value you are estimating.

    The honest reading of the table is that all three are 33-43% optimistic
    and the cross-fit structure is not what is driving that: ESS near 75 on
    1500 logged attempts means the importance-weighted correction term is too
    thin to pull the direct term back to earth, whatever produced it. That is
    the same overlap problem the deployment gate reports, and it is fixed by
    logging exploration data, not by rearranging estimators.

    Averaging over several independent 2-way splits would recover the variance
    reduction without reintroducing the dependence -- average the per-split DR
    *estimates*, never the per-split ``pi`` and ``q_hat`` -- and is the natural
    next step here.
    """
    if n_folds != 2:
        raise ValueError(
            "cross_fitted_plan requires n_folds=2: with more folds the "
            "leave-one-out models share training data pairwise, so no pair of "
            "them is disjoint and the cross-fit guarantee does not hold."
        )
    rng = np.random.default_rng(seed)
    inv_ids = train["invoice_id"].astype(str).unique()
    rng.shuffle(inv_ids)
    halves = np.array_split(inv_ids, 2)
    # Model j is fitted on half j only; the two parameter vectors share no data.
    models = [fit_model(train[train["invoice_id"].astype(str).isin(set(h))])
              for h in halves]

    n = len(eval_invoices)
    # Which model plans each evaluated invoice; the other one values it.
    group = rng.integers(0, 2, size=n)

    plans_pol = [make_planner(m).plan(eval_invoices) for m in models]
    plans_val = [make_planner(m).plan(eval_invoices, explore=False) for m in models]

    pi = np.zeros((n, N_DT_BUCKETS))
    dl = np.zeros((n, N_DT_BUCKETS))
    q = np.zeros((n, N_DT_BUCKETS))
    for g in (0, 1):
        rows = np.flatnonzero(group == g)
        if rows.size == 0:
            continue
        pi[rows] = plans_pol[g].pi[rows]
        dl[rows] = plans_pol[g].delays[rows]
        q[rows] = plans_val[1 - g].q_first[rows]   # the *other* half's model
    return pi, dl, q


# ---------------------------------------------------------------------------
# Deployment gate
# ---------------------------------------------------------------------------


@dataclass
class GateResult:
    deploy: bool
    delta: float
    lo: float
    hi: float
    ess: float
    reason: str

    def __str__(self) -> str:
        verdict = "DEPLOY" if self.deploy else "HOLD"
        return (f"{verdict}: delta {self.delta:+.4f} USD/invoice "
                f"[{self.lo:+.4f}, {self.hi:+.4f}] ESS {self.ess:.0f} -- {self.reason}")


def deployment_gate(logged: pd.DataFrame, candidate_pi: np.ndarray,
                    incumbent_pi: np.ndarray, q_hat: np.ndarray, cfg: PolicyConfig,
                    *, min_ess: float = 100.0, n_boot: int = 400,
                    alpha: float = 0.05, seed: int = 0) -> GateResult:
    """Should this policy replace the incumbent?

    Requires the paired DR difference to be positive across a bootstrap
    confidence interval, and the effective sample size to clear ``min_ess``.
    Ties and thin evidence resolve in favour of the incumbent, because the cost
    of a bad swap is paid in real revenue while the cost of waiting is another
    month of logging.
    """
    a = logged["dt_bucket"].to_numpy().astype(int)
    p_log = np.clip(logged["propensity"].to_numpy(dtype=float), 1e-4, 1.0)
    r = logged["reward"].to_numpy(dtype=float)
    rows = np.arange(len(logged))

    def dr_terms(pi: np.ndarray) -> np.ndarray:
        w = np.clip(pi[rows, a] / p_log, 0.0, 25.0)
        return np.sum(pi * q_hat, axis=1) + w * (r - q_hat[rows, a])

    diff = dr_terms(candidate_pi) - dr_terms(incumbent_pi)
    w_cand = np.clip(candidate_pi[rows, a] / p_log, 0.0, 25.0)
    ess = float(w_cand.sum() ** 2 / max(np.sum(w_cand ** 2), 1e-9))

    rng = np.random.default_rng(seed)
    boot = np.array([diff[rng.integers(0, len(diff), len(diff))].mean()
                     for _ in range(n_boot)])
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    delta = float(diff.mean())

    if ess < min_ess:
        return GateResult(False, delta, float(lo), float(hi), ess,
                          f"effective sample size {ess:.0f} below {min_ess:.0f}; "
                          "the candidate prefers delays the logs barely cover")
    if lo <= 0:
        return GateResult(False, delta, float(lo), float(hi), ess,
                          "confidence interval on the improvement includes zero")
    return GateResult(True, delta, float(lo), float(hi), ess,
                      "improvement is positive across the bootstrap interval")
