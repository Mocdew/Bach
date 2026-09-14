"""The decision layer.

The model produces two curves per invoice -- P(customer is gone) and
P(success | not gone, delay, attempt). It does *not* decide anything. The
policy is:

1. **Hard constraints first.** Hard decline codes and network advice that
   forbids retrying route straight to "ask for a new payment method". Network
   attempt caps, per-rail caps, quiet hours and minimum spacing are applied as
   action masks *before* any optimisation, so a learned model can never
   produce a rule violation.

2. **Plan the whole schedule, not the next retry.** With at most 3-4 attempts
   and a couple of dozen candidate delays, the finite-horizon problem is small
   enough to solve *exactly* by enumerating schedules. The value of a schedule
   accounts for the fact that a failed early attempt is evidence the customer
   is gone, and that later attempts only happen if earlier ones failed. A
   myopic argmax over the next delay is a special case (``horizon=1``) and is
   kept for ablation.

3. **Expected value, not probability.** Per attempt:

       EV = P(success)·amount·discount(t) − attempt_cost − annoyance(rail)
            − P(success)·P(dispute | t, k)·(amount + fee)

   ``annoyance`` is the brake on wallet rails, where a retry is a push prompt
   on the customer's phone and chargebacks do not exist. Without it nothing
   stops the optimiser from prompting forever.

4. **Thompson sampling for exploration.** Instead of ε-uniform over buckets
   (which spends a quarter of retries on delays everyone knows are bad), the
   policy re-plans under posterior draws of the model and acts on the draw.
   Exploration lands where the model is actually uncertain. Propensities are
   the Monte-Carlo frequency of each first-action bucket, floored so that
   every viable action stays visible to off-policy evaluation.

The dispute model is injected. ``TableDisputeModel`` is the production default
and can be fitted on the disputes endpoint; nothing here imports the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Literal, Protocol

import numpy as np
import pandas as pd

from .domain import (
    DT_BUCKET_EDGES_H,
    HARD_DECLINES,
    N_DT_BUCKETS,
    NO_RETRY_ADVICE,
    DeclineReason,
    NetworkAdvice,
    NetworkRules,
    Rail,
    dt_bucket,
)
from .features import _calendar, build_features


# ---------------------------------------------------------------------------
# Dispute model (injected)
# ---------------------------------------------------------------------------


class DisputeModel(Protocol):
    def __call__(self, attempt_index: int, elapsed_h: float, rail: Rail) -> float: ...


@dataclass
class TableDisputeModel:
    """P(chargeback | recovered) = base[rail] · (1 + a·k) · (1 + b·days), capped.

    ``base`` is the thing to fit from the disputes endpoint; the growth terms
    are priors that a small logistic can replace once there are enough
    disputes to fit one (there usually are not).
    """

    base: dict[Rail, float] = field(default_factory=lambda: {
        Rail.CARD: 0.0022, Rail.BANK_TRANSFER: 0.0022,
        Rail.MOBILE_MONEY: 0.0005, Rail.STABLECOIN: 0.0005,
    })
    per_attempt: float = 0.55
    per_day: float = 0.16
    cap: float = 0.09

    def __call__(self, attempt_index: int, elapsed_h: float, rail: Rail) -> float:
        b = self.base.get(rail, 0.002)
        if rail in (Rail.MOBILE_MONEY, Rail.STABLECOIN):
            return b * (1 + attempt_index)
        return float(min(b * (1 + self.per_attempt * attempt_index)
                         * (1 + self.per_day * elapsed_h / 24.0), self.cap))

    @classmethod
    def fit(cls, attempts: pd.DataFrame, invoices: pd.DataFrame,
            prior_strength: float = 200.0) -> "TableDisputeModel":
        """Per-rail base rate from logged disputes, shrunk toward the default."""
        m = cls()
        df = attempts.merge(invoices[["invoice_id", "rail"]], on="invoice_id")
        ok = df[df["success"] == 1]
        for r in Rail:
            g = ok[ok["rail"] == r.value]
            n, d = len(g), float(g["disputed"].sum()) if len(g) else 0.0
            prior = m.base[r]
            m.base[r] = (d + prior_strength * prior) / (n + prior_strength)
        return m


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyConfig:
    # --- economics --------------------------------------------------------
    attempt_cost_usd: float = 0.09     # gateway + network fee per attempt
    dispute_fee_usd: float = 15.0      # chargeback fee on top of lost principal
    daily_discount: float = 0.004      # cash now beats cash in two weeks
    # Per-attempt cost of bothering the customer on push-prompt rails. This
    # is a stand-in for churn risk; fit it from retention data when you can.
    annoyance_cost_usd: tuple[tuple[str, float], ...] = (
        (Rail.MOBILE_MONEY.value, 0.30), (Rail.STABLECOIN.value, 0.15))
    # --- constraints ------------------------------------------------------
    max_attempts: int = 4
    min_spacing_h: float = 2.0
    quiet_hours_local: tuple[int, int] = (23, 6)
    # --- extrapolation guard (defensive; the posterior is the real guard) --
    min_support: int = 10
    # --- planning ---------------------------------------------------------
    horizon: int = 4                   # attempts planned jointly; 1 = myopic
    candidates_per_bucket: int = 2
    # --- exploration ------------------------------------------------------
    n_posterior_samples: int = 16
    # Scale on posterior deviations. 1.0 = full Thompson sampling; smaller
    # explores less. The Laplace posterior is conservative (wide), so 0.5 lands
    # near a 15-20% exploration rate on the benchmark.
    posterior_temperature: float = 0.35
    propensity_floor: float = 0.02

    def annoyance(self, rail: str) -> float:
        return dict(self.annoyance_cost_usd).get(rail, 0.0)


Action = Literal["retry", "route_to_update_method", "stop"]


@dataclass
class Decision:
    action: Action
    delay_hours: float | None
    bucket: int | None
    p_success: float
    p_gone: float
    expected_value_usd: float
    propensity: float
    rationale: str
    schedule_h: tuple[float, ...] = ()
    curve: pd.DataFrame | None = None


class SupportMap:
    """Observation counts per (reason_class, rail, bucket) from training data."""

    def __init__(self, train: pd.DataFrame, min_support: int = 10):
        self.min_support = min_support
        key = list(zip(
            train["reason_class"].astype(str),
            train["rail"].astype(str),
            train["dt_bucket"].astype(int),
        ))
        self.counts = pd.Series(key).value_counts().to_dict()

    def supported(self, reason_class: str, rail: str, bucket: int) -> bool:
        return self.counts.get((reason_class, rail, int(bucket)), 0) >= self.min_support


def _in_quiet_hours(local_hour: np.ndarray, window: tuple[int, int]) -> np.ndarray:
    start, end = window
    h = np.asarray(local_hour, dtype=float)
    if start <= end:
        return (h >= start) & (h < end)
    return (h >= start) | (h < end)


def candidate_grid(n_per_bucket: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Candidate delays (hours since failure) and their bucket ids."""
    delays, buckets = [], []
    for b in range(N_DT_BUCKETS):
        lo, hi = DT_BUCKET_EDGES_H[b], DT_BUCKET_EDGES_H[b + 1]
        pts = np.linspace(lo + (hi - lo) / (2 * n_per_bucket),
                          hi - (hi - lo) / (2 * n_per_bucket), n_per_bucket)
        delays.extend(pts)
        buckets.extend([b] * n_per_bucket)
    return np.array(delays, dtype=float), np.array(buckets, dtype=int)


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


@dataclass
class Plan:
    """Result of planning for a batch of invoices (first decision only is
    exposed as a distribution; the full best schedule is kept for inspection)."""

    pi: np.ndarray            # (n, N_DT_BUCKETS)  first-action propensities
    delays: np.ndarray        # (n, N_DT_BUCKETS)  delay used if that bucket is first
    q: np.ndarray             # (n, N_DT_BUCKETS)  schedule value of committing to that bucket first
    q_first: np.ndarray       # (n, N_DT_BUCKETS)  first-attempt EV at that bucket's delay
                              #                    (the reward model for first-decision OPE)
    value: np.ndarray         # (n,)  value of the best schedule (0 if no retry)
    schedule: list            # per invoice: tuple of delays in hours
    p_gone: np.ndarray        # (n,)
    p_first: np.ndarray       # (n,)  P(success) of the first planned attempt
    viable: np.ndarray | None = None  # (n, N_DT_BUCKETS) bucket passes constraints
    schedule_by_bucket: list | None = None  # per invoice: {bucket: best schedule starting there}


class RetryPlanner:
    """Exact finite-horizon planner over retry schedules."""

    def __init__(self, model, cfg: PolicyConfig | None = None,
                 rules: NetworkRules | None = None,
                 dispute_model: DisputeModel | None = None,
                 support: SupportMap | None = None,
                 rng: np.random.Generator | None = None):
        self.model = model
        self.cfg = cfg or PolicyConfig()
        self.rules = rules or NetworkRules()
        self.dispute = dispute_model or TableDisputeModel()
        self.support = support
        self.rng = rng or np.random.default_rng(0)
        self.grid, self.grid_bucket = candidate_grid(self.cfg.candidates_per_bucket)
        self._combos = self._enumerate_schedules()

    # -- schedule enumeration --------------------------------------------------

    def _enumerate_schedules(self) -> list[np.ndarray]:
        """All increasing candidate-index tuples of length 1..horizon that
        respect minimum spacing. One array of shape (m_L, L) per length."""
        C = len(self.grid)
        out = []
        for L in range(1, self.cfg.horizon + 1):
            keep = []
            for combo in combinations(range(C), L):
                d = self.grid[list(combo)]
                if L == 1 or np.all(np.diff(d) >= self.cfg.min_spacing_h):
                    keep.append(combo)
            out.append(np.array(keep, dtype=int).reshape(-1, L))
        return out

    # -- components -------------------------------------------------------------

    def _components(self, invoices: pd.DataFrame, theta: np.ndarray | None = None):
        """pi (n,), h (n, K, C), viable (n, C), K per invoice (n,)."""
        n, C, K = len(invoices), len(self.grid), self.cfg.horizon
        triples = pd.DataFrame({
            "invoice_id": np.repeat(invoices["invoice_id"].to_numpy(), K * C),
            "attempt_index": np.tile(np.repeat(np.arange(K), C), n),
            "delay_hours": np.tile(self.grid, n * K),
        })
        feats = build_features(invoices, triples)
        if hasattr(self.model, "predict_components"):
            if hasattr(self.model, "_designs"):
                Xh, Xc = self.model._designs(feats, fit=False)
                nh = self.model.n_h_
                from scipy.special import expit

                def predict(th):
                    return expit(Xc @ th[nh:]), expit(Xh @ th[:nh])
            else:
                def predict(th):
                    return self.model.predict_components(feats, th)
        else:  # plain classifier: no cure component
            def predict(th):
                h_ = self.model.predict_proba1(feats)
                return np.zeros_like(h_), h_
        self._predict = predict
        pi, h = predict(theta if theta is not None else getattr(self.model, "theta_", None))
        pi = pi.reshape(n, K, C)[:, 0, 0]
        h = h.reshape(n, K, C)

        local_hour = feats["local_hour"].to_numpy().reshape(n, K, C)[:, 0, :]
        quiet = _in_quiet_hours(local_hour, self.cfg.quiet_hours_local)
        viable = ~quiet & (self.grid[None, :] >= self.cfg.min_spacing_h)

        rc = feats["reason_class"].astype(str).to_numpy().reshape(n, K, C)[:, 0, 0]
        rails = invoices["rail"].astype(str).to_numpy()
        if self.support is not None:
            sup = np.array([[self.support.supported(rc[i], rails[i], int(b))
                             for b in self.grid_bucket] for i in range(n)])
            viable &= sup

        reasons = invoices["decline_reason"].astype(str).to_numpy()
        advice = (invoices["network_advice"].astype(str).to_numpy()
                  if "network_advice" in invoices.columns
                  else np.full(n, NetworkAdvice.NONE.value))
        blocked = np.array([DeclineReason(r) in HARD_DECLINES or NetworkAdvice(a) in NO_RETRY_ADVICE
                            for r, a in zip(reasons, advice)])
        viable[blocked] = False

        caps = np.array([min(self.cfg.max_attempts, self.rules.attempt_cap(Rail(r)))
                         for r in rails])
        return pi, h, viable, caps, feats

    def _per_attempt_terms(self, invoices: pd.DataFrame):
        """Reward pieces that do not depend on the model: (n, K, C) arrays."""
        n, C, K = len(invoices), len(self.grid), self.cfg.horizon
        amount = invoices["amount_usd"].to_numpy(dtype=float)[:, None, None]
        rails = invoices["rail"].astype(str).to_numpy()
        disc = np.exp(-self.cfg.daily_discount * self.grid / 24.0)[None, None, :]
        p_disp = np.array([[[self.dispute(k, float(t), Rail(r)) for t in self.grid]
                            for k in range(K)] for r in rails])
        gain = amount * disc - p_disp * (amount + self.cfg.dispute_fee_usd)   # on success
        cost = np.array([self.cfg.attempt_cost_usd + self.cfg.annoyance(r) for r in rails])
        return gain, cost

    # -- planning ------------------------------------------------------------------

    def _schedule_values(self, pi, h, viable, caps, gain, cost, chunk: int = 200):
        """Value of every schedule for every invoice, one (n, m_L) array per length."""
        n = len(pi)
        out = [np.full((n, len(idx)), -np.inf) for idx in self._combos]
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            for L, idx in enumerate(self._combos, start=1):
                if idx.size == 0:
                    continue
                # gather: (b, m, L)
                kk = np.arange(L)[None, :]
                hh = h[s:e][:, kk, idx]                   # h[i, k, idx[m, k]]
                gg = gain[s:e][:, kk, idx]
                ok = np.all(viable[s:e][:, idx], axis=2) & (caps[s:e, None] >= L)
                surv_alive = np.cumprod(1 - hh, axis=2)   # (b, m, L)
                prev_alive = np.concatenate(
                    [np.ones_like(surv_alive[..., :1]), surv_alive[..., :-1]], axis=2)
                p_i = pi[s:e][:, None, None]
                p_succ_k = (1 - p_i) * hh * prev_alive
                p_attempt_k = p_i + (1 - p_i) * prev_alive
                v = np.sum(p_succ_k * gg - p_attempt_k * cost[s:e][:, None, None], axis=2)
                out[L - 1][s:e] = np.where(ok, v, -np.inf)
        return out

    def _first_action_summary(self, values):
        """Best value and schedule per first-action bucket: (n, N_B) each."""
        n = values[0].shape[0]
        q = np.full((n, N_DT_BUCKETS), -np.inf)
        best_L = np.full((n, N_DT_BUCKETS), -1)
        best_m = np.full((n, N_DT_BUCKETS), -1)
        for L, (idx, v) in enumerate(zip(self._combos, values)):
            if idx.size == 0:
                continue
            fb = self.grid_bucket[idx[:, 0]]
            for b in range(N_DT_BUCKETS):
                cols = np.flatnonzero(fb == b)
                if cols.size == 0:
                    continue
                sub = v[:, cols]
                j = np.argmax(sub, axis=1)
                val = sub[np.arange(n), j]
                better = val > q[:, b]
                q[better, b] = val[better]
                best_L[better, b] = L
                best_m[better, b] = cols[j[better]]
        return q, best_L, best_m

    def plan(self, invoices: pd.DataFrame, explore: bool = True) -> Plan:
        invoices = invoices.reset_index(drop=True)
        n = len(invoices)
        pi, h, viable, caps, _ = self._components(invoices)
        gain, cost = self._per_attempt_terms(invoices)

        values = self._schedule_values(pi, h, viable, caps, gain, cost)
        q, best_L, best_m = self._first_action_summary(values)

        finite = np.isfinite(q)
        q_out = np.where(finite, q, 0.0)
        delays = np.zeros((n, N_DT_BUCKETS))
        q_first = np.zeros((n, N_DT_BUCKETS))
        for b in range(N_DT_BUCKETS):
            for i in np.flatnonzero(finite[:, b]):
                c = self._combos[best_L[i, b]][best_m[i, b], 0]
                delays[i, b] = self.grid[c]
                q_first[i, b] = (1 - pi[i]) * h[i, 0, c] * gain[i, 0, c] - cost[i]

        best_val = np.where(finite.any(1), q.max(1), -np.inf)
        retry = best_val > 0
        best_b = np.argmax(np.where(finite, q, -np.inf), axis=1)

        # -- propensities: Thompson frequency of the first bucket ----------------
        pi_out = np.zeros((n, N_DT_BUCKETS))
        if explore and hasattr(self.model, "sample_params") and self.cfg.n_posterior_samples > 0:
            S = self.cfg.n_posterior_samples
            thetas = self.model.sample_params(S, self.rng)
            thetas = self.model.theta_ + self.cfg.posterior_temperature * (thetas - self.model.theta_)
            counts = np.zeros((n, N_DT_BUCKETS))
            K, C = self.cfg.horizon, len(self.grid)
            for th in thetas:
                pi_s, h_s = self._predict(th)
                pi_s = pi_s.reshape(n, K, C)[:, 0, 0]
                h_s = h_s.reshape(n, K, C)
                v_s = self._schedule_values(pi_s, h_s, viable, caps, gain, cost)
                q_s, _, _ = self._first_action_summary(v_s)
                bs = np.argmax(np.where(np.isfinite(q_s), q_s, -np.inf), axis=1)
                counts[np.arange(n), bs] += 1.0
            freq = counts / S
        else:
            freq = np.zeros((n, N_DT_BUCKETS))
            freq[np.arange(n), best_b] = 1.0

        for i in np.flatnonzero(retry):
            ok = finite[i]
            p = np.where(ok, freq[i] + self.cfg.propensity_floor, 0.0)
            pi_out[i] = p / p.sum()

        schedule, by_bucket = [], []
        p_first = np.zeros(n)
        for i in range(n):
            per_b = {}
            for b in np.flatnonzero(finite[i]):
                idx = self._combos[best_L[i, b]][best_m[i, b]]
                per_b[int(b)] = tuple(float(x) for x in self.grid[idx])
            by_bucket.append(per_b)
            if not retry[i]:
                schedule.append(())
                continue
            b = best_b[i]
            idx = self._combos[best_L[i, b]][best_m[i, b]]
            schedule.append(per_b[int(b)])
            p_first[i] = (1 - pi[i]) * h[i, 0, idx[0]]

        return Plan(pi=pi_out, delays=delays, q=q_out, q_first=q_first,
                    value=np.where(retry, best_val, 0.0), schedule=schedule,
                    p_gone=pi, p_first=p_first, viable=finite, schedule_by_bucket=by_bucket)


    def plan_greedy(self, invoices: pd.DataFrame) -> list:
        """Ablation: the argmax policy applied *sequentially* -- at each attempt
        pick the single delay with the best immediate EV, conditioned on the
        previous attempts having failed. This is what a per-attempt classifier
        plus argmax actually does in production."""
        invoices = invoices.reset_index(drop=True)
        n, C = len(invoices), len(self.grid)
        pi, h, viable, caps, _ = self._components(invoices)
        gain, cost = self._per_attempt_terms(invoices)
        out = []
        for i in range(n):
            sched, last, alive = [], -np.inf, 1.0
            p_gone = pi[i]
            for k in range(int(caps[i])):
                ok = viable[i] & (self.grid >= last + self.cfg.min_spacing_h)
                if not ok.any():
                    break
                ev = (1 - p_gone) * h[i, k] * gain[i, k] - cost[i]
                ev = np.where(ok, ev, -np.inf)
                c = int(np.argmax(ev))
                if ev[c] <= 0:
                    break
                sched.append(float(self.grid[c]))
                last = self.grid[c]
                # Bayes update on "gone" after an (assumed) failure.
                p_gone = p_gone / (p_gone + (1 - p_gone) * (1 - h[i, k, c]))
            out.append(tuple(sched))
        return out


# ---------------------------------------------------------------------------
# Single-invoice API
# ---------------------------------------------------------------------------


class RetryPolicy:
    """Wraps a planner for one-invoice-at-a-time decisions."""

    def __init__(self, model, invoices: pd.DataFrame, support: SupportMap | None = None,
                 cfg: PolicyConfig | None = None, rules: NetworkRules | None = None,
                 dispute_model: DisputeModel | None = None,
                 rng: np.random.Generator | None = None):
        self.invoices = invoices.set_index("invoice_id", drop=False)
        self.cfg = cfg or PolicyConfig()
        self.rules = rules or NetworkRules()
        self.rng = rng or np.random.default_rng(0)
        self.planner = RetryPlanner(model, self.cfg, self.rules, dispute_model, support, self.rng)

    def decide(self, invoice_id: str, attempt_index: int = 0) -> Decision:
        inv = self.invoices.loc[[invoice_id]].reset_index(drop=True)
        reason = DeclineReason(inv["decline_reason"].iloc[0])
        advice = NetworkAdvice(inv["network_advice"].iloc[0]) if "network_advice" in inv else NetworkAdvice.NONE
        cap = min(self.cfg.max_attempts, self.rules.attempt_cap(Rail(inv["rail"].iloc[0])))

        if reason in HARD_DECLINES:
            return Decision("route_to_update_method", None, None, 0.0, 1.0, 0.0, 1.0,
                            f"{reason.value} is a hard decline; retrying it is prohibited")
        if advice in NO_RETRY_ADVICE:
            return Decision("route_to_update_method", None, None, 0.0, 1.0, 0.0, 1.0,
                            f"network advice '{advice.value}' forbids a retry")
        if attempt_index >= cap:
            return Decision("stop", None, None, 0.0, 0.0, 0.0, 1.0,
                            f"attempt cap of {cap} reached for this rail")

        # Remaining attempts shrink with attempt_index; shift the planner's
        # attempt index by re-labelling the invoice's history length.
        planner = self.planner
        if attempt_index > 0:
            cfg = PolicyConfig(**{**self.cfg.__dict__, "horizon": max(1, self.cfg.horizon - attempt_index),
                                  "max_attempts": cap - attempt_index})
            planner = RetryPlanner(self.planner.model, cfg, self.rules, self.planner.dispute,
                                   self.planner.support, self.rng)

        plan = planner.plan(inv)
        if plan.pi[0].sum() <= 0:
            if plan.viable is not None and plan.viable[0].any():
                return Decision("route_to_update_method", None, None, float(plan.p_first[0]),
                                float(plan.p_gone[0]), float(plan.value[0]), 1.0,
                                "no schedule has positive expected value; ask for a new method")
            return Decision("stop", None, None, 0.0, float(plan.p_gone[0]), 0.0, 1.0,
                            "no candidate delay passes constraints and support guard")

        b = int(self.rng.choice(N_DT_BUCKETS, p=plan.pi[0]))
        explored = b != int(np.argmax(plan.q[0] * (plan.pi[0] > 0)))
        curve = pd.DataFrame({"bucket": np.arange(N_DT_BUCKETS), "delay_h": plan.delays[0],
                              "value": plan.q[0], "propensity": plan.pi[0]})
        return Decision(
            action="retry", delay_hours=float(plan.delays[0, b]), bucket=b,
            p_success=float(plan.p_first[0]), p_gone=float(plan.p_gone[0]),
            expected_value_usd=float(plan.q[0, b]), propensity=float(plan.pi[0, b]),
            rationale=("posterior draw (exploration)" if explored else
                       f"best of {len(plan.schedule[0])}-attempt schedule; "
                       f"P(gone)={plan.p_gone[0]:.2f}"),
            schedule_h=plan.schedule_by_bucket[0].get(b, plan.schedule[0]), curve=curve,
        )


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def _blocked(invoices: pd.DataFrame) -> np.ndarray:
    reasons = invoices["decline_reason"].astype(str).to_numpy()
    advice = (invoices["network_advice"].astype(str).to_numpy()
              if "network_advice" in invoices.columns
              else np.full(len(invoices), NetworkAdvice.NONE.value))
    return np.array([DeclineReason(r) in HARD_DECLINES or NetworkAdvice(a) in NO_RETRY_ADVICE
                     for r, a in zip(reasons, advice)])


def fixed_ladder_policy(invoices: pd.DataFrame, ladder_h: tuple[float, ...],
                        attempt_index: int = 0):
    """The incumbent: same delay for everyone, regardless of why they failed."""
    n, k = len(invoices), N_DT_BUCKETS
    h = ladder_h[min(attempt_index, len(ladder_h) - 1)]
    b = int(dt_bucket(h))
    pi = np.zeros((n, k)); pi[:, b] = 1.0
    delays = np.zeros((n, k)); delays[:, b] = h
    pi[_blocked(invoices)] = 0.0
    return pi, delays


def month_end_policy(invoices: pd.DataFrame, attempt_index: int = 0, n_per_bucket: int = 2):
    """'Just wait for payday.' The heuristic the model must beat to be worth shipping."""
    grid, _ = candidate_grid(n_per_bucket)
    n, k = len(invoices), N_DT_BUCKETS
    pi = np.zeros((n, k)); delays = np.zeros((n, k))
    fail = invoices["fail_time_h"].to_numpy(dtype=float)
    mkts = invoices["market"].astype(str).to_numpy()
    blocked = _blocked(invoices)
    for i in range(n):
        if blocked[i]:
            continue
        cal = _calendar(fail[i] + grid, pd.Series([mkts[i]] * len(grid)))
        score = cal["days_to_payday"].to_numpy() + grid / 2000.0
        h = float(grid[int(np.argmin(score))])
        b = int(dt_bucket(h))
        pi[i, b] = 1.0
        delays[i, b] = h
    return pi, delays
