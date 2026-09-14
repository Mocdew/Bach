"""Outcome models.

The primary model is a **mixture-cure discrete hazard**:

    P(success at attempt k, delay t | x) = (1 - pi(x)) * h(t, k, x)

``pi`` is the probability the customer is gone for good (churned, card in a
drawer) and ``h`` is the success probability of an attempt *given they are
not*. The two are identified jointly from repeated failures within an invoice:
an invoice that fails three times at good delays is evidence for ``pi``, an
invoice that fails once at a bad delay is evidence about ``h``. Without the
split the model confounds "not funded yet" with "never coming back" and the
planner either over-waits for zombies or over-retries them.

Merchant effects enter both parts as ridge-penalised offsets, which is the
maximum-a-posteriori form of a hierarchical normal prior: a merchant with 40
attempts is pulled almost entirely to the population curve, one with 40,000
gets their own. That replaces the old two-tier router -- there is one model
and the pooling is continuous.

Uncertainty comes from a Laplace approximation around the MAP. It is crude but
it is what makes Thompson sampling possible, and Thompson sampling is what
makes exploration tolerable to a merchant (see ``policy.py``).

Everything is scipy; no probabilistic-programming dependency. The design
matrix is hand-built so that every term is legible to a payments analyst.

``BetaBinomialHazard`` and ``GBMHazard`` remain as baselines.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier

from .domain import N_DT_BUCKETS, MARKETS, NetworkAdvice, Rail
from .features import CATEGORICAL, FEATURES, NUMERIC

LABEL = "success"


# ---------------------------------------------------------------------------
# Cure-hazard model
# ---------------------------------------------------------------------------


def _fourier(x: np.ndarray, period: float, K: int) -> list[np.ndarray]:
    cols = []
    for k in range(1, K + 1):
        cols.append(np.sin(2 * np.pi * k * x / period))
        cols.append(np.cos(2 * np.pi * k * x / period))
    return cols


@dataclass
class CureHazardModel:
    """Mixture-cure discrete hazard with ridge-pooled merchant effects.

    Parameters
    ----------
    merchant_shrinkage
        L2 penalty on merchant offsets. Larger = closer to full pooling.
        In hierarchical terms this is 1 / (2 sigma^2) of the merchant prior.
    ridge
        Small L2 on all non-intercept coefficients for conditioning.
    """

    merchant_shrinkage: float = 4.0
    ridge: float = 0.02
    max_iter: int = 1500

    theta_: np.ndarray | None = None
    cov_: np.ndarray | None = None
    merchants_: list = field(default_factory=list)
    h_names_: list = field(default_factory=list)
    c_names_: list = field(default_factory=list)
    scale_: dict = field(default_factory=dict)
    n_h_: int = 0
    converged_: bool = False

    # -- design --------------------------------------------------------------

    def _num(self, df: pd.DataFrame, col: str, transform=None) -> np.ndarray:
        x = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        return transform(x) if transform else x

    def _hazard_design(self, df: pd.DataFrame, fit: bool) -> tuple[np.ndarray, list[str]]:
        rc = df["reason_class"].astype(str).to_numpy()
        rail = df["rail"].astype(str).to_numpy()
        mkt = df["market"].astype(str).to_numpy()
        adv = df["network_advice"].astype(str).to_numpy()

        elapsed = self._num(df, "delay_hours")
        le = np.log1p(elapsed)
        d2p = self._num(df, "days_to_payday")
        hour = self._num(df, "local_hour")
        dow = self._num(df, "local_dow")
        k = self._num(df, "attempt_index")

        cols, names = [], []

        def add(name, x):
            cols.append(np.asarray(x, dtype=float)); names.append(name)

        classes = ("balance", "infra", "opaque")
        for c in classes:
            ind = (rc == c).astype(float)
            add(f"int:{c}", ind)
            # elapsed-time shape, one curve per decline class
            add(f"le:{c}", ind * le)
            add(f"le2:{c}", ind * le ** 2)
            add(f"lin:{c}", ind * elapsed / 100.0)
            add(f"k:{c}", ind * k)

        # payday: raw distance to the market's payday, two shapes -- one for
        # balance declines (the thesis) and one for everything else.
        for grp, ind in (("bal", (rc == "balance").astype(float)),
                         ("oth", (rc != "balance").astype(float))):
            add(f"d2p:{grp}", ind * d2p / 10.0)
            add(f"d2p2:{grp}", ind * (d2p / 10.0) ** 2)
            add(f"d2p<=2:{grp}", ind * (d2p <= 2.0))
            add(f"d2p<=5:{grp}", ind * (d2p <= 5.0))

        # hour of day (cyclic), extra shape for bank rails (business hours)
        bank = (rail == Rail.BANK_TRANSFER.value).astype(float)
        for i, f in enumerate(_fourier(hour, 24.0, 2)):
            add(f"hour_f{i}", f)
            add(f"hour_f{i}:bank", f * bank)

        # day of week (cyclic), wallet rails only
        wallet = np.isin(rail, [Rail.MOBILE_MONEY.value, Rail.STABLECOIN.value]).astype(float)
        for i, f in enumerate(_fourier(dow, 7.0, 1)):
            add(f"dow_f{i}:wallet", f * wallet)

        add("advice:try_later", (adv == NetworkAdvice.TRY_AGAIN_LATER.value).astype(float))

        add("log_amount", self._num(df, "log_amount"))
        add("active", self._num(df, "active_in_window"))
        add("log_tenure", self._num(df, "tenure_days", np.log1p))
        add("log_prior_succ", self._num(df, "prior_successful_payments", np.log1p))
        add("n_prior_fail", self._num(df, "n_prior_failures"))
        add("n_prior_rec", self._num(df, "n_prior_recoveries"))
        add("has_prior_rec", self._num(df, "has_prior_recovery"))
        add("log_prev_rec_delay", self._num(df, "prev_recovery_delay_h", np.log1p))

        for r in Rail:
            add(f"rail:{r.value}", (rail == r.value).astype(float))
        for m in MARKETS:
            add(f"mkt:{m}", (mkt == m).astype(float))

        X = np.column_stack(cols)
        return X, names

    def _cure_design(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        rail = df["rail"].astype(str).to_numpy()
        cols, names = [], []

        def add(name, x):
            cols.append(np.asarray(x, dtype=float)); names.append(name)

        add("int", np.ones(len(df)))
        add("active", self._num(df, "active_in_window"))
        add("log_tenure", self._num(df, "tenure_days", np.log1p))
        add("log_prior_succ", self._num(df, "prior_successful_payments", np.log1p))
        add("n_prior_fail", self._num(df, "n_prior_failures"))
        add("n_prior_rec", self._num(df, "n_prior_recoveries"))
        add("has_prior_rec", self._num(df, "has_prior_recovery"))
        add("log_amount", self._num(df, "log_amount"))
        for r in Rail:
            add(f"rail:{r.value}", (rail == r.value).astype(float))
        return np.column_stack(cols), names

    def _merchant_design(self, df: pd.DataFrame) -> np.ndarray:
        m = df["merchant_id"].astype(str).to_numpy()
        X = np.zeros((len(df), len(self.merchants_)))
        idx = {mid: j for j, mid in enumerate(self.merchants_)}
        for i, mid in enumerate(m):
            j = idx.get(mid)
            if j is not None:
                X[i, j] = 1.0
        return X

    def _scale(self, X: np.ndarray, names: list[str], key: str, fit: bool) -> np.ndarray:
        """Standardise continuous columns; leave indicators and intercepts alone."""
        if fit:
            mu = X.mean(axis=0)
            sd = X.std(axis=0)
            is_ind = np.array([set(np.unique(X[:, j])) <= {0.0, 1.0} for j in range(X.shape[1])])
            mu[is_ind] = 0.0
            sd[is_ind] = 1.0
            sd[sd < 1e-9] = 1.0
            self.scale_[key] = (mu, sd)
        mu, sd = self.scale_[key]
        return (X - mu) / sd

    def _designs(self, df: pd.DataFrame, fit: bool = False):
        Xh, hn = self._hazard_design(df, fit)
        Xh = self._scale(Xh, hn, "h", fit)
        Xc, cn = self._cure_design(df)
        Xc = self._scale(Xc, cn, "c", fit)
        M = self._merchant_design(df)
        Xh = np.hstack([Xh, M])
        Xc = np.hstack([Xc, M])
        if fit:
            self.h_names_ = hn + [f"mch:{m}" for m in self.merchants_]
            self.c_names_ = cn + [f"mch:{m}" for m in self.merchants_]
            self.n_h_ = Xh.shape[1]
        return Xh, Xc

    # -- likelihood ------------------------------------------------------------

    def _penalty_vec(self) -> np.ndarray:
        """Per-parameter L2 weights."""
        pen = []
        for n in self.h_names_:
            if n.startswith("mch:"):
                pen.append(self.merchant_shrinkage)
            elif n.startswith("int:"):
                pen.append(self.ridge * 0.25)
            else:
                pen.append(self.ridge)
        for n in self.c_names_:
            if n.startswith("mch:"):
                pen.append(self.merchant_shrinkage)
            elif n == "int":
                pen.append(self.ridge * 0.25)
            else:
                pen.append(self.ridge)
        return np.asarray(pen, dtype=float)

    def _nll_grad(self, theta, Xh, Xc_inv, inv_idx, y, A, pen):
        nh = self.n_h_
        bh, bc = theta[:nh], theta[nh:]
        eta = Xh @ bh
        h = np.clip(expit(eta), 1e-9, 1 - 1e-9)
        z = Xc_inv @ bc
        pi = np.clip(expit(z), 1e-9, 1 - 1e-9)

        lr = y * np.log(h) + (1 - y) * np.log1p(-h)
        b_inv = np.bincount(inv_idx, weights=lr, minlength=len(A))
        B = np.exp(b_inv)
        s = pi * A + (1 - pi) * B
        s = np.maximum(s, 1e-300)
        ll = np.sum(np.log(s)) - np.sum(pen * theta ** 2)

        g_z = pi * (1 - pi) * (A - B) / s
        g_b = (1 - pi) * B / s
        g_eta = g_b[inv_idx] * (y - h)
        grad = np.concatenate([Xh.T @ g_eta, Xc_inv.T @ g_z]) - 2 * pen * theta
        return -ll, -grad

    def fit(self, df: pd.DataFrame) -> "CureHazardModel":
        df = df.reset_index(drop=True)
        self.merchants_ = sorted(df["merchant_id"].astype(str).unique().tolist())
        Xh, Xc = self._designs(df, fit=True)
        y = df[LABEL].to_numpy(dtype=float)

        inv_codes, inv_uniques = pd.factorize(df["invoice_id"].astype(str))
        first = np.zeros(len(inv_uniques), dtype=int)
        first[inv_codes[::-1]] = np.arange(len(df))[::-1]  # first row of each invoice
        Xc_inv = Xc[first]
        A = 1.0 - np.bincount(inv_codes, weights=y, minlength=len(inv_uniques)).clip(0, 1)

        pen = self._penalty_vec()
        theta0 = np.zeros(Xh.shape[1] + Xc.shape[1])
        theta0[self.n_h_] = -1.0          # cure intercept: prior ~27% gone
        base = np.log(max(y.mean(), 1e-3) / max(1 - y.mean(), 1e-3))
        for j, n in enumerate(self.h_names_):
            if n.startswith("int:"):
                theta0[j] = base + 0.5

        res = minimize(self._nll_grad, theta0, args=(Xh, Xc_inv, inv_codes, y, A, pen),
                       jac=True, method="L-BFGS-B",
                       options={"maxiter": self.max_iter, "ftol": 1e-10})
        self.theta_ = res.x
        self.converged_ = bool(res.success)
        self.cov_ = self._laplace_cov(res.x, Xh, Xc_inv, inv_codes, y, A, pen)
        return self

    def _laplace_cov(self, theta, *args) -> np.ndarray:
        """Inverse Hessian of the negative log posterior, by finite differences
        of the analytic gradient. Eigenvalues are floored so sampling is safe."""
        d = len(theta)
        eps = 1e-4
        H = np.empty((d, d))
        g0 = self._nll_grad(theta, *args)[1]
        for j in range(d):
            t = theta.copy(); t[j] += eps
            H[:, j] = (self._nll_grad(t, *args)[1] - g0) / eps
        H = 0.5 * (H + H.T)
        w, V = np.linalg.eigh(H)
        w = np.maximum(w, 0.5)
        return (V / w) @ V.T

    # -- prediction ----------------------------------------------------------

    def predict_components(self, df: pd.DataFrame, theta: np.ndarray | None = None
                           ) -> tuple[np.ndarray, np.ndarray]:
        """Return (pi, h): P(gone) and P(success | not gone) per row."""
        assert self.theta_ is not None, "call fit first"
        theta = self.theta_ if theta is None else theta
        Xh, Xc = self._designs(df, fit=False)
        h = expit(Xh @ theta[:self.n_h_])
        pi = expit(Xc @ theta[self.n_h_:])
        return pi, h

    def predict_proba1(self, df: pd.DataFrame, theta: np.ndarray | None = None) -> np.ndarray:
        pi, h = self.predict_components(df, theta)
        return np.clip((1 - pi) * h, 1e-6, 1 - 1e-6)

    def sample_params(self, n: int, rng: np.random.Generator) -> np.ndarray:
        assert self.theta_ is not None and self.cov_ is not None
        return rng.multivariate_normal(self.theta_, self.cov_, size=n, method="eigh")

    def coef_table(self) -> pd.DataFrame:
        """Named coefficients with Laplace standard errors -- for the analyst."""
        names = self.h_names_ + self.c_names_
        part = ["hazard"] * len(self.h_names_) + ["cure"] * len(self.c_names_)
        se = np.sqrt(np.diag(self.cov_)) if self.cov_ is not None else np.nan
        return pd.DataFrame({"part": part, "term": names, "coef": self.theta_, "se": se})


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


@dataclass
class BetaBinomialHazard:
    """Partially pooled cell estimates with a three-level backoff.

    global -> (reason_class, attempt, bucket) -> (reason_class, rail, attempt, bucket)

    Attempt index is part of the key: attempt-2 rows come from customers who
    already failed once and are a different population.
    """

    concentration: float = 24.0
    global_rate_: float = 0.15
    level1_: dict = field(default_factory=dict)
    level2_: dict = field(default_factory=dict)
    parent: "BetaBinomialHazard | None" = None

    @staticmethod
    def _keys(df: pd.DataFrame):
        rc = df["reason_class"].astype(str).to_numpy()
        rail = df["rail"].astype(str).to_numpy()
        b = df["dt_bucket"].to_numpy().astype(int)
        k = np.minimum(df["attempt_index"].to_numpy().astype(int), 2)
        return rc, rail, b, k

    def fit(self, df: pd.DataFrame) -> "BetaBinomialHazard":
        y = df[LABEL].to_numpy(dtype=float)
        base = self.parent.global_rate_ if self.parent else 0.15
        c = self.concentration
        self.global_rate_ = float((y.sum() + c * base) / (len(y) + c)) if len(y) else base

        rc, rail, b, k = self._keys(df)

        self.level1_ = {}
        for key in set(zip(rc, k, b)):
            m = (rc == key[0]) & (k == key[1]) & (b == key[2])
            prior = self.parent.level1_.get(key, self.global_rate_) if self.parent else self.global_rate_
            self.level1_[key] = float((y[m].sum() + c * prior) / (m.sum() + c))

        self.level2_ = {}
        for key in set(zip(rc, rail, k, b)):
            m = (rc == key[0]) & (rail == key[1]) & (k == key[2]) & (b == key[3])
            parent_rate = self.level1_.get((key[0], key[2], key[3]), self.global_rate_)
            prior = self.parent.level2_.get(key, parent_rate) if self.parent else parent_rate
            self.level2_[key] = float((y[m].sum() + c * prior) / (m.sum() + c))
        return self

    def predict_proba1(self, df: pd.DataFrame) -> np.ndarray:
        rc, rail, b, k = self._keys(df)
        out = np.empty(len(df), dtype=float)
        for i in range(len(df)):
            key2 = (rc[i], rail[i], k[i], b[i])
            if key2 in self.level2_:
                out[i] = self.level2_[key2]
            else:
                out[i] = self.level1_.get((rc[i], k[i], b[i]), self.global_rate_)
        return np.clip(out, 1e-6, 1 - 1e-6)


@dataclass
class GBMHazard:
    """Histogram gradient boosting, monotone-decreasing in attempt index."""

    max_iter: int = 260
    learning_rate: float = 0.06
    max_leaf_nodes: int = 24
    min_samples_leaf: int = 40
    l2: float = 1.0
    seed: int = 0
    model_: HistGradientBoostingClassifier | None = None

    def fit(self, df: pd.DataFrame) -> "GBMHazard":
        mono = [0] * len(FEATURES)
        mono[FEATURES.index("attempt_index")] = -1
        self.model_ = HistGradientBoostingClassifier(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2,
            categorical_features=[FEATURES.index(c) for c in CATEGORICAL],
            monotonic_cst=mono,
            early_stopping=True,
            validation_fraction=0.12,
            random_state=self.seed,
        )
        self.model_.fit(_matrix(df), df[LABEL].to_numpy(dtype=int))
        return self

    def predict_proba1(self, df: pd.DataFrame) -> np.ndarray:
        assert self.model_ is not None, "call fit first"
        return self.model_.predict_proba(_matrix(df))[:, 1]


def _matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].copy()
    for c in CATEGORICAL:
        X[c] = X[c].astype("category")
    for c in NUMERIC:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
    return X
