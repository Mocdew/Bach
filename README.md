# recoup

Failed-payment recovery timing for [Bachs](https://bachs.io) merchants.

Decides **when** to retry a failed invoice, **how many times**, and **when to
stop and ask for a new payment method** — as one planned schedule, not a
sequence of independent guesses. Independent open-source package, not built or
endorsed by Bachs.

```
cure-hazard model  ->  exact finite-horizon schedule planner  ->  Thompson-sampled exploration
                   ->  cross-fitted off-policy evaluation      ->  deployment gate
```

---

## The problem has three layers

**1. Estimate.** Two things, not one: the probability the customer is *gone*
(churned, card in a drawer — nothing you do will recover them), and the
probability a retry at time *t* on attempt *k* succeeds *given they are not*.
Conflate them and the model confounds "not funded yet" with "never coming
back": it over-waits for zombies and over-retries them.

**2. Plan.** A dunning window has 3–4 attempts. The value of retrying at 24h
depends on what you'll do at 72h if it fails, and whether a failure at 24h
should change your mind about attempt three. That's a sequential decision, and
with a couple of dozen candidate delays it's small enough to solve *exactly*.

**3. Learn safely.** Training data comes from whatever ladder was already
running. Any policy that deviates from it is being evaluated on delays the logs
barely cover, and any model that both chose the policy and grades it is marking
its own homework.

Each layer gets its own tool.

## Layer 1 — mixture-cure hazard with partial pooling

```
P(success at attempt k, delay t | x) = (1 − π(x)) · h(t, k, x)

logit π(x)     = δ + u′_merchant + ζ·(active_in_window, tenure, prior history)
logit h(t,k,x) = α_class + f_class(elapsed) + g(days_to_payday | class)
               + cyclic(local hour) + cyclic(day of week | wallet rail)
               + β_class·k + advice + γ·x + u_merchant
```

Fitted by penalised maximum likelihood with an **invoice-level grouped
likelihood** — an invoice that fails three times at good delays is evidence for
π; one that fails once at a bad delay is evidence about h. That's what
identifies the cure fraction. Merchant offsets are ridge-penalised, which is
the MAP form of a hierarchical normal prior: a merchant with 40 attempts is
pulled to the population curve, one with 40,000 gets their own. **There is no
tier router any more** — one model, continuous pooling.

A Laplace approximation around the MAP gives a posterior. It's crude, but it's
what makes Thompson sampling possible.

Everything is scipy. `models.py` also keeps a monotone GBM and a Beta-Binomial
table as baselines.

## Layer 2 — exact schedule planning

`RetryPlanner` enumerates every increasing schedule of 1..4 candidate delays
that respects spacing, quiet hours, network caps and the support guard, and
values each one in closed form under the model:

```
V(schedule) = Σ_k  P(reach k) · [ P(succeed at k)·amount·disc(t_k) − cost_k − P(succeed)·P(dispute|t_k,k)·(amount+fee) ]
P(reach k)  = π + (1−π)·Π_{j<k}(1−h_j)
```

`cost_k` includes an **annoyance cost on wallet rails** — a mobile-money retry
is a push prompt on the customer's phone, chargebacks don't exist there, and
without this term nothing stops the optimiser from prompting forever.

`plan_greedy` is the ablation: argmax over the next delay, applied one attempt
at a time with a Bayes update on P(gone) after each assumed failure. It
reliably picks a *better first attempt* and a *worse schedule*, which is why
first-decision evaluation alone was misleading.

The constraint layer runs before any optimisation. Hard decline codes and
network advice (`NetworkAdvice.DO_NOT_RETRY`, `CANCELLED_RECURRING`,
`UPDATED_INFO_REQUIRED`) route straight to "ask for a new method"; per-rail
attempt caps live in `NetworkRules`.

## Layer 3 — Thompson sampling and cross-fitted OPE

Exploration: instead of ε-uniform over delay buckets (which spends a quarter
of retries on delays everyone knows are bad), the planner re-plans under
posterior draws and acts on the draw. Exploration lands where the model is
actually uncertain. Propensities are the Monte-Carlo frequency of each first
bucket, floored so every viable action stays visible to off-policy evaluation.

Evaluation: `cross_fitted_plan` fits the model on two halves of the training
data; the policy is planned under one and valued (the `q_hat` of the
doubly-robust estimator) under the other, then the roles swap. The
`deployment_gate` then refuses to swap policies unless a paired bootstrap CI on
the DR difference excludes zero and effective sample size clears a floor.

## What was removed, and why

| Removed | Why |
|---|---|
| `payday_window = exp(−½(d/3.2)²)` feature | It was the simulator's ground-truth funding function copied verbatim. The model was being handed the generative variable; the benchmark lift measured nothing. Calendar inputs are now raw. |
| `from simulator import true_dispute_prob` in the policy | Production code cannot depend on the oracle. The dispute model is injected (`TableDisputeModel`, fittable from the disputes endpoint). |
| `SIM_START` imported into `features.py` | Same. The epoch is `domain.DEFAULT_EPOCH` or an argument. |
| Two-tier model router | Replaced by continuous partial pooling in one model. |
| Argmax over the next delay | Replaced by exact schedule planning. |
| ε-uniform exploration | Replaced by Thompson sampling. |
| Self-graded DR | Replaced by cross-fitting. |
| Random-delay baseline | Told you nothing a reviewer needs. |

## Results

`python benchmark.py` — 9,000 invoices, 3,300 recurring customers, 12
merchants, 6 markets, 70/30 temporal split. All numbers describe the simulator,
not the world; the misspecification table is the part that matters.

**Outcome model** (test attempts, no leaked features):

| model | PR-AUC | logloss | ECE |
|---|---|---|---|
| cure-hazard | **0.360** | **0.384** | 0.044 |
| GBM (monotone in attempt) | 0.347 | 0.386 | 0.034 |
| Beta-Binomial table | 0.267 | 0.412 | 0.038 |

P(gone): model 0.21 vs true churned share among test attempts 0.25. The cure
fraction is identified from repeated failures, not from a label.

**Full dunning schedule**, true value per failed invoice under the oracle:

| policy | value | vs. ladder |
|---|---|---|
| fixed ladder 24h / 72h / 168h (incumbent) | $12.92 | — |
| greedy argmax, one attempt at a time | $15.46 | +19.7% |
| **exact schedule planning** | **$17.43** | **+34.9%** |

Planning ahead is worth about as much again as the model. Evaluated on the
*first decision only*, greedy looks better than planned ($9.65 vs $8.84) — the
planner deliberately spends the first attempt on information. This is why the
old first-decision-only evaluation was the wrong yardstick.

**Does the lift survive a different world?** Re-simulate under a perturbed
ground truth, refit, replan:

| world | ladder | planned | lift |
|---|---|---|---|
| baseline | $11.69 | $15.50 | +32.6% |
| **no payday effect at all** | $13.52 | $17.76 | **+31.4%** |
| payday lags 3 days | $11.57 | $15.17 | +31.1% |
| payday spread wide | $13.39 | $16.41 | +22.6% |
| payday spread narrow | $11.03 | $15.24 | +38.2% |
| slow infra recovery | $12.26 | $16.02 | +30.7% |
| customers drift faster | $9.50 | $14.48 | +52.5% |

The lift survives removing the payday effect entirely. The value is in
decline-class-specific timing and knowing when to stop, not in the payday
thesis the earlier version of this package led with.

**Volume:**

| invoices | PR-AUC | ladder | planned | lift |
|---|---|---|---|---|
| 1,500 | 0.358 | $13.41 | $16.12 | +20.2% |
| 3,000 | 0.309 | $11.38 | $14.47 | +27.1% |
| 6,000 | 0.363 | $11.69 | $15.50 | +32.6% |
| 9,000 | 0.360 | $12.54 | $16.85 | +34.3% |

The earlier GBM-plus-argmax lost money below 3,000 invoices. The pooled
parametric model does not, because a small merchant is mostly running the
population curve.

**The gate holds everywhere.** On the baseline benchmark:

```
planned vs incumbent    HOLD: delta +2.21 USD/invoice [-0.23, +5.27] ESS 140
                        -- confidence interval on the improvement includes zero
```

True lift is +35% and the gate still refuses. That is correct: the logged
ladder (ε = 0.25 uniform over buckets) barely visited the delays the planner
wants, so effective sample size is 140 of 2,250 and the evidence cannot yet
justify the swap. Cross-fitted DR lands within 9% of the oracle on the planned
policy — the estimator is honest, the logs are thin. **Ship the constraint
layer with Thompson exploration first; evaluate after a month of its logs.**

## Install and run

```bash
pip install -e .
python benchmark.py            # full evaluation incl. perturbed worlds
python quickstart.py           # what a single decision looks like
python -m pytest tests/ -q
python tests/run_tests.py      # same suite, no pytest needed
```

Python 3.10+, numpy / pandas / scipy / scikit-learn.

## Layout

```
recoup/
  domain.py      rails, decline reasons, network advice + rules, markets, epoch
  simulator.py   generator with a perturbable ground truth (Truth), recurring customers
  features.py    one feature path for training and scoring; customer history
  models.py      CureHazardModel (primary); GBM and Beta-Binomial baselines
  policy.py      constraints, dispute model, schedule planner, Thompson propensities
  evaluate.py    temporal split, oracle (first decision + full schedule), DR, cross-fitting, gate
  bachs.py       API adapter -- SEE THE WARNING IN THAT FILE
benchmark.py     end-to-end comparison incl. misspecified worlds and volume sweep
quickstart.py    single-invoice decision surface
tests/           invariants that would silently cost money if they broke
```

## Honest limitations

- **The posterior is a Laplace approximation** and is wide in directions where
  the intercept and elapsed-time terms trade off. `posterior_temperature`
  scales the draws; the default 0.5 is a tuning choice, not a derivation.
- **Historical Bachs data has no logged propensities.** A deterministic ladder
  has propensity 1 on one bucket and 0 elsewhere, so there is no honest
  pre-deployment estimate of a policy that deviates from it. `assume_propensities()`
  exists but the real answer is: ship the constraint layer with Thompson
  exploration first, learn nothing for a month, then evaluate.
- **Actions are delays.** "Send a reminder before payday" and "retry on the
  customer's other stored method" are probably higher-EV for balance declines
  than any timing choice and are not in the action space yet.
- **Annoyance cost is a constant per rail.** It should be P(churn | attempts) ×
  remaining LTV, fitted from retention data.
- **Decline-code and advice-code mapping is unverified** against the Bachs
  spec. Unmapped codes fail closed.
- **Volume.** Per-merchant deployment below a few thousand failed invoices is
  running the population prior with small merchant offsets. That's fine — it's
  a better ladder — but call it that. The pooled model is a PSP-side product.

MIT.
