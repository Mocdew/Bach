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

## In plain terms

*Skip this if you're comfortable with the sections below; it says the same
thing without the jargon.*

**The problem.** When a customer's subscription payment fails, the business
has to decide *when* to try charging them again, *how many times*, and *when
to give up and ask for a new card*. Get it wrong and you either lose money you
could have recovered, or you annoy customers into cancelling.

**What most businesses do today.** A fixed ladder: try again after 1 day, then
3 days, then 7 days. Same for everyone, no matter why the payment failed.
That's the baseline we're trying to beat.

**What this package does.** Three pieces:

1. *A model that answers two questions about each failed payment.* "Is this
   customer gone for good?" (card cancelled, moved on — nothing you do will
   work) and "if they're not gone, when are they most likely to have money?"
   Keeping those separate matters: without the split, the model can't tell
   "not paid yet" from "never coming back", so it keeps retrying dead accounts
   and gives up too early on live ones.
2. *A planner that thinks ahead.* Instead of "what's the best time for the
   next retry?", it asks "what's the best *sequence* of retries?" — because
   the right first attempt depends on what you'd do if it fails. Sometimes the
   best first move is a cheap early retry that mostly tells you whether the
   customer is still around.
3. *A safety system.* It refuses to switch the business over to the new
   approach until there's real evidence it's better, and it explores new
   timings cautiously rather than randomly.

**The headline result.** Per failed invoice, in a simulated business:

| approach | recovered per failed invoice |
|---|---|
| fixed ladder (what everyone does) | $12.12 |
| smart timing, one retry at a time | $14.39 — 19% more |
| smart timing, planning the whole sequence | $15.93 — **31% more** |

For every $100 the fixed ladder recovers, the planner recovers about $131.
Around 40% of that gain comes from planning ahead rather than from the
prediction model — a cheap idea that's usually skipped.

**The result we didn't expect.** The original pitch was "salaries land at
month-end, so time retries around payday." We tested what happens if you
delete the payday effect from the simulated world entirely. The gain barely
changed (+26% instead of +29%). The value isn't coming from payday timing. It
comes from two simpler things: treating a "bank was down" failure differently
from an "insufficient funds" failure, and knowing when to stop. That's good
news — the approach doesn't depend on a theory about salaries being right. We
tried six other "what if the world is different" scenarios; the gain held in
all of them, between +20% and +35%.

**Small businesses.** The earlier version only worked for big merchants —
with fewer than ~3,000 failed payments of history it actually *lost* money
versus the ladder. This version gains +19% even at 1,500, because a small
merchant borrows what's been learned across all merchants and only overrides
it where they have enough data of their own.

**Why the system still says "don't switch yet".** The safety gate says HOLD
on every test, and that is the right answer. All the historical data comes
from the old 1/3/7-day ladder, which almost never tried the timings the new
planner wants. So there's no real-world evidence — only the model's opinion —
that those timings work, and the gate refuses to bet revenue on an opinion.
The practical meaning: **you can't validate this from old logs alone.** The
rollout is: switch on the safety rules and cautious exploration first, run for
a month, and *then* the evidence exists to judge the planner.

**What to be honest about.** Every number here is from a simulation —
plausible, not measured; what transfers to a real business is the method, not
the "31%". The system doesn't yet do the two things that may matter most for
"insufficient funds": remind the customer before payday, or try their other
saved payment method. And how much a retry annoys a mobile-money customer is
currently a guess, not a measurement.

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

## Fixed in 0.3 -- found by running real-shaped data through it

| defect | effect | fix |
|---|---|---|
| Simulator's logging policy drew each retry's delay independently | ~2% of histories had attempt *k+1* **before** attempt *k*; harmless in memory, a corrupted decision-log join once data is time-ordered | continuation draws are restricted to later delays and renormalised, so logged propensities stay exact; attempt 0 is bit-identical. Benchmark rerun: the numbers below moved by a few points |
| `RetryPolicy.decide()` for attempt > 0 | scored the hazard at attempt 0 (no fatigue), ignored the failures so far when estimating P(gone), and could schedule a retry in the past | planner takes `attempt_offset`, `past_delays` (Bayes update of P(gone) under every posterior draw) and an earliest-delay mask |
| Adapter delays = difference of two hours-since-epoch floats | a 24h retry read as 23.99999h -- and the ladder rungs sit exactly on bucket edges, so every ladder retry was filed one bucket early | delays rounded to timestamp resolution |
| Adapter dropped `customer_id`, never joined customers, disputes or FX | customer history and cure labels were impossible on real data; NGN amounts would have been read as USD | full join, FX conversion with an unconverted-currency report, `validate_schema()` |
| `test_dispute_cost_shortens_chosen_delays` asserted on the first delay | passed by fixture luck; a planner facing a costly tail can rationally move the first attempt *later* | asserts the schedule's last and mean delay instead |

## Results

`python benchmark.py` — 9,000 invoices, 3,300 recurring customers, 12
merchants, 6 markets, 70/30 temporal split. All numbers describe the simulator,
not the world; the misspecification table is the part that matters.

**Outcome model** (test attempts, no leaked features):

| model | PR-AUC | logloss | ECE |
|---|---|---|---|
| cure-hazard | 0.330 | 0.383 | 0.036 |
| GBM (monotone in attempt) | **0.353** | **0.380** | 0.031 |
| Beta-Binomial table | 0.261 | 0.408 | 0.034 |

The GBM now edges the cure-hazard model on raw attempt-level accuracy; it did
not before the simulator fix below changed the training data. The planner
still runs on the cure-hazard model, because it needs what the GBM cannot give
it: P(gone) separated from the hazard, and a posterior to Thompson-sample. The
comparison that matters is the schedule value, not PR-AUC.

P(gone) per invoice: model 0.16 vs true 0.20 on the test window (churn
prevalence ramps from 0.09 in training to 0.20 in test by construction). The
cure fraction is identified from repeated failures, not from a label.

**Full dunning schedule**, true value per failed invoice under the oracle:

| policy | value | vs. ladder |
|---|---|---|
| fixed ladder 24h / 72h / 168h (incumbent) | $12.12 | — |
| greedy argmax, one attempt at a time | $14.39 | +18.8% |
| **exact schedule planning** | **$15.93** | **+31.4%** |

Planning ahead adds most of what the model alone does again. Evaluated on the
*first decision only*, greedy looks better than planned ($8.79 vs $7.51; the
ladder is $7.09) — the
planner deliberately spends the first attempt on information. This is why the
old first-decision-only evaluation was the wrong yardstick.

**Does the lift survive a different world?** Re-simulate under a perturbed
ground truth, refit, replan:

| world | ladder | planned | lift |
|---|---|---|---|
| baseline | $11.67 | $15.11 | +29.4% |
| **no payday effect at all** | $12.48 | $15.70 | **+25.7%** |
| payday lags 3 days | $12.24 | $14.65 | +19.8% |
| payday spread wide | $13.98 | $17.21 | +23.2% |
| payday spread narrow | $11.13 | $15.00 | +34.8% |
| slow infra recovery | $10.99 | $14.23 | +29.4% |
| customers drift faster | $9.34 | $12.50 | +33.8% |

The lift survives removing the payday effect entirely. The value is in
decline-class-specific timing and knowing when to stop, not in the payday
thesis the earlier version of this package led with.

**Volume:**

| invoices | PR-AUC | ladder | planned | lift |
|---|---|---|---|---|
| 1,500 | 0.293 | $12.30 | $14.64 | +19.1% |
| 3,000 | 0.348 | $11.93 | $14.41 | +20.7% |
| 6,000 | 0.355 | $11.67 | $15.11 | +29.4% |
| 9,000 | 0.365 | $12.04 | $15.85 | +31.7% |

The earlier GBM-plus-argmax lost money below 3,000 invoices. The pooled
parametric model does not, because a small merchant is mostly running the
population curve.

**The gate holds everywhere.** On the baseline benchmark:

```
planned vs incumbent    HOLD: delta -0.02 USD/invoice [-2.25, +2.31] ESS 113
                        -- confidence interval on the improvement includes zero
```

True schedule lift is +31% and the gate still refuses. That is correct: the
logged ladder (ε = 0.25 uniform over buckets) barely visited the delays the
planner wants, so effective sample size is 113 and the evidence cannot yet
justify the swap. Cross-fitted DR lands within 1.1% of the oracle on the
planned policy's first decision — the estimator is honest, the logs are thin.
Note also that the gate judges the first decision, where the true gap is
only $0.42 ($7.51 vs $7.09); see Honest limitations. **Ship the constraint
layer with Thompson exploration first; evaluate after a month of its logs.**

## Synthetic data and the operations layer

The benchmark runs on in-memory tables. A deployment runs on raw payment
objects, and everything between the two -- decline-code mapping, currency
conversion, the customer join, cure labels, the audit log that carries
propensities -- is where integrations quietly break. So the same ground-truth
world is also rendered as **the raw export a merchant would pull from Bachs**:

```
data/synthetic/
  payments.jsonl         Bachs payment objects, in the bachs.FIELD_MAP shape:
                         failures, retries, renewals; amounts in local minor units
  customers.csv          market, signup date, plan tier
  disputes.csv           chargebacks on recovered payments
  customer_events.csv    subscription.cancelled / customer.deleted / card swaps
  legacy_decisions.csv   the incumbent's decision log, with its propensities
  manifest.json          epoch, clock (as_of), FX rates, generator settings
  _truth/                hidden simulator state -- read only by `advance` and `score`
```

It is deliberately untidy the way real exports are: two spellings for some
decline codes, unmapped codes on a share of hard declines (the adapter must
fail closed and say so), `crypto` and `stablecoin` for the same rail, five
currencies, cancellations that lag churn by up to three weeks, and churned
customers who never cancel at all.

The world is split at a cutover date. History before it was generated under
the ε-randomised ladder. After it, first failures, renewals and cancellations
wait in `_truth/pending.jsonl` -- and **no retries exist until the live policy
decides them**. `recoup-ops advance` moves the clock, releases events and
executes the queued retries against the ground-truth hazard. That closes the
loop the rollout plan depends on.

`recoup-ops` is also what the systemd units in `deploy/` run:

```bash
recoup-ops synth                       # write data/synthetic (~10 s, 9,000 invoices)
recoup-ops retrain                     # fit, holdout-check, promote var/state/models/current
recoup-ops plan                        # decide next retries; audit log + queue
recoup-ops advance --hours 24          # (synthetic) execute the queue, move the clock
recoup-ops gate                        # cross-fitted OPE -> var/state/gate/decision.json
recoup-ops score                       # (synthetic) realised $/invoice vs the ladder's oracle value
recoup-ops demo --fresh                # all of it: 45 simulated days of the rollout
```

| job | reads | writes | guard |
|---|---|---|---|
| `retrain` | dataset | `models/<ts>/`, then repoints `models/current` atomically | refuses (exit 2, old model stays) if the fit does not converge or temporal-holdout PR-AUC does not beat the base rate |
| `plan` | model, gate file, audit log | `decisions-<run>.csv`, `queue/<run>.jsonl` | never decides an (invoice, attempt) twice; hard declines and no-retry advice are routed before the model is asked |
| `gate` | model, audit log, dataset | `gate/decision.json` | HOLD on too few logged decisions, on ESS below the floor, or on a CI that includes zero; SWITCH reverts only on evidence of harm |

**HOLD is not "do nothing".** Under HOLD `plan` runs the incumbent ladder,
except that a share ε (`--explore-rate`, default 0.2) of retries follow the
planner's Thompson draw, and the logged propensity is the mixture's. That is
the README's "cautious exploration first" made concrete: at most ε of revenue
is exposed to the new policy, and the gate gets logs it can actually use.

### What the loop did, end to end

`recoup-ops demo --fresh` — 180 days of legacy history, then 45 simulated days
live with weekly retrain and gate, every step through the same CLI the
systemd units run:

| gate run | decision | why |
|---|---|---|
| day 0 (legacy logs only) | HOLD | ESS 79 < 100 — the ladder never tried the planner's delays |
| days 7–28 | HOLD | ESS climbing 72 → 99 as HOLD-mode exploration logs accumulate |
| day 35 | HOLD | ESS clears the floor; interval still includes zero |
| **day 42** | **SWITCH** | **Δ +$2.04/invoice, 95% interval [+$0.39, +$3.79], ESS 240 on 1,503 logged first retries** |

Exploration cost nothing measurable: invoices decided under HOLD realised
$10.04 ± $1.24 net each against the ladder's exact expectation of $9.82 on the
same invoices. This is the README's rollout claim — *you cannot validate from
old logs; explore cautiously for a month, then the evidence exists* — run
rather than asserted. It is still a simulation.

## The operator console

`recoup-ops serve` runs a FastAPI read model over the jobs' files
(`recoup/api/`) and serves a React console (`web/`) at `/`:

- **Overview** — gate verdict with its interval and ESS against the floor,
  the gate's history, where first retries landed under each logging mode, and
  decisions per day by policy.
- **Decisions** — the audit log with every propensity, filterable.
- **Invoices** — status, retry timeline, and a "why this delay?" inspector
  that re-runs the planner: its best schedule, the value of each first
  bucket, and how often exploration departs from it.
- **Model** — holdout metrics, coefficients with standard errors, versions.
- **Simulation** (synthetic data only) — advance the clock and score the live
  policy against the ladder's oracle value.

```bash
pip install -e .[api]
recoup-ops demo --fresh                     # or: recoup-ops synth && recoup-ops retrain
cd web && npm ci && npm run build && cd ..
recoup-ops serve --workdir var/demo         # http://127.0.0.1:8080/
```

The Pydantic models in `recoup/api/schemas.py` are the contract; the
frontend's TypeScript types are generated from them (`npm run gen:api`) and a
test fails if the two drift. `web/README.md` has the workflow.

## Install and run

```bash
pip install -e .
python benchmark.py            # full evaluation incl. perturbed worlds
python quickstart.py           # what a single decision looks like
recoup-ops demo --fresh        # the operations loop on synthetic data (see above)
recoup-ops serve --workdir var/demo   # the console, after `npm run build` in web/
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
  bachs.py       API adapter -- SEE THE WARNING IN THAT FILE; validate_schema()
  synthetic.py   the simulator rendered as a raw Bachs export; advance_world()
  store.py       dataset loading as of a time, model bundles, audit log, queue
  ops.py         the recoup-ops CLI: synth / retrain / plan / gate / advance / score / demo / serve
  api/           console API: schemas.py (the contract), service.py (read models), app.py (routes)
web/             the console: React + TS + Vite, types generated from the API (web/README.md)
benchmark.py     end-to-end comparison incl. misspecified worlds and volume sweep
quickstart.py    single-invoice decision surface
deploy/          systemd units + operator console (deploy/SERVER.md)
tests/           invariants that would silently cost money if they broke
                 (test_ops.py: the data path, the jobs and their guards)
```

## Honest limitations

- **Cure labels are off by default, because they make things worse.** With
  renewals in the payment stream, "this customer came back" labels are
  abundant (5,266 on the default dataset) and "this customer left" labels are
  rare (25). Against a true churn share of 0.090 among retried invoices, the
  unlabelled model says 0.187, the labelled one 0.008. Labelling is
  outcome-dependent and the likelihood has no selection term for it, so the
  labels collapse π and the planner would keep retrying customers who are
  gone. The fix is a positive-unlabelled correction in `CureHazardModel`, not
  a threshold; `recoup-ops retrain --cure-labels` stays available for that
  work.
- **The gate judges the first retry, and most of the gain is later.** On the
  benchmark the planner beats the ladder by $0.42 per invoice on the first
  decision (+6%) and by $3.81 on the whole schedule (+31%). The gate sees
  only the first number, with intervals about ±$2.3 wide, so on legacy logs
  it is conservative by construction. Whole-schedule off-policy evaluation from the
  per-attempt propensities `plan` already logs is the next step.
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
