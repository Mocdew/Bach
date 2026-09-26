# Deploying recoup under systemd

recoup is a library, not a daemon. It has three batch jobs, each of which maps
onto one layer of the package, and this directory turns them into timers.

```
                 nightly 03:15            hourly                weekly Sun 05:00
                 ───────────────          ─────────────         ────────────────
Bachs API ──►  recoup-retrain  ──►  models/current  ──►  recoup-plan  ──►  retry queue
               (CureHazardModel,                         (RetryPlanner,     + audit log
                TableDisputeModel,                        PolicyConfig,     (decisions +
                SupportMap)                               Thompson)         propensities)
                                                              ▲  │
                                            gate/decision.json│  ▼
                                                        recoup-gate
                                                        (off_policy_value,
                                                         deployment_gate)
```

| unit | layer | what it does | cadence |
|---|---|---|---|
| `recoup-retrain` | 1 estimate | pull history via `recoup.bachs.from_payments`, fit, atomic-swap `models/current` | nightly |
| `recoup-plan` | 2 plan | plan schedules for open failed invoices, log propensities, push to retry queue | hourly |
| `recoup-gate` | 3 learn safely | cross-fitted OPE of planner vs incumbent ladder; write HOLD/SWITCH | weekly |
| `recoup-model-present` | — | guard so `plan` can `Requires=` a model on a fresh install | on demand |
| `recoup-dashboard` | UI | build `dashboard.json` for the operator console (`../dashboard/`) | nightly |
| `recoup-web` | UI | serve the static console (`dashboard.json`) on port 8080, LAN only | always |
| `recoup-console` | UI | the live console — API + React UI over the jobs' state — on port 8081 | always |
| `recoup.slice` | — | CPU/memory cap for all of the above | — |
| `recoup.target` | — | one enable/disable handle | — |

Design decisions:

- **Batch, not a request/response service.** `RetryPlanner` is vectorised
  over an invoice DataFrame and the candidate grid is hours-to-days, so an
  hourly batch loses nothing versus a socket-activated HTTP service and avoids
  writing (and sandboxing) a server that doesn't exist yet. If per-invoice
  latency ever matters, add `recoup-decide.socket` + `Type=notify` service and
  keep everything else as is.
- **The gate is a file, not a flag in the plan job.** `plan` always runs and
  always logs propensities; `gate/decision.json` only controls whether it
  emits the planner's schedule or the incumbent ladder's. That is the
  "switch on the safety rules and cautious exploration first, then judge"
  rollout from the README, expressed as two units.
- **`Conflicts=` between retrain and plan** rather than a lock file: systemd
  serialises them and the model symlink swap is atomic anyway.
- **Secrets via `LoadCredentialEncrypted`**, not `Environment=`. The Bachs key
  is decrypted into `$CREDENTIALS_DIRECTORY` for the process lifetime only.
- **`DynamicUser=yes` + `StateDirectory=`**: no `recoup` user to create, state
  ownership handled by systemd, and `ProtectSystem=strict` makes everything
  outside `/var/lib/recoup`, `/var/log/recoup`, `/run/recoup` read-only.

## `recoup-ops`

The units call `/opt/recoup/venv/bin/recoup-ops {retrain,plan,gate}`, the
`[project.scripts]` entry point in `recoup/ops.py`. Paths default from the
environment `recoup-common.conf` sets (`RECOUP_DATA`, `RECOUP_STATE`,
`RECOUP_LOG`), which is why the `ExecStart=` lines stay short.

1. `retrain` reads `$RECOUP_DATA` through `recoup.bachs.from_payments()`,
   fits `CureHazardModel` (with cure labels when both sides clear a floor),
   `TableDisputeModel` and `SupportMap` on invoices whose dunning window has
   closed, checks a temporal holdout, and promotes `models/<ts>/` by
   atomically repointing `models/current`. A failed check exits 2 and leaves
   the previous model live.
2. `plan` decides the next retry of every open invoice, writes decisions +
   propensities to `$RECOUP_LOG/decisions/decisions-<run>.csv` (append-only)
   and the actions to `$RECOUP_STATE/queue/<run>.jsonl`. Under HOLD it runs
   the incumbent ladder with an ε share (`--explore-rate`, default 0.2) of
   planner-chosen retries and logs the mixture propensity; under SWITCH it
   runs the planner. It is idempotent: an (invoice, attempt) already in the
   audit log is never decided twice.
3. `gate` joins first-retry decisions in the last `--eval-days` to their
   outcomes, runs cross-fitted DR of the planner against the ladder, and
   writes `gate/decision.json`. SWITCH is sticky: it reverts only if the
   interval says the planner is *worse*.

**What is still missing** is the fetcher that writes `$RECOUP_DATA` from the
live Bachs API, and the executor that turns `queue/*.jsonl` into Bachs retry
calls. Both need `bachs.FIELD_MAP` pinned against the OpenAPI spec first.
Until then `recoup-ops synth` writes a synthetic dataset in the export layout
and `recoup-ops advance` plays the executor against the simulator's ground
truth, so the whole loop runs:

```bash
recoup-ops synth --out /var/lib/recoup/data
recoup-ops retrain && recoup-ops plan && recoup-ops advance --hours 24 && recoup-ops gate
```

See `../SERVER.md` for the step-by-step on a home Linux laptop.

## Install

```bash
sudo install -Dm644 recoup.slice recoup.target recoup-*.service recoup-*.timer -t /etc/systemd/system/
for u in retrain plan gate model-present; do
  sudo install -Dm644 recoup-common.conf /etc/systemd/system/recoup-$u.service.d/00-common.conf
done
# once a Bachs fetcher exists (and the LoadCredentialEncrypted= line is uncommented):
# sudo install -dm700 /etc/recoup
# printf '%s' "$BACHS_API_KEY" | sudo systemd-creds encrypt --name=bachs_api_key - /etc/recoup/bachs_api_key.cred
sudo systemctl daemon-reload
sudo systemctl enable --now recoup.target
```

Check: `systemctl list-timers 'recoup-*'`, `journalctl -u recoup-plan -f`,
`systemd-analyze security recoup-plan.service` (should score well under 2.0).
