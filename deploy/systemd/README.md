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
| `recoup-web` | UI | serve the console on port 8080, LAN only | always |
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

## What does not exist yet

The units call `/opt/recoup/venv/bin/recoup-ops {retrain,plan,gate}`. There is
no such entry point in the package — `quickstart.py` and `benchmark.py` run on
the simulator. `recoup-ops` needs to:

1. `retrain`: fetch payments/disputes from Bachs (the `FIELD_MAP` in
   `recoup/bachs.py` must be pinned against the OpenAPI spec first — it is a
   sketch), `from_payments()` → `customer_history()` → `build_features()` →
   fit `CureHazardModel`, `TableDisputeModel`, `SupportMap`; pickle to
   `models/<ts>/` and `os.replace` the `current` symlink.
2. `plan`: load bundle, `RetryPolicy.decide()` over open failed invoices,
   write decisions + propensities to `decisions/<ts>.parquet`, push the
   chosen action to whatever executes retries (Bachs subscription retry
   endpoint or the merchant's queue).
3. `gate`: read the decision log, `off_policy_value()` + `deployment_gate()`,
   write `gate/decision.json`.

Add it as a `[project.scripts]` entry in `pyproject.toml`.

See `../SERVER.md` for the step-by-step on a home Linux laptop, including
which units to enable before `recoup-ops` exists.

## Install

```bash
sudo install -Dm644 recoup.slice recoup.target recoup-*.service recoup-*.timer -t /etc/systemd/system/
for u in retrain plan gate model-present; do
  sudo install -Dm644 recoup-common.conf /etc/systemd/system/recoup-$u.service.d/00-common.conf
done
sudo install -dm700 /etc/recoup
printf '%s' "$BACHS_API_KEY" | sudo systemd-creds encrypt --name=bachs_api_key - /etc/recoup/bachs_api_key.cred
sudo systemctl daemon-reload
sudo systemctl enable --now recoup.target
```

Check: `systemctl list-timers 'recoup-*'`, `journalctl -u recoup-plan -f`,
`systemd-analyze security recoup-plan.service` (should score well under 2.0).
