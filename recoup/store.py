"""On-disk state for the operations layer.

Three kinds of state, each with one rule:

* **Datasets** are read-only exports (see ``synthetic`` for the layout).
  ``load_dataset`` reads one *as of* a point in time, so every job -- and
  every test -- can replay history without peeking past its own clock.
* **Model bundles** are immutable directories under ``models/``. A bundle is
  promoted by atomically repointing ``models/current``; a job that is
  reading the old bundle keeps reading a complete one.
* **The decision audit log** is append-only: one CSV per plan run, never
  rewritten. It is the only place propensities exist, and the gate's
  off-policy estimate is only as honest as this file is complete.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .bachs import BachsTables, from_payments, normalise_payments
from .domain import DEFAULT_EPOCH, NetworkRules
from .models import CureHazardModel
from .policy import PolicyConfig, SupportMap, TableDisputeModel

log = logging.getLogger("recoup")

BUNDLE_FORMAT = 1


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


@dataclass
class Dataset:
    path: Path
    tables: BachsTables
    customers: pd.DataFrame
    as_of_h: float
    manifest: dict
    legacy_decisions: pd.DataFrame | None = None

    @property
    def invoices(self) -> pd.DataFrame:
        return self.tables.invoices

    @property
    def attempts(self) -> pd.DataFrame:
        return self.tables.attempts


def _hours(ts: pd.Series, epoch) -> np.ndarray:
    t = pd.to_datetime(ts, utc=True).dt.tz_localize(None)
    return ((t - pd.Timestamp(epoch)) / pd.Timedelta(hours=1)).to_numpy(dtype=float)


def _read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path)
    return pd.DataFrame(columns=columns)


def hours_now(epoch=DEFAULT_EPOCH) -> float:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    return (now - epoch).total_seconds() / 3600.0


def load_dataset(path: str | os.PathLike, as_of_h: float | None = None) -> Dataset:
    """Read an export directory as it looked at ``as_of_h``.

    ``as_of_h`` defaults to the manifest's ``as_of_h`` (a synthetic dataset
    carries its own clock) and otherwise to the wall clock. Anything stamped
    after it -- payments, disputes, customer events -- is invisible.
    """
    root = Path(path)
    if not (root / "payments.jsonl").exists():
        raise FileNotFoundError(f"{root / 'payments.jsonl'} not found -- is --data a dataset "
                                "directory? (`recoup-ops synth` makes one)")
    manifest = {}
    if (root / "manifest.json").exists():
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    epoch_iso = manifest.get("epoch", DEFAULT_EPOCH.strftime("%Y-%m-%dT%H:%M:%SZ"))
    epoch = pd.Timestamp(epoch_iso).tz_localize(None).to_pydatetime()
    if as_of_h is None:
        as_of_h = float(manifest["as_of_h"]) if "as_of_h" in manifest else hours_now(epoch)

    with open(root / "payments.jsonl", encoding="utf-8") as f:
        objs = [json.loads(line) for line in f if line.strip()]
    pay, report = normalise_payments(objs, epoch_iso=epoch_iso,
                                     fx_to_usd=manifest.get("fx_to_usd"))
    if len(pay):
        pay = pay[pay["t_h"] <= as_of_h].reset_index(drop=True)

    customers = _read_csv(root / "customers.csv",
                          ["customer_id", "merchant_id", "market", "created_at", "plan_tier"])
    disputes = _read_csv(root / "disputes.csv", ["payment_id", "created_at"])
    if len(disputes):
        disputes = disputes[_hours(disputes["created_at"], epoch) <= as_of_h]
    events = _read_csv(root / "customer_events.csv", ["customer_id", "type", "created_at"])
    if len(events):
        events = events[_hours(events["created_at"], epoch) <= as_of_h]

    tables = from_payments(pay, epoch_iso=epoch_iso, customers=customers, disputes=disputes,
                           customer_events=events)
    report.n_payments = len(pay)
    report.n_failed = int(pay["status"].isin({"failed", "declined"}).sum()) if len(pay) else 0
    report.n_customers_without_profile = tables.report.n_customers_without_profile
    tables.report = report

    legacy = None
    if (root / "legacy_decisions.csv").exists():
        legacy = pd.read_csv(root / "legacy_decisions.csv")
        legacy = legacy[legacy["decided_at_h"] <= as_of_h]
    return Dataset(root, tables, customers, float(as_of_h), manifest, legacy)


# ---------------------------------------------------------------------------
# Model bundles
# ---------------------------------------------------------------------------


@dataclass
class ModelBundle:
    model: CureHazardModel
    dispute: TableDisputeModel
    support: SupportMap
    policy_cfg: PolicyConfig
    rules: NetworkRules
    ladder_h: tuple[float, ...]
    meta: dict = field(default_factory=dict)
    format: int = BUNDLE_FORMAT

    @property
    def version(self) -> str:
        return str(self.meta.get("version", "unknown"))


def _pointer_target(models_dir: Path) -> str | None:
    cur = models_dir / "current"
    if cur.is_symlink():
        return os.path.basename(os.readlink(cur))
    if cur.is_file():
        return cur.read_text(encoding="utf-8").strip() or None
    return None


def save_bundle(bundle: ModelBundle, models_dir: str | os.PathLike, *, keep: int = 5) -> Path:
    """Write ``models/<version>/`` and atomically repoint ``models/current``.

    Uses a symlink where the OS allows one (Linux; the systemd units expect
    it) and a one-line pointer file where it does not (Windows without
    developer mode). ``resolve_bundle`` reads either. Keeps the newest
    ``keep`` bundles plus whatever ``current`` points at.
    """
    root = Path(models_dir)
    root.mkdir(parents=True, exist_ok=True)
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    n = 0
    while (root / (version if n == 0 else f"{version}-{n}")).exists():
        n += 1
    version = version if n == 0 else f"{version}-{n}"
    bundle.meta["version"] = version

    staging = root / f".staging-{version}"
    staging.mkdir()
    with open(staging / "bundle.pkl", "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    (staging / "meta.json").write_text(json.dumps(bundle.meta, indent=1, default=str),
                                       encoding="utf-8")
    bundle.model.coef_table().to_csv(staging / "coefficients.csv", index=False)
    os.replace(staging, root / version)

    tmp = root / f".current-{version}"
    try:
        os.symlink(version, tmp, target_is_directory=True)
    except (OSError, NotImplementedError):
        tmp.write_text(version + "\n", encoding="utf-8")
    os.replace(tmp, root / "current")

    versions = sorted(p.name for p in root.iterdir()
                      if p.is_dir() and not p.is_symlink() and not p.name.startswith("."))
    for old in versions[:-keep] if keep > 0 else []:
        if old != version:
            shutil.rmtree(root / old, ignore_errors=True)
    return root / version


def resolve_bundle(path: str | os.PathLike) -> Path:
    """Accept a bundle dir, ``models/current`` (symlink or pointer), or ``models/``."""
    p = Path(path)
    if (p / "bundle.pkl").exists():
        return p
    if p.name == "current" and p.is_file() and not p.is_symlink():
        return p.parent / p.read_text(encoding="utf-8").strip()
    if p.is_dir() and (p / "current").exists():
        target = _pointer_target(p)
        if target:
            return p / target
    raise FileNotFoundError(f"no model bundle at {p} -- run `recoup-ops retrain` first")


def load_bundle(path: str | os.PathLike) -> ModelBundle:
    # Pickle executes code on load. Bundles are only ever read from this
    # service's own state directory, which no other user can write to
    # (StateDirectory= with DynamicUser=, UMask=0077).
    d = resolve_bundle(path)
    with open(d / "bundle.pkl", "rb") as f:
        bundle = pickle.load(f)
    if getattr(bundle, "format", None) != BUNDLE_FORMAT:
        raise ValueError(f"bundle {d} has format {getattr(bundle, 'format', None)}, "
                         f"expected {BUNDLE_FORMAT}; retrain")
    return bundle


# ---------------------------------------------------------------------------
# Decision audit log
# ---------------------------------------------------------------------------

AUDIT_COLUMNS = [
    "decision_id", "run_id", "decided_at_h", "decided_at", "invoice_id", "attempt_index",
    "action", "delay_hours", "execute_at_h", "dt_bucket", "propensity", "propensities",
    "mode", "policy", "model_version", "p_gone", "p_success", "expected_value_usd",
    "rationale",
]


def write_audit(audit_dir: str | os.PathLike, run_id: str, rows: list[dict]) -> Path | None:
    if not rows:
        return None
    d = Path(audit_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"decisions-{run_id}.csv"
    tmp = path.with_suffix(".csv.tmp")
    pd.DataFrame(rows, columns=AUDIT_COLUMNS).to_csv(tmp, index=False)
    os.replace(tmp, path)
    return path


def read_audit(audit_dir: str | os.PathLike | None, dataset: Dataset | None = None
               ) -> pd.DataFrame:
    """Every logged decision: the plan runs' files plus the dataset's legacy log."""
    frames = []
    if dataset is not None and dataset.legacy_decisions is not None:
        frames.append(dataset.legacy_decisions)
    if audit_dir is not None and Path(audit_dir).exists():
        for f in sorted(Path(audit_dir).glob("decisions-*.csv")):
            if f.stat().st_size:
                frames.append(pd.read_csv(f))
    if not frames:
        return pd.DataFrame(columns=AUDIT_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    out["invoice_id"] = out["invoice_id"].astype(str)
    out["attempt_index"] = out["attempt_index"].astype(int)
    return out


def write_queue(queue_dir: str | os.PathLike, run_id: str, items: list[dict]) -> Path | None:
    if not items:
        return None
    d = Path(queue_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{run_id}.jsonl"
    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for it in items:
            f.write(json.dumps(it, separators=(",", ":"), default=float) + "\n")
    os.replace(tmp, path)
    return path


def atomic_write_json(path: str | os.PathLike, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")
    os.replace(tmp, p)
