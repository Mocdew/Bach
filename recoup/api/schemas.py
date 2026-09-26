"""The API contract.

Every response the console reads is one of these models. FastAPI turns them
into ``openapi.json``; the frontend generates its TypeScript types from that
file (``npm run gen:api`` in ``web/``), so a renamed field here is a type
error there rather than a blank tile in production.

Conventions: times are ISO-8601 UTC strings *and* ``*_h`` floats (hours since
the dataset epoch) where the frontend needs to do arithmetic; money is USD;
probabilities are in [0, 1].
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel as _PydanticBase, ConfigDict, Field

Decision = Literal["HOLD", "SWITCH"]


class BaseModel(_PydanticBase):
    # A field the server always sends must be required in the *response*
    # schema, even when it has a default -- otherwise the generated TypeScript
    # marks it optional and the frontend litters `?? {}` everywhere.
    model_config = ConfigDict(json_schema_serialization_defaults_required=True)


class Paths(BaseModel):
    data: str
    models: str
    gate: str
    audit: str
    queue: str


class Health(BaseModel):
    ok: bool
    version: str
    synthetic: bool
    data_as_of: Optional[str] = None
    has_model: bool
    has_gate: bool
    paths: Paths


# --- gate ---------------------------------------------------------------------


class GateStatus(BaseModel):
    decision: Decision
    reason: str
    on_file: bool = Field(description="False when no gate has run yet (treated as HOLD)")
    generated_at: Optional[str] = None
    data_as_of: Optional[str] = None
    window_start: Optional[str] = None
    model_version: Optional[str] = None
    previous: Optional[str] = None
    delta: Optional[float] = Field(None, description="DR(planner) - DR(ladder), USD/invoice")
    lo: Optional[float] = None
    hi: Optional[float] = None
    ess: Optional[float] = None
    min_ess: Optional[float] = None
    dr_planner: Optional[float] = None
    dr_incumbent: Optional[float] = None
    n_logged: Optional[int] = None
    logged_by_mode: dict[str, int] = Field(default_factory=dict)


class GateHistoryPoint(BaseModel):
    generated_at: str
    data_as_of: Optional[str] = None
    decision: Decision
    delta: Optional[float] = None
    lo: Optional[float] = None
    hi: Optional[float] = None
    ess: Optional[float] = None
    n_logged: Optional[int] = None


# --- model --------------------------------------------------------------------


class Holdout(BaseModel):
    pr_auc: float
    roc_auc: float
    log_loss: float
    ece: float
    base_rate: float
    n_test_attempts: int


class ModelSummary(BaseModel):
    version: str
    trained_at: str
    data_as_of: str
    n_invoices: int
    n_attempts: int
    n_merchants: int
    success_rate: float
    mean_p_gone: float
    converged: bool
    holdout: Optional[Holdout] = None
    cure_labels: dict = Field(default_factory=dict)
    fit_warnings: list[str] = Field(default_factory=list)
    dispute_base: dict[str, float] = Field(default_factory=dict)
    ladder_h: list[float] = Field(default_factory=list)
    unmapped_codes: dict[str, int] = Field(default_factory=dict)


class ModelVersion(BaseModel):
    version: str
    current: bool
    trained_at: Optional[str] = None
    data_as_of: Optional[str] = None
    pr_auc: Optional[float] = None
    mean_p_gone: Optional[float] = None


class Coefficient(BaseModel):
    part: Literal["hazard", "cure"]
    term: str
    coef: float
    se: float


# --- overview -----------------------------------------------------------------


class Overview(BaseModel):
    data_as_of: str
    synthetic: bool
    gate: GateStatus
    model: Optional[ModelSummary] = None
    open_invoices: int
    awaiting_decision: int
    queued_retries: int = Field(description="retries decided and not yet due")
    decisions_24h: int
    explore_share_7d: Optional[float] = Field(
        None, description="share of retries decided in the last 7 days that were exploration")
    invoices_30d: int
    recovered_30d: float = Field(description="recovery rate of invoices whose window closed in the last 30 days")


# --- decisions ----------------------------------------------------------------


class DecisionRow(BaseModel):
    decision_id: str
    decided_at: str
    decided_at_h: float
    invoice_id: str
    attempt_index: int
    action: str
    delay_hours: Optional[float] = None
    execute_at_h: Optional[float] = None
    dt_bucket: Optional[int] = None
    propensity: Optional[float] = None
    mode: str
    policy: str
    model_version: Optional[str] = None
    p_gone: Optional[float] = None
    p_success: Optional[float] = None
    expected_value_usd: Optional[float] = None
    rationale: Optional[str] = None


class DecisionPage(BaseModel):
    total: int
    offset: int
    limit: int
    rows: list[DecisionRow]
    facets: dict[str, dict[str, int]] = Field(
        description="counts per value of mode / policy / action over the filtered set")


class DailyCount(BaseModel):
    day: str
    policy: str
    n: int


class BucketInfo(BaseModel):
    bucket: int
    label: str
    lo_h: float
    hi_h: float


class Coverage(BaseModel):
    buckets: list[BucketInfo]
    ladder_bucket: Optional[int] = None
    by_mode: dict[str, list[float]] = Field(
        description="share of first-retry decisions per bucket, per logging mode")
    n_by_mode: dict[str, int]


# --- invoices -----------------------------------------------------------------


class InvoiceRow(BaseModel):
    invoice_id: str
    merchant_id: str
    customer_id: str
    market: str
    rail: str
    decline_reason: str
    reason_class: str
    network_advice: str
    amount_usd: float
    fail_time_h: float
    failed_at: str
    elapsed_h: float
    attempts: int
    recovered: bool
    status: Literal["open", "recovered", "routed", "closed"]
    next_action: Optional[str] = None
    next_execute_at_h: Optional[float] = None
    next_policy: Optional[str] = None


class InvoicePage(BaseModel):
    total: int
    offset: int
    limit: int
    rows: list[InvoiceRow]


class AttemptRow(BaseModel):
    attempt_index: int
    attempted_at: str
    delay_hours: float
    dt_bucket: int
    success: bool
    disputed: bool


class CurvePoint(BaseModel):
    bucket: int
    label: str
    delay_h: float
    value_usd: Optional[float] = Field(None, description="schedule value if this bucket goes first")
    propensity: float
    viable: bool


class Explanation(BaseModel):
    """The planner's *best* plan for this invoice (no exploration), plus the
    Thompson distribution it samples the logged action from. The logged
    decision can differ from ``delay_hours`` -- that is exploration."""
    as_of: Literal["now", "at_failure"]
    attempt_index: int
    action: str
    rationale: str
    p_gone: float
    p_success: float = Field(description="P(success) of the best schedule's first attempt")
    expected_value_usd: float = Field(description="value of the best schedule")
    delay_hours: Optional[float] = Field(None, description="best first delay, hours since failure")
    best_bucket: Optional[int] = None
    schedule_h: list[float]
    explore_mass: float = Field(0.0, description="probability the logged action is NOT the best bucket")
    curve: list[CurvePoint]
    model_version: str


class InvoiceDetail(BaseModel):
    invoice: InvoiceRow
    customer_tenure_days: Optional[float] = None
    prior_successful_payments: int
    active_in_window: bool
    plan_tier: str
    attempts: list[AttemptRow]
    decisions: list[DecisionRow]
    explanation: Optional[Explanation] = None
    explanation_error: Optional[str] = None


# --- simulation ---------------------------------------------------------------


class SimProgress(BaseModel):
    done: int
    total: int


class StepResult(BaseModel):
    as_of: str
    steps: int
    decisions: int
    released_failures: int
    retries: int
    recovered: int
    stale: int
    seconds: float


class SimStatus(BaseModel):
    synthetic: bool
    as_of: Optional[str] = None
    as_of_h: Optional[float] = None
    cutover: Optional[str] = None
    cutover_h: Optional[float] = None
    horizon_h: Optional[float] = None
    pending_events: int = 0
    busy: bool = False
    progress: Optional[SimProgress] = None
    last_step: Optional[StepResult] = None
    last_error: Optional[str] = None


class StepRequest(BaseModel):
    hours: float = Field(24.0, gt=0, le=24 * 14)
    plan: bool = Field(True, description="run the plan job before moving the clock")
    explore_rate: float = Field(0.2, ge=0, le=1)


class PhaseScore(BaseModel):
    n: int
    realised: float
    se: float
    ladder_oracle: float
    lift: Optional[float] = None


class Score(BaseModel):
    n_invoices: int
    note: Optional[str] = None
    window: list[str] = Field(default_factory=list)
    realised_per_invoice: Optional[float] = None
    realised_se: Optional[float] = None
    ladder_oracle_per_invoice: Optional[float] = None
    lift_vs_ladder: Optional[float] = None
    recovery_rate: Optional[float] = None
    attempts_per_invoice: Optional[float] = None
    by_mode: dict[str, PhaseScore] = Field(default_factory=dict)


class ApiError(BaseModel):
    detail: str
