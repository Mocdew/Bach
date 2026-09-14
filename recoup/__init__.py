"""recoup -- failed-payment recovery timing.

Cure-hazard model -> exact finite-horizon schedule planner -> Thompson-sampled
exploration -> cross-fitted off-policy evaluation and a deployment gate.
"""

from .domain import (
    DEFAULT_EPOCH, HARD_DECLINES, MARKETS, N_DT_BUCKETS, NO_RETRY_ADVICE,
    DeclineReason, NetworkAdvice, NetworkRules, Rail, dt_bucket, is_retryable,
)
from .evaluate import (
    cross_fitted_plan, deployment_gate, expected_calibration_error, ladder_schedules,
    logged_first_attempts, off_policy_value, oracle_policy_value,
    oracle_schedule_value, realised_reward, score_model, temporal_split,
)
from .features import build_features, customer_history
from .models import BetaBinomialHazard, CureHazardModel, GBMHazard
from .policy import (
    Decision, Plan, PolicyConfig, RetryPlanner, RetryPolicy, SupportMap,
    TableDisputeModel, candidate_grid, fixed_ladder_policy, month_end_policy,
)
from .simulator import PERTURBED_WORLDS, SimConfig, Truth, simulate

__all__ = [
    "DEFAULT_EPOCH", "HARD_DECLINES", "MARKETS", "N_DT_BUCKETS", "NO_RETRY_ADVICE",
    "DeclineReason", "NetworkAdvice", "NetworkRules", "Rail", "dt_bucket", "is_retryable",
    "cross_fitted_plan", "deployment_gate", "expected_calibration_error", "ladder_schedules",
    "logged_first_attempts", "off_policy_value", "oracle_policy_value",
    "oracle_schedule_value", "realised_reward", "score_model", "temporal_split",
    "build_features", "customer_history",
    "BetaBinomialHazard", "CureHazardModel", "GBMHazard",
    "Decision", "Plan", "PolicyConfig", "RetryPlanner", "RetryPolicy", "SupportMap",
    "TableDisputeModel", "candidate_grid", "fixed_ladder_policy", "month_end_policy",
    "PERTURBED_WORLDS", "SimConfig", "Truth", "simulate",
]
