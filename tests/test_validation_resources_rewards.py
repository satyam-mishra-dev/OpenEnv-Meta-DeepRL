# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging

from shopOps.server.shopOps_environment import (
    ACTION_COSTS,
    BUDGET_LIMIT_USD,
    CaseInternal,
    ShopopsEnvironment,
    TIME_LIMIT_MINUTES,
)
from shopOps.models import (
    ActionType,
    CaseType,
    CustomerTier,
    DeliveryStatus,
    EscalationReason,
    IssueSeverity,
    ItemCategory,
    ShopopsAction,
)

logger = logging.getLogger(__name__)


def _make_case(
    case_type: CaseType,
    order_value: float = 120.0,
    days_since_order: int = 10,
    delivery_status: DeliveryStatus | None = None,
    issue_severity: IssueSeverity = IssueSeverity.MEDIUM,
    fraud_score: float = 0.1,
    customer_tier: CustomerTier = CustomerTier.BRONZE,
    evidence_provided: bool = True,
    prior_refund_count: int = 0,
) -> CaseInternal:
    return CaseInternal(
        case_id="case-001",
        case_type=case_type,
        customer_tier=customer_tier,
        order_value_usd=order_value,
        days_since_order=days_since_order,
        delivery_status=delivery_status,
        issue_severity=issue_severity,
        fraud_score=fraud_score,
        item_category=ItemCategory.APPAREL,
        return_window_open=days_since_order <= 30,
        evidence_provided=evidence_provided,
        prior_refund_count=prior_refund_count,
    )


def _init_env_with_case(case: CaseInternal) -> ShopopsEnvironment:
    env = ShopopsEnvironment(debug_mode=True)
    env.reset(seed=1, tier="easy", split="train")
    env._cases = [case]
    env._case_index = 0
    env._time_used = 0
    env._budget_used = 0.0
    env._invalid_count = 0
    env._cumulative_score = 0.0
    env._case_results = []
    return env


def test_validation_missing_refund_amount():
    case = _make_case(case_type=CaseType.REFUND_REQUEST, order_value=100.0)
    env = _init_env_with_case(case)

    obs = env.step(ShopopsAction(action_type=ActionType.REFUND))
    logger.info("validation_missing_refund=%s", obs.metadata)

    assert obs.metadata.get("validation_error") == "refund_amount_required"
    assert obs.metadata.get("invalid_count") == 1


def test_validation_escalate_requires_reason():
    case = _make_case(case_type=CaseType.FRAUD_SIGNAL, fraud_score=0.9)
    env = _init_env_with_case(case)

    obs = env.step(ShopopsAction(action_type=ActionType.ESCALATE))
    logger.info("validation_missing_reason=%s", obs.metadata)

    assert obs.metadata.get("validation_error") == "escalation_reason_required"


def test_validation_refund_exceeds_order_value():
    case = _make_case(case_type=CaseType.REFUND_REQUEST, order_value=50.0)
    env = _init_env_with_case(case)

    obs = env.step(ShopopsAction(action_type=ActionType.REFUND, refund_amount_usd=200.0))
    logger.info("validation_exceeds_value=%s", obs.metadata)

    assert obs.metadata.get("validation_error") == "refund_exceeds_order_value"


def test_validation_three_invalid_terminates():
    case = _make_case(case_type=CaseType.REFUND_REQUEST, order_value=50.0)
    env = _init_env_with_case(case)

    for _ in range(3):
        obs = env.step(ShopopsAction(action_type=ActionType.REFUND))

    logger.info("validation_three_invalid=%s", obs.metadata)
    assert obs.done is True
    assert obs.metadata.get("termination_reason") == "invalid_action_limit"


def test_resource_budget_updates():
    case = _make_case(case_type=CaseType.REFUND_REQUEST, order_value=120.0)
    env = _init_env_with_case(case)

    obs = env.step(ShopopsAction(action_type=ActionType.REFUND, refund_amount_usd=120.0))
    logger.info("resource_budget=%s", obs.metadata)

    assert env._budget_used == 120.0
    assert obs.resources.budget_remaining_usd == round(BUDGET_LIMIT_USD - 120.0, 2)


def test_resource_time_updates():
    case = _make_case(case_type=CaseType.DELIVERY_ISSUE, delivery_status=DeliveryStatus.LOST)
    env = _init_env_with_case(case)

    obs = env.step(ShopopsAction(action_type=ActionType.REPLACE, replacement_expedite=True))
    logger.info("resource_time=%s", obs.metadata)

    expected_time = ACTION_COSTS[ActionType.REPLACE].time_minutes + 2
    assert env._time_used == expected_time
    assert obs.resources.time_remaining_minutes == TIME_LIMIT_MINUTES - expected_time


def test_budget_exhaustion_termination():
    case = _make_case(case_type=CaseType.REFUND_REQUEST, order_value=1500.0)
    env = _init_env_with_case(case)
    env._budget_used = BUDGET_LIMIT_USD - 200.0

    obs = env.step(ShopopsAction(action_type=ActionType.REFUND, refund_amount_usd=1500.0))
    logger.info("budget_exhaust=%s", obs.metadata)

    assert obs.done is True
    assert obs.metadata.get("termination_reason") == "budget_exhausted"


def test_time_exhaustion_termination():
    case = _make_case(case_type=CaseType.DELIVERY_ISSUE, delivery_status=DeliveryStatus.IN_TRANSIT)
    env = _init_env_with_case(case)
    env._time_used = TIME_LIMIT_MINUTES - 1

    obs = env.step(
        ShopopsAction(action_type=ActionType.ESCALATE, escalation_reason=EscalationReason.POLICY_EXCEPTION)
    )
    logger.info("time_exhaust=%s", obs.metadata)

    assert obs.done is True
    assert obs.metadata.get("termination_reason") == "time_exhausted"


def test_reward_formula_consistency():
    case = _make_case(case_type=CaseType.DELIVERY_ISSUE, delivery_status=DeliveryStatus.DELAYED)
    env = _init_env_with_case(case)

    obs = env.step(ShopopsAction(action_type=ActionType.REFUND, refund_amount_usd=24.0))
    breakdown = obs.metadata["reward_breakdown"]
    expected = (
        0.6 * breakdown["correctness"]
        + 0.25 * breakdown["cost_efficiency"]
        + 0.15 * breakdown["prioritization"]
    )
    logger.info("reward_breakdown=%s", breakdown)
    assert abs((obs.reward or 0.0) - expected) < 1e-6
