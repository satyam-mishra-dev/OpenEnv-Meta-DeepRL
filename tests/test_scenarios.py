# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import List
import logging

from shopOps.server.shopOps_environment import (
    BUDGET_LIMIT_USD,
    CaseInternal,
    ShopopsEnvironment,
)
from shopOps.models import (
    ActionType,
    CaseType,
    CustomerTier,
    DeliveryStatus,
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


def _log_obs(scenario: str, obs) -> None:
    logger.info(
        "%s reward=%.4f done=%s meta=%s",
        scenario,
        float(obs.reward or 0.0),
        obs.done,
        obs.metadata,
    )


def _init_env_with_cases(cases: List[CaseInternal]) -> ShopopsEnvironment:
    env = ShopopsEnvironment(debug_mode=True)
    env.reset(seed=1, tier="easy")
    env._cases = cases  # test-only override
    env._case_index = 0
    env._time_used = 0
    env._budget_used = 0.0
    env._invalid_count = 0
    env._cumulative_score = 0.0
    env._case_results = []
    return env


def test_scenario_1_correct_classification_easy():
    case = _make_case(
        case_type=CaseType.WRONG_ITEM,
        evidence_provided=True,
        customer_tier=CustomerTier.SILVER,
    )
    env = _init_env_with_cases([case])

    obs = env.step(ShopopsAction(action_type=ActionType.REPLACE))
    _log_obs("scenario_1", obs)
    assert obs.reward is not None
    assert obs.reward > 0.7
    assert obs.metadata["correct"] is True


def test_scenario_2_wrong_action_penalty():
    case = _make_case(
        case_type=CaseType.DELIVERY_ISSUE,
        delivery_status=DeliveryStatus.DELAYED,
        order_value=120.0,
    )
    env = _init_env_with_cases([case])

    # Incorrect action: refund full amount instead of 20% refund
    obs = env.step(ShopopsAction(action_type=ActionType.REFUND, refund_amount_usd=120.0))
    _log_obs("scenario_2", obs)
    assert obs.reward is not None
    assert 0.1 <= obs.reward <= 0.5
    assert obs.metadata["correct"] is False


def test_scenario_3_vip_priority_penalty_via_cost():
    case = _make_case(
        case_type=CaseType.WRONG_ITEM,
        evidence_provided=True,
        customer_tier=CustomerTier.PLATINUM,
        issue_severity=IssueSeverity.HIGH,
        order_value=600.0,
    )
    env = _init_env_with_cases([case])

    # Replacement with expedite should be preferred; omit expedite to lower correctness
    obs = env.step(ShopopsAction(action_type=ActionType.REPLACE, replacement_expedite=False))
    _log_obs("scenario_3", obs)
    assert obs.reward is not None
    assert obs.reward < 1.0


def test_scenario_4_budget_constraint_failure():
    case = _make_case(
        case_type=CaseType.REFUND_REQUEST,
        order_value=1500.0,
    )
    env = _init_env_with_cases([case])
    env._budget_used = BUDGET_LIMIT_USD - 200.0

    obs = env.step(ShopopsAction(action_type=ActionType.REFUND, refund_amount_usd=1500.0))
    _log_obs("scenario_4", obs)
    assert obs.done is True
    assert obs.metadata["termination_reason"] == "budget_exhausted"


def test_scenario_5_sequential_quality_penalty():
    cases = [
        _make_case(
            case_type=CaseType.REFUND_REQUEST,
            order_value=700.0,
            customer_tier=CustomerTier.PLATINUM,
        ),
        _make_case(
            case_type=CaseType.DELIVERY_ISSUE,
            delivery_status=DeliveryStatus.DELAYED,
            order_value=80.0,
            customer_tier=CustomerTier.BRONZE,
        ),
        _make_case(
            case_type=CaseType.FRAUD_SIGNAL,
            order_value=300.0,
            fraud_score=0.9,
        ),
    ]
    env = _init_env_with_cases(cases)

    # Poor handling: reject VIP refund and ignore fraud escalation
    obs1 = env.step(ShopopsAction(action_type=ActionType.REJECT))
    obs2 = env.step(ShopopsAction(action_type=ActionType.REFUND, refund_amount_usd=80.0))
    obs3 = env.step(ShopopsAction(action_type=ActionType.REJECT))

    _log_obs("scenario_5_step1", obs1)
    _log_obs("scenario_5_step2", obs2)
    _log_obs("scenario_5_step3", obs3)

    assert (obs1.reward or 0.0) < 0.6
    assert (obs3.reward or 0.0) < 0.6


def test_scenario_6_perfect_episode():
    env = ShopopsEnvironment(debug_mode=True)
    obs = env.reset(seed=7, tier="easy")
    total_reward = 0.0

    while True:
        case = env._cases[env._case_index]
        action = env._expected_action(case)
        obs = env.step(action)
        total_reward += float(obs.reward or 0.0)
        _log_obs("scenario_6_step", obs)
        if obs.done:
            summary = obs.metadata.get("episode_summary", {})
            assert summary.get("final_score", 0.0) > 8.0
            assert total_reward > 8.0
            break
