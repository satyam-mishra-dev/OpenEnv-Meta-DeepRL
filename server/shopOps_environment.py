# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
ShopOps Environment Implementation.

A realistic environment simulating daily e-commerce support operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from uuid import uuid4
import random

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from ..models import (
    ActionType,
    CaseType,
    CaseView,
    CustomerTier,
    DeliveryStatus,
    EscalationReason,
    FraudSignal,
    IssueSeverity,
    ItemCategory,
    Resources,
    ShopopsAction,
    ShopopsObservation,
)

ENV_SCHEMA_VERSION = "1.0.0"

MAX_CASES = 20
TIME_LIMIT_MINUTES = 480
BUDGET_LIMIT_USD = 2000.0
INVALID_LIMIT = 3

REWARD_WEIGHTS = {
    "correctness": 0.6,
    "cost_efficiency": 0.25,
    "prioritization": 0.15,
}


@dataclass(frozen=True)
class ActionCostRule:
    time_minutes: int
    base_cost_usd: float


ACTION_COSTS: Dict[ActionType, ActionCostRule] = {
    ActionType.REFUND: ActionCostRule(time_minutes=2, base_cost_usd=0.0),
    ActionType.REPLACE: ActionCostRule(time_minutes=5, base_cost_usd=0.0),
    ActionType.ESCALATE: ActionCostRule(time_minutes=10, base_cost_usd=0.0),
    ActionType.REJECT: ActionCostRule(time_minutes=1, base_cost_usd=0.0),
}


@dataclass
class CaseInternal:
    case_id: str
    case_type: CaseType
    customer_tier: CustomerTier
    order_value_usd: float
    days_since_order: int
    delivery_status: Optional[DeliveryStatus]
    issue_severity: IssueSeverity
    fraud_score: float
    item_category: ItemCategory
    return_window_open: bool
    evidence_provided: bool
    prior_refund_count: int
    hidden_fields: Set[str] = field(default_factory=set)
    adversarial: bool = False


class ShopopsEnvironment(Environment[ShopopsAction, ShopopsObservation, State]):
    """
    ShopOps environment.

    Each episode is a fixed 20-case queue representing a workday.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, debug_mode: bool = False):
        super().__init__()
        self._debug_mode = debug_mode
        self._rng = random.Random()
        self._cases: List[CaseInternal] = []
        self._case_index = 0
        self._time_used = 0
        self._budget_used = 0.0
        self._invalid_count = 0
        self._cumulative_score = 0.0
        self._case_results: List[Tuple[CaseType, bool]] = []
        self._adversarial_case_ids: Set[str] = set()
        self._tier = "easy"
        self._split = "train"
        self._expose_expected_action = False
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        tier: Optional[str] = None,
        split: Optional[str] = None,
        debug_mode: Optional[bool] = None,
        expose_expected_action: Optional[bool] = None,
        **kwargs: object,
    ) -> ShopopsObservation:
        self._rng = random.Random(seed)
        self._tier = tier if tier in {"easy", "medium", "hard"} else "easy"
        self._split = split if split in {"train", "test"} else "train"
        if debug_mode is not None:
            self._debug_mode = debug_mode
        if expose_expected_action is not None:
            self._expose_expected_action = expose_expected_action

        self._cases = self._generate_cases()
        self._case_index = 0
        self._time_used = 0
        self._budget_used = 0.0
        self._invalid_count = 0
        self._cumulative_score = 0.0
        self._case_results = []
        self._adversarial_case_ids = {case.case_id for case in self._cases if case.adversarial}
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        info = {"reset": True}
        if self._cases:
            info["is_adversarial_case"] = self._cases[0].adversarial
        return self._build_observation(reward=None, done=False, info=info)

    def step(
        self,
        action: ShopopsAction,
        timeout_s: Optional[float] = None,
        **kwargs: object,
    ) -> ShopopsObservation:
        if self._is_done():
            return self._build_observation(reward=0.0, done=True, info={"already_done": True})

        case = self._cases[self._case_index]
        validation_error = self._validate_action(action, case)
        if validation_error:
            self._invalid_count += 1
            self._time_used += 1
            self._state.step_count += 1
            done = self._invalid_count >= INVALID_LIMIT
            reward = -1.0
            self._cumulative_score += reward
            info = {
                "validation_error": validation_error,
                "invalid_count": self._invalid_count,
                "reward_breakdown": {
                    "correctness": 0.0,
                    "cost_efficiency": 0.0,
                    "prioritization": 0.0,
                },
                "correct": False,
                "case_id": case.case_id,
                "case_type": case.case_type.value,
                "budget_used": round(self._budget_used, 2),
                "budget_remaining": round(max(self._budget_remaining(), 0.0), 2),
                "time_used": self._time_used,
                "time_remaining": max(self._time_remaining(), 0),
                "cumulative_score": round(self._cumulative_score, 4),
            }
            if done:
                info["termination_reason"] = "invalid_action_limit"
            return self._build_observation(reward=reward, done=done, info=info)

        expected = self._expected_action(case)
        correctness = self._score_correctness(action, expected)

        actual_cost, actual_time = self._compute_costs(action, case)
        expected_cost, _ = self._compute_costs(expected, case)

        self._time_used += actual_time
        self._budget_used += actual_cost

        cost_efficiency = self._score_cost_efficiency(actual_cost, expected_cost)
        effective_cost_efficiency = cost_efficiency * correctness
        prioritization = self._score_prioritization(case, correctness)

        reward = (
            REWARD_WEIGHTS["correctness"] * correctness
            + REWARD_WEIGHTS["cost_efficiency"] * effective_cost_efficiency
            + REWARD_WEIGHTS["prioritization"] * prioritization
        )

        if self._budget_remaining() < 0 or self._time_remaining() < 0:
            reward -= 0.5
        if self._tier == "hard" and correctness < 1.0:
            reward -= 0.1

        self._cumulative_score += reward
        self._state.step_count += 1
        self._case_results.append((case.case_type, correctness >= 0.9))
        self._case_index += 1

        done = self._is_done()
        info = {
            "reward_breakdown": {
                "correctness": correctness,
                "cost_efficiency": effective_cost_efficiency,
                "prioritization": prioritization,
            },
            "correct": correctness >= 0.9,
            "case_id": case.case_id,
            "case_type": case.case_type.value,
            "budget_used": round(self._budget_used, 2),
            "budget_remaining": round(max(self._budget_remaining(), 0.0), 2),
            "time_used": self._time_used,
            "time_remaining": max(self._time_remaining(), 0),
            "cumulative_score": round(self._cumulative_score, 4),
            "invalid_count": self._invalid_count,
            "is_adversarial_case": case.adversarial,
        }

        if self._debug_mode and self._expose_expected_action:
            info["expected_action"] = self._serialize_action(expected)

        if done:
            info["termination_reason"] = self._termination_reason()
            info["episode_summary"] = self._episode_summary()

        return self._build_observation(reward=reward, done=done, info=info)

    @property
    def state(self) -> State:
        return self._state

    def _generate_cases(self) -> List[CaseInternal]:
        cases: List[CaseInternal] = []
        for idx in range(MAX_CASES):
            case_type = self._sample_case_type()
            order_value = round(self._rng.uniform(20, 800), 2)
            days_since_order = self._rng.randint(1, 45)
            delivery_status = self._sample_delivery_status(case_type)
            issue_severity = self._sample_issue_severity()
            fraud_score = self._rng.random()
            customer_tier = self._sample_customer_tier()
            item_category = self._sample_item_category()
            prior_refund_count = self._rng.randint(0, 4)
            return_window_open = days_since_order <= 30
            evidence_provided = self._rng.random() < 0.7

            hidden_fields = self._select_hidden_fields(case_type, delivery_status)

            cases.append(
                CaseInternal(
                    case_id=f"case-{idx+1:03d}",
                    case_type=case_type,
                    customer_tier=customer_tier,
                    order_value_usd=order_value,
                    days_since_order=days_since_order,
                    delivery_status=delivery_status,
                    issue_severity=issue_severity,
                    fraud_score=fraud_score,
                    item_category=item_category,
                    return_window_open=return_window_open,
                    evidence_provided=evidence_provided,
                    prior_refund_count=prior_refund_count,
                    hidden_fields=hidden_fields,
                )
            )
        if self._tier == "hard" and self._split == "test":
            cases = self._inject_adversarial_cases(cases)
        return cases

    def _select_hidden_fields(
        self, case_type: CaseType, delivery_status: Optional[DeliveryStatus]
    ) -> Set[str]:
        if self._tier != "hard":
            return set()
        candidates = [
            "order_value_bucket",
            "order_age_bucket",
            "prior_refund_count_bucket",
        ]
        if case_type == CaseType.DELIVERY_ISSUE and delivery_status is not None:
            candidates.append("delivery_status")
        hidden = set()
        if candidates:
            hidden.add(self._rng.choice(candidates))
        return hidden

    def _inject_adversarial_cases(self, cases: List[CaseInternal]) -> List[CaseInternal]:
        adversarial = self._adversarial_templates()
        for idx, template in enumerate(adversarial):
            if idx >= len(cases):
                break
            cases[idx] = template
        return cases

    def _adversarial_templates(self) -> List[CaseInternal]:
        templates: List[CaseInternal] = []
        templates.append(
            CaseInternal(
                case_id="adv-001",
                case_type=CaseType.FRAUD_SIGNAL,
                customer_tier=CustomerTier.GOLD,
                order_value_usd=850.0,
                days_since_order=8,
                delivery_status=None,
                issue_severity=IssueSeverity.MEDIUM,
                fraud_score=0.25,
                item_category=ItemCategory.ELECTRONICS,
                return_window_open=True,
                evidence_provided=False,
                prior_refund_count=1,
                hidden_fields={"order_value_bucket"},
                adversarial=True,
            )
        )
        templates.append(
            CaseInternal(
                case_id="adv-002",
                case_type=CaseType.DELIVERY_ISSUE,
                customer_tier=CustomerTier.PLATINUM,
                order_value_usd=420.0,
                days_since_order=32,
                delivery_status=DeliveryStatus.DELAYED,
                issue_severity=IssueSeverity.HIGH,
                fraud_score=0.15,
                item_category=ItemCategory.HOME,
                return_window_open=False,
                evidence_provided=True,
                prior_refund_count=0,
                hidden_fields={"delivery_status"},
                adversarial=True,
            )
        )
        templates.append(
            CaseInternal(
                case_id="adv-003",
                case_type=CaseType.REFUND_REQUEST,
                customer_tier=CustomerTier.SILVER,
                order_value_usd=120.0,
                days_since_order=29,
                delivery_status=None,
                issue_severity=IssueSeverity.LOW,
                fraud_score=0.78,
                item_category=ItemCategory.BEAUTY,
                return_window_open=True,
                evidence_provided=False,
                prior_refund_count=3,
                hidden_fields={"prior_refund_count_bucket"},
                adversarial=True,
            )
        )
        return templates

    def _sample_case_type(self) -> CaseType:
        weights = {
            "easy": [0.3, 0.3, 0.3, 0.1],
            "medium": [0.25, 0.3, 0.25, 0.2],
            "hard": [0.2, 0.25, 0.25, 0.3],
        }[self._tier]
        return self._rng.choices(
            [
                CaseType.REFUND_REQUEST,
                CaseType.DELIVERY_ISSUE,
                CaseType.WRONG_ITEM,
                CaseType.FRAUD_SIGNAL,
            ],
            weights=weights,
        )[0]

    def _sample_delivery_status(self, case_type: CaseType) -> Optional[DeliveryStatus]:
        if case_type != CaseType.DELIVERY_ISSUE:
            return None
        return self._rng.choices(
            [
                DeliveryStatus.DELIVERED,
                DeliveryStatus.IN_TRANSIT,
                DeliveryStatus.DELAYED,
                DeliveryStatus.LOST,
            ],
            weights=[0.2, 0.3, 0.35, 0.15],
        )[0]

    def _sample_issue_severity(self) -> IssueSeverity:
        weights = {
            "easy": [0.5, 0.35, 0.15],
            "medium": [0.35, 0.4, 0.25],
            "hard": [0.25, 0.4, 0.35],
        }[self._tier]
        return self._rng.choices(
            [IssueSeverity.LOW, IssueSeverity.MEDIUM, IssueSeverity.HIGH],
            weights=weights,
        )[0]

    def _sample_customer_tier(self) -> CustomerTier:
        return self._rng.choices(
            [CustomerTier.BRONZE, CustomerTier.SILVER, CustomerTier.GOLD, CustomerTier.PLATINUM],
            weights=[0.45, 0.3, 0.2, 0.05],
        )[0]

    def _sample_item_category(self) -> ItemCategory:
        return self._rng.choice(
            [
                ItemCategory.APPAREL,
                ItemCategory.ELECTRONICS,
                ItemCategory.HOME,
                ItemCategory.BEAUTY,
                ItemCategory.GROCERY,
            ]
        )

    def _build_observation(
        self, reward: Optional[float], done: bool, info: Dict[str, object]
    ) -> ShopopsObservation:
        case = self._cases[self._case_index] if self._case_index < len(self._cases) else None
        case_view = self._case_to_view(case) if case else self._empty_case_view()
        resources = Resources(
            time_remaining_minutes=max(self._time_remaining(), 0),
            budget_remaining_usd=round(max(self._budget_remaining(), 0.0), 2),
            time_used_minutes=self._time_used,
            budget_used_usd=round(self._budget_used, 2),
        )
        return ShopopsObservation(
            case=case_view,
            resources=resources,
            case_index=self._case_index,
            step_index=self._state.step_count,
            episode_id=self._state.episode_id or "",
            tier=self._tier,
            env_schema_version=ENV_SCHEMA_VERSION,
            reward=reward,
            done=done,
            metadata=info,
        )

    def _case_to_view(self, case: CaseInternal) -> CaseView:
        fraud_signal = self._fraud_signal(case.fraud_score)
        order_value_bucket = self._bucket_order_value(case.order_value_usd)
        order_age_bucket = self._bucket_order_age(case.days_since_order)
        prior_refund_bucket = self._bucket_prior_refunds(case.prior_refund_count)

        show_exact = self._tier == "easy"
        show_partial = self._tier == "medium"

        if "order_value_bucket" in case.hidden_fields:
            order_value_bucket = None
        if "order_age_bucket" in case.hidden_fields:
            order_age_bucket = None
        if "delivery_status" in case.hidden_fields:
            delivery_status = None
        else:
            delivery_status = (
                case.delivery_status
                if case.case_type == CaseType.DELIVERY_ISSUE
                else None
            )
        if "return_window_open" in case.hidden_fields:
            return_window_open = None
        else:
            return_window_open = case.return_window_open if show_exact else None
        if "evidence_provided" in case.hidden_fields:
            evidence_provided = None
        else:
            evidence_provided = case.evidence_provided if (show_exact or show_partial) else None
        if "prior_refund_count_bucket" in case.hidden_fields:
            prior_refund_bucket = None

        return CaseView(
            case_id=case.case_id,
            case_type=case.case_type,
            customer_tier=case.customer_tier,
            issue_severity=case.issue_severity,
            fraud_signal=fraud_signal,
            item_category=case.item_category,
            order_value_usd=case.order_value_usd if (show_exact or show_partial) else None,
            order_value_bucket=None if (show_exact or show_partial) else order_value_bucket,
            days_since_order=case.days_since_order if (show_exact or show_partial) else None,
            order_age_bucket=None if (show_exact or show_partial) else order_age_bucket,
            delivery_status=delivery_status,
            return_window_open=return_window_open,
            evidence_provided=evidence_provided,
            prior_refund_count_bucket=prior_refund_bucket if not show_exact else None,
        )

    def _empty_case_view(self) -> CaseView:
        return CaseView(
            case_id="",
            case_type=CaseType.REFUND_REQUEST,
            customer_tier=CustomerTier.BRONZE,
            issue_severity=IssueSeverity.LOW,
            fraud_signal=FraudSignal.LOW,
            item_category=ItemCategory.APPAREL,
        )

    def _validate_action(self, action: ShopopsAction, case: CaseInternal) -> Optional[str]:
        if action.action_type == ActionType.REFUND:
            if action.refund_amount_usd is None:
                return "refund_amount_required"
            if action.refund_amount_usd <= 0:
                return "refund_amount_must_be_positive"
            if action.refund_amount_usd > case.order_value_usd:
                return "refund_exceeds_order_value"
            if action.replacement_expedite:
                return "expedite_not_allowed_for_refund"
            if action.escalation_reason is not None:
                return "escalation_reason_not_allowed_for_refund"
        elif action.action_type == ActionType.REPLACE:
            if action.refund_amount_usd is not None:
                return "refund_amount_not_allowed_for_replace"
            if action.escalation_reason is not None:
                return "escalation_reason_not_allowed_for_replace"
        elif action.action_type == ActionType.ESCALATE:
            if action.escalation_reason is None:
                return "escalation_reason_required"
            if action.refund_amount_usd is not None:
                return "refund_amount_not_allowed_for_escalate"
            if action.replacement_expedite:
                return "expedite_not_allowed_for_escalate"
        elif action.action_type == ActionType.REJECT:
            if action.refund_amount_usd is not None:
                return "refund_amount_not_allowed_for_reject"
            if action.replacement_expedite:
                return "expedite_not_allowed_for_reject"
            if action.escalation_reason is not None:
                return "escalation_reason_not_allowed_for_reject"
        else:
            return "unsupported_action_type"
        return None

    def _expected_action(self, case: CaseInternal) -> ShopopsAction:
        priority = self._case_priority(case)
        high_value = case.order_value_usd >= 500

        if case.case_type == CaseType.FRAUD_SIGNAL:
            if case.fraud_score >= 0.7:
                return ShopopsAction(action_type=ActionType.ESCALATE, escalation_reason=EscalationReason.SUSPECTED_FRAUD)
            if case.fraud_score >= 0.4:
                return ShopopsAction(action_type=ActionType.REJECT)
            return ShopopsAction(action_type=ActionType.REFUND, refund_amount_usd=round(case.order_value_usd * 0.5, 2))

        if case.case_type == CaseType.REFUND_REQUEST:
            if not case.return_window_open:
                return ShopopsAction(action_type=ActionType.REJECT)
            if case.fraud_score >= 0.7 or case.prior_refund_count >= 3:
                return ShopopsAction(action_type=ActionType.ESCALATE, escalation_reason=EscalationReason.SUSPECTED_FRAUD)
            if high_value:
                return ShopopsAction(action_type=ActionType.ESCALATE, escalation_reason=EscalationReason.HIGH_VALUE)
            return ShopopsAction(action_type=ActionType.REFUND, refund_amount_usd=case.order_value_usd)

        if case.case_type == CaseType.DELIVERY_ISSUE:
            if case.delivery_status == DeliveryStatus.LOST:
                return ShopopsAction(action_type=ActionType.REPLACE, replacement_expedite=priority >= 0.8)
            if case.delivery_status == DeliveryStatus.DELAYED:
                refund_amount = round(case.order_value_usd * 0.2, 2)
                return ShopopsAction(action_type=ActionType.REFUND, refund_amount_usd=refund_amount)
            if case.delivery_status == DeliveryStatus.IN_TRANSIT:
                return ShopopsAction(action_type=ActionType.ESCALATE, escalation_reason=EscalationReason.POLICY_EXCEPTION)
            return ShopopsAction(action_type=ActionType.REJECT)

        if case.case_type == CaseType.WRONG_ITEM:
            if case.evidence_provided:
                return ShopopsAction(action_type=ActionType.REPLACE, replacement_expedite=priority >= 0.8)
            if case.customer_tier in {CustomerTier.GOLD, CustomerTier.PLATINUM} and case.prior_refund_count < 2:
                return ShopopsAction(action_type=ActionType.REPLACE)
            return ShopopsAction(action_type=ActionType.REJECT)

        return ShopopsAction(action_type=ActionType.REJECT)

    def _score_correctness(self, action: ShopopsAction, expected: ShopopsAction) -> float:
        if action.action_type != expected.action_type:
            return 0.0

        if action.action_type == ActionType.REFUND:
            if action.refund_amount_usd is None or expected.refund_amount_usd is None:
                return 0.5
            if abs(action.refund_amount_usd - expected.refund_amount_usd) <= 1.0:
                return 1.0
            return 0.5

        if action.action_type == ActionType.REPLACE:
            return 1.0 if action.replacement_expedite == expected.replacement_expedite else 0.5

        if action.action_type == ActionType.ESCALATE:
            return 1.0 if action.escalation_reason == expected.escalation_reason else 0.5

        return 1.0

    def _score_cost_efficiency(self, actual_cost: float, expected_cost: float) -> float:
        if expected_cost <= 0:
            return 1.0 if actual_cost <= 0 else max(0.0, 1.0 - actual_cost / 100.0)
        if actual_cost <= expected_cost:
            penalty = 0.2 * (expected_cost - actual_cost) / expected_cost
            return max(0.0, 1.0 - penalty)
        overage = (actual_cost - expected_cost) / expected_cost
        return max(0.0, 1.0 - overage)

    def _score_prioritization(self, case: CaseInternal, correctness: float) -> float:
        if correctness <= 0:
            return 0.0
        return self._case_priority(case)

    def _case_priority(self, case: CaseInternal) -> float:
        tier_weight = {
            CustomerTier.BRONZE: 0.1,
            CustomerTier.SILVER: 0.2,
            CustomerTier.GOLD: 0.4,
            CustomerTier.PLATINUM: 0.6,
        }[case.customer_tier]
        severity_weight = {
            IssueSeverity.LOW: 0.1,
            IssueSeverity.MEDIUM: 0.3,
            IssueSeverity.HIGH: 0.6,
        }[case.issue_severity]
        value_weight = min(case.order_value_usd / 1000.0, 0.6)
        fraud_weight = 0.2 if case.fraud_score >= 0.7 else 0.0
        return min(1.0, tier_weight + severity_weight + value_weight + fraud_weight)

    def _compute_costs(self, action: ShopopsAction, case: CaseInternal) -> Tuple[float, int]:
        rule = ACTION_COSTS[action.action_type]
        time_minutes = rule.time_minutes
        cost = rule.base_cost_usd

        if action.action_type == ActionType.REFUND:
            cost += float(action.refund_amount_usd or 0.0)
        elif action.action_type == ActionType.REPLACE:
            cost += min(case.order_value_usd * 0.6, 300.0)
            if action.replacement_expedite:
                time_minutes += 2
                cost += 20.0
        return cost, time_minutes

    def _fraud_signal(self, score: float) -> FraudSignal:
        if score < 0.3:
            return FraudSignal.LOW
        if score < 0.7:
            return FraudSignal.MEDIUM
        return FraudSignal.HIGH

    def _bucket_order_value(self, value: float) -> str:
        if value < 50:
            return "low"
        if value < 200:
            return "medium"
        if value < 500:
            return "high"
        return "very_high"

    def _bucket_order_age(self, days: int) -> str:
        if days <= 7:
            return "recent"
        if days <= 30:
            return "normal"
        return "old"

    def _bucket_prior_refunds(self, count: int) -> str:
        if count == 0:
            return "none"
        if count <= 2:
            return "few"
        return "many"

    def _budget_remaining(self) -> float:
        return BUDGET_LIMIT_USD - self._budget_used

    def _time_remaining(self) -> int:
        return TIME_LIMIT_MINUTES - self._time_used

    def _is_done(self) -> bool:
        return (
            self._case_index >= MAX_CASES
            or self._time_remaining() <= 0
            or self._budget_remaining() <= 0
            or self._invalid_count >= INVALID_LIMIT
        )

    def _termination_reason(self) -> str:
        if self._invalid_count >= INVALID_LIMIT:
            return "invalid_action_limit"
        if self._budget_remaining() <= 0:
            return "budget_exhausted"
        if self._time_remaining() <= 0:
            return "time_exhausted"
        if self._case_index >= MAX_CASES:
            return "cases_completed"
        return "unknown"

    def _episode_summary(self) -> Dict[str, object]:
        success_by_type: Dict[str, int] = {}
        total_by_type: Dict[str, int] = {}
        for case_type, correct in self._case_results:
            key = case_type.value
            total_by_type[key] = total_by_type.get(key, 0) + 1
            if correct:
                success_by_type[key] = success_by_type.get(key, 0) + 1

        success_rate_by_type = {}
        for case_type, total in total_by_type.items():
            successes = success_by_type.get(case_type, 0)
            success_rate_by_type[case_type] = round(successes / total, 4) if total else 0.0

        return {
            "final_score": round(self._cumulative_score, 4),
            "cases_completed": self._case_index,
            "invalid_actions": self._invalid_count,
            "time_used": self._time_used,
            "budget_used": round(self._budget_used, 2),
            "tier": self._tier,
            "success_rate_by_case_type": success_rate_by_type,
            "adversarial_case_count": len(self._adversarial_case_ids),
            "adversarial_case_ids": sorted(self._adversarial_case_ids),
        }

    def _serialize_action(self, action: ShopopsAction) -> Dict[str, object]:
        return {
            "action_type": action.action_type.value,
            "refund_amount_usd": action.refund_amount_usd,
            "replacement_expedite": action.replacement_expedite,
            "escalation_reason": action.escalation_reason.value if action.escalation_reason else None,
        }
