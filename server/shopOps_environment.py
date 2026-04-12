from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        ActionType,
        BusinessMetrics,
        CarrierStatus,
        CasePriority,
        CaseStatus,
        CaseType,
        CaseView,
        CustomerTier,
        EscalationReason,
        EvidenceStatus,
        FraudSignal,
        OrderStatus,
        QueueItemView,
        Resources,
        ShopopsAction,
        ShopopsObservation,
        ToolResult,
    )
except ImportError:
    from models import (
        ActionType,
        BusinessMetrics,
        CarrierStatus,
        CasePriority,
        CaseStatus,
        CaseType,
        CaseView,
        CustomerTier,
        EscalationReason,
        EvidenceStatus,
        FraudSignal,
        OrderStatus,
        QueueItemView,
        Resources,
        ShopopsAction,
        ShopopsObservation,
        ToolResult,
    )


ENV_SCHEMA_VERSION = "2.0.0"
INVALID_LIMIT = 4
SCORE_MIN = 1e-9
SCORE_MAX = 1.0 - 1e-9
TASK_ALIASES = {
    "easy": "refund_policy_recovery",
    "medium": "sla_queue_juggle",
    "hard": "fraud_stockout_cascade",
}

ACTION_TIME_COSTS = {
    ActionType.INSPECT_ORDER: 4,
    ActionType.INSPECT_POLICY: 4,
    ActionType.INSPECT_INVENTORY: 3,
    ActionType.INSPECT_CUSTOMER_HISTORY: 4,
    ActionType.REQUEST_EVIDENCE: 3,
    ActionType.CONTACT_CARRIER: 6,
    ActionType.ISSUE_REFUND: 5,
    ActionType.SHIP_REPLACEMENT: 8,
    ActionType.ESCALATE_RISK: 5,
    ActionType.ADD_INTERNAL_NOTE: 2,
    ActionType.CLOSE_CASE: 2,
    ActionType.SWITCH_CASE: 1,
}

PRIORITY_SCORES = {
    CasePriority.LOW: 0.25,
    CasePriority.MEDIUM: 0.5,
    CasePriority.HIGH: 0.75,
    CasePriority.CRITICAL: 1.0,
}


@dataclass
class PendingEvent:
    case_id: str
    event_type: str
    ready_step: int
    summary: str
    outcome: str


@dataclass
class CaseInternal:
    case_id: str
    title: str
    case_type: CaseType
    customer_tier: CustomerTier
    priority: CasePriority
    order_value_usd: float
    days_since_order: int
    order_status: OrderStatus
    fraud_signal: FraudSignal
    requested_compensation_usd: Optional[float] = None
    replacement_sku: Optional[str] = None
    status: CaseStatus = CaseStatus.OPEN
    evidence_status: EvidenceStatus = EvidenceStatus.NOT_REQUESTED
    carrier_status: CarrierStatus = CarrierStatus.NOT_CONTACTED
    order_summary: Optional[str] = None
    policy_summary: Optional[str] = None
    history_summary: Optional[str] = None
    inventory_summary: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    completed_checks: Set[str] = field(default_factory=set)
    pending_events: List[str] = field(default_factory=list)
    required_checks: Set[str] = field(default_factory=set)
    needs_evidence: bool = False
    needs_carrier_contact: bool = False
    requires_note: bool = False
    preferred_resolution: ActionType = ActionType.CLOSE_CASE
    refund_range: Optional[Tuple[float, float]] = None
    preferred_expedite: bool = False
    preferred_escalation_reason: Optional[EscalationReason] = None
    order_details_text: str = ""
    policy_details_text: str = ""
    history_details_text: str = ""
    evidence_response_text: str = ""
    carrier_response_text: str = ""
    evidence_outcome: EvidenceStatus = EvidenceStatus.NOT_REQUESTED
    carrier_outcome: CarrierStatus = CarrierStatus.NOT_CONTACTED
    sla_minutes: int = 120
    fraud_loss_if_bad_close_usd: float = 0.0
    customer_satisfaction_bonus: float = 0.05
    can_reopen: bool = False
    resolution_action: Optional[ActionType] = None
    resolution_amount_usd: Optional[float] = None
    resolution_expedite: bool = False
    resolution_escalation_reason: Optional[EscalationReason] = None
    resolution_summary: Optional[str] = None
    closed_at_step: Optional[int] = None
    reopened_count: int = 0
    sla_breached: bool = False


@dataclass(frozen=True)
class ScenarioConfig:
    task_name: str
    difficulty: str
    time_limit_minutes: int
    budget_limit_usd: float
    step_limit: int
    inventory: Dict[str, int]


@dataclass
class ActionOutcome:
    reward: float
    summary: str
    details: Dict[str, object] = field(default_factory=dict)
    budget_delta: float = 0.0


class ShopopsEnvironment(Environment[ShopopsAction, ShopopsObservation, State]):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    _EPISODE_STORE: Dict[str, Dict[str, object]] = {}

    def __init__(self, debug_mode: bool = False):
        super().__init__()
        self._debug_mode = debug_mode
        self._rng = random.Random()
        self._scenario = self._scenario_configs()["refund_policy_recovery"]
        self._cases: List[CaseInternal] = []
        self._inventory: Dict[str, int] = {}
        self._pending_events: List[PendingEvent] = []
        self._active_case_id: Optional[str] = None
        self._task_name = "refund_policy_recovery"
        self._difficulty = "easy"
        self._time_used = 0
        self._budget_used = 0.0
        self._invalid_count = 0
        self._stockouts = 0
        self._reopened_cases = 0
        self._sla_breaches = 0
        self._fraud_loss_usd = 0.0
        self._customer_satisfaction = 0.7
        self._cumulative_reward = 0.0
        self._episode_terminated = False
        self._terminal_bonus_applied = False
        self._last_action_error: Optional[str] = None
        self._latest_tool_result: Optional[ToolResult] = None
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def _scenario_configs(self) -> Dict[str, ScenarioConfig]:
        return {
            "refund_policy_recovery": ScenarioConfig(
                task_name="refund_policy_recovery",
                difficulty="easy",
                time_limit_minutes=40,
                budget_limit_usd=250.0,
                step_limit=8,
                inventory={},
            ),
            "sla_queue_juggle": ScenarioConfig(
                task_name="sla_queue_juggle",
                difficulty="medium",
                time_limit_minutes=160,
                budget_limit_usd=1200.0,
                step_limit=30,
                inventory={"tablet-purple": 1, "wireless-headset": 2},
            ),
            "fraud_stockout_cascade": ScenarioConfig(
                task_name="fraud_stockout_cascade",
                difficulty="hard",
                time_limit_minutes=220,
                budget_limit_usd=1800.0,
                step_limit=40,
                inventory={"console-pro": 1, "earbuds-lite": 1},
            ),
        }

    def _snapshot_state(self) -> Dict[str, object]:
        return {
            "rng_state": self._rng.getstate(),
            "scenario": copy.deepcopy(self._scenario),
            "cases": copy.deepcopy(self._cases),
            "inventory": copy.deepcopy(self._inventory),
            "pending_events": copy.deepcopy(self._pending_events),
            "active_case_id": self._active_case_id,
            "task_name": self._task_name,
            "difficulty": self._difficulty,
            "time_used": self._time_used,
            "budget_used": self._budget_used,
            "invalid_count": self._invalid_count,
            "stockouts": self._stockouts,
            "reopened_cases": self._reopened_cases,
            "sla_breaches": self._sla_breaches,
            "fraud_loss_usd": self._fraud_loss_usd,
            "customer_satisfaction": self._customer_satisfaction,
            "cumulative_reward": self._cumulative_reward,
            "episode_terminated": self._episode_terminated,
            "terminal_bonus_applied": self._terminal_bonus_applied,
            "last_action_error": self._last_action_error,
            "latest_tool_result": copy.deepcopy(self._latest_tool_result),
            "state": copy.deepcopy(self._state),
        }

    def _load_state(self, snapshot: Dict[str, object]) -> None:
        self._rng.setstate(snapshot["rng_state"])  # type: ignore[arg-type]
        self._scenario = copy.deepcopy(snapshot["scenario"])  # type: ignore[assignment]
        self._cases = copy.deepcopy(snapshot["cases"])  # type: ignore[assignment]
        self._inventory = copy.deepcopy(snapshot["inventory"])  # type: ignore[assignment]
        self._pending_events = copy.deepcopy(snapshot["pending_events"])  # type: ignore[assignment]
        self._active_case_id = snapshot["active_case_id"]  # type: ignore[assignment]
        self._task_name = str(snapshot["task_name"])
        self._difficulty = str(snapshot["difficulty"])
        self._time_used = int(snapshot["time_used"])
        self._budget_used = float(snapshot["budget_used"])
        self._invalid_count = int(snapshot["invalid_count"])
        self._stockouts = int(snapshot["stockouts"])
        self._reopened_cases = int(snapshot["reopened_cases"])
        self._sla_breaches = int(snapshot["sla_breaches"])
        self._fraud_loss_usd = float(snapshot["fraud_loss_usd"])
        self._customer_satisfaction = float(snapshot["customer_satisfaction"])
        self._cumulative_reward = float(snapshot["cumulative_reward"])
        self._episode_terminated = bool(snapshot["episode_terminated"])
        self._terminal_bonus_applied = bool(snapshot["terminal_bonus_applied"])
        self._last_action_error = snapshot["last_action_error"]  # type: ignore[assignment]
        self._latest_tool_result = copy.deepcopy(snapshot["latest_tool_result"])  # type: ignore[assignment]
        self._state = copy.deepcopy(snapshot["state"])  # type: ignore[assignment]

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task: Optional[str] = None,
        tier: Optional[str] = None,
        **_: object,
    ) -> ShopopsObservation:
        self._rng = random.Random(seed)
        task_name = task or TASK_ALIASES.get(tier or "", "refund_policy_recovery")
        scenario = self._scenario_configs().get(task_name)
        if scenario is None:
            raise ValueError(f"Unknown task: {task_name}")

        self._scenario = scenario
        self._task_name = scenario.task_name
        self._difficulty = scenario.difficulty
        self._cases = self._build_cases_for_task(scenario.task_name)
        self._inventory = copy.deepcopy(scenario.inventory)
        self._pending_events = []
        self._active_case_id = self._cases[0].case_id if self._cases else None
        self._time_used = 0
        self._budget_used = 0.0
        self._invalid_count = 0
        self._stockouts = 0
        self._reopened_cases = 0
        self._sla_breaches = 0
        self._fraud_loss_usd = 0.0
        self._customer_satisfaction = 0.72
        self._cumulative_reward = 0.0
        self._episode_terminated = False
        self._terminal_bonus_applied = False
        self._last_action_error = None
        self._latest_tool_result = ToolResult(
            action_type=None,
            target_case_id=self._active_case_id,
            outcome="Episode reset.",
            details={"task": self._task_name, "difficulty": self._difficulty},
        )
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        ShopopsEnvironment._EPISODE_STORE[self._state.episode_id] = self._snapshot_state()
        return self._build_observation(
            reward=None,
            done=False,
            info={"reset": True, "task": self._task_name, "difficulty": self._difficulty},
        )

    def step(
        self,
        action: ShopopsAction,
        timeout_s: Optional[float] = None,
        episode_id: Optional[str] = None,
        **_: object,
    ) -> ShopopsObservation:
        del timeout_s
        if episode_id:
            snapshot = ShopopsEnvironment._EPISODE_STORE.get(episode_id)
            if snapshot is None:
                raise RuntimeError(f"Unknown episode_id: {episode_id}")
            self._load_state(snapshot)
        elif not self._cases:
            raise RuntimeError("Episode not initialized. Call reset() first.")

        if self._is_done():
            return self._build_observation(
                reward=0.0,
                done=True,
                info={"already_done": True, "termination_reason": self._termination_reason()},
            )

        validation_error = self._validate_action(action)
        if validation_error:
            self._last_action_error = validation_error
            self._invalid_count += 1
            self._time_used += 1
            self._state.step_count += 1
            self._advance_events()
            self._update_sla_breaches()
            reward = -0.25
            self._cumulative_reward += reward
            self._latest_tool_result = ToolResult(
                action_type=action.action_type,
                target_case_id=action.case_id or self._active_case_id,
                outcome=f"Invalid action: {validation_error}",
                details={},
            )
            done = self._is_done()
            info = {
                "validation_error": validation_error,
                "invalid_count": self._invalid_count,
                "last_action_error": validation_error,
                "reward_breakdown": {
                    "information_gain": 0.0,
                    "workflow_progress": -0.15,
                    "business_outcome": -0.1,
                },
            }
            if done:
                reward += self._apply_terminal_bonus()
                info["termination_reason"] = self._termination_reason()
                info["episode_summary"] = self._episode_summary()
            obs = self._build_observation(reward=reward, done=done, info=info)
            self._persist_episode(done)
            return obs

        case = self._active_case()
        reward_breakdown = {
            "information_gain": 0.0,
            "workflow_progress": 0.0,
            "business_outcome": 0.0,
        }
        outcome = self._handle_action(action, case)
        reward = outcome.reward
        reward_breakdown.update(outcome.details.pop("reward_breakdown", {}))
        self._budget_used += outcome.budget_delta
        self._time_used += ACTION_TIME_COSTS[action.action_type]
        self._state.step_count += 1
        self._last_action_error = None
        self._latest_tool_result = ToolResult(
            action_type=action.action_type,
            target_case_id=case.case_id if action.action_type != ActionType.SWITCH_CASE else action.case_id,
            outcome=outcome.summary,
            details=outcome.details,
        )
        self._advance_events()
        self._update_sla_breaches()

        done = self._is_done()
        if done:
            reward += self._apply_terminal_bonus()
        self._cumulative_reward += reward
        info = {
            "reward_breakdown": reward_breakdown,
            "invalid_count": self._invalid_count,
            "last_action_error": None,
            "active_case_id": self._active_case_id,
        }
        if done:
            info["termination_reason"] = self._termination_reason()
            info["episode_summary"] = self._episode_summary()
        obs = self._build_observation(reward=reward, done=done, info=info)
        self._persist_episode(done)
        return obs

    @property
    def state(self) -> State:
        return self._state

    def _persist_episode(self, done: bool) -> None:
        ShopopsEnvironment._EPISODE_STORE[self._state.episode_id] = self._snapshot_state()
        if done:
            ShopopsEnvironment._EPISODE_STORE.pop(self._state.episode_id, None)

    def _build_cases_for_task(self, task_name: str) -> List[CaseInternal]:
        order_bump = round(self._rng.uniform(-10.0, 10.0), 2)
        if task_name == "refund_policy_recovery":
            return [
                CaseInternal(
                    case_id="RPR-1",
                    title="Gold customer asks for full refund after late delivery",
                    case_type=CaseType.REFUND_REQUEST,
                    customer_tier=CustomerTier.GOLD,
                    priority=CasePriority.HIGH,
                    order_value_usd=280.0 + order_bump,
                    days_since_order=32,
                    order_status=OrderStatus.DELIVERED,
                    fraud_signal=FraudSignal.LOW,
                    requested_compensation_usd=280.0 + order_bump,
                    required_checks={"order", "policy"},
                    preferred_resolution=ActionType.ISSUE_REFUND,
                    refund_range=(84.0, 98.0),
                    requires_note=True,
                    order_details_text=(
                        "Delivered after the promised date. Return window is closed, but shipping was 6 days late."
                    ),
                    policy_details_text=(
                        "Service recovery policy allows up to 35% refund for severe shipping-delay complaints once delivery is confirmed."
                    ),
                    sla_minutes=35,
                    customer_satisfaction_bonus=0.12,
                )
            ]

        if task_name == "sla_queue_juggle":
            cases = [
                CaseInternal(
                    case_id="SLA-1",
                    title="VIP wrong-item complaint with photo proof",
                    case_type=CaseType.WRONG_ITEM,
                    customer_tier=CustomerTier.PLATINUM,
                    priority=CasePriority.CRITICAL,
                    order_value_usd=420.0 + order_bump,
                    days_since_order=9,
                    order_status=OrderStatus.DELIVERED,
                    fraud_signal=FraudSignal.LOW,
                    replacement_sku="tablet-purple",
                    evidence_status=EvidenceStatus.RECEIVED,
                    required_checks={"order", "inventory"},
                    preferred_resolution=ActionType.SHIP_REPLACEMENT,
                    preferred_expedite=True,
                    requires_note=True,
                    order_details_text="Warehouse shipped the wrong tablet color; photo evidence already attached.",
                    inventory_summary="tablet-purple inventory is limited but available.",
                    sla_minutes=45,
                    customer_satisfaction_bonus=0.13,
                ),
                CaseInternal(
                    case_id="SLA-2",
                    title="Lost shipment for standard headset",
                    case_type=CaseType.DELIVERY_ISSUE,
                    customer_tier=CustomerTier.SILVER,
                    priority=CasePriority.HIGH,
                    order_value_usd=145.0,
                    days_since_order=12,
                    order_status=OrderStatus.LOST,
                    fraud_signal=FraudSignal.LOW,
                    replacement_sku="wireless-headset",
                    required_checks={"order", "inventory"},
                    preferred_resolution=ActionType.SHIP_REPLACEMENT,
                    preferred_expedite=False,
                    order_details_text="Carrier marked the parcel lost after a failed handoff scan.",
                    inventory_summary="wireless-headset inventory is healthy.",
                    sla_minutes=80,
                    customer_satisfaction_bonus=0.08,
                ),
                CaseInternal(
                    case_id="SLA-3",
                    title="Chargeback warning on high-value order",
                    case_type=CaseType.FRAUD_SIGNAL,
                    customer_tier=CustomerTier.GOLD,
                    priority=CasePriority.HIGH,
                    order_value_usd=510.0,
                    days_since_order=2,
                    order_status=OrderStatus.DELIVERED,
                    fraud_signal=FraudSignal.HIGH,
                    required_checks={"order", "history"},
                    preferred_resolution=ActionType.ESCALATE_RISK,
                    preferred_escalation_reason=EscalationReason.SUSPECTED_FRAUD,
                    order_details_text="Device fingerprint mismatch and unusual payment velocity were detected.",
                    history_details_text="Customer has two prior suspicious refund attempts in the last month.",
                    sla_minutes=110,
                    customer_satisfaction_bonus=0.05,
                ),
                CaseInternal(
                    case_id="SLA-4",
                    title="Simple within-window return request",
                    case_type=CaseType.REFUND_REQUEST,
                    customer_tier=CustomerTier.SILVER,
                    priority=CasePriority.LOW,
                    order_value_usd=98.0,
                    days_since_order=11,
                    order_status=OrderStatus.DELIVERED,
                    fraud_signal=FraudSignal.LOW,
                    requested_compensation_usd=98.0,
                    required_checks={"order"},
                    preferred_resolution=ActionType.ISSUE_REFUND,
                    refund_range=(96.0, 100.0),
                    order_details_text="Returned inside policy window with no fraud flags.",
                    sla_minutes=180,
                    customer_satisfaction_bonus=0.06,
                ),
                CaseInternal(
                    case_id="SLA-5",
                    title="Delayed shipment requesting partial refund",
                    case_type=CaseType.DELIVERY_ISSUE,
                    customer_tier=CustomerTier.BRONZE,
                    priority=CasePriority.MEDIUM,
                    order_value_usd=190.0,
                    days_since_order=8,
                    order_status=OrderStatus.DELAYED,
                    fraud_signal=FraudSignal.LOW,
                    required_checks={"order", "policy"},
                    needs_carrier_contact=True,
                    preferred_resolution=ActionType.ISSUE_REFUND,
                    refund_range=(42.0, 58.0),
                    order_details_text="Tracking is stalled and the customer is asking for compensation.",
                    policy_details_text="Delayed-shipment credits are only available after carrier confirmation.",
                    carrier_response_text="Carrier confirmed a service failure and approved a partial shipping refund.",
                    carrier_outcome=CarrierStatus.APPROVED,
                    sla_minutes=140,
                    customer_satisfaction_bonus=0.07,
                ),
            ]
            tail = cases[1:]
            self._rng.shuffle(tail)
            cases = [cases[0], *tail]
            return cases

        console_value = round(760.0 + self._rng.uniform(-35.0, 35.0), 2)
        suspicious_refund_value = round(620.0 + self._rng.uniform(-55.0, 45.0), 2)
        suspicious_partial_refund = round((390.0 + self._rng.uniform(-20.0, 35.0)) * 0.35, 2)
        delayed_credit_value = round(240.0 + self._rng.uniform(-18.0, 22.0), 2)
        delayed_credit_target = round(delayed_credit_value * self._rng.uniform(0.26, 0.34), 2)
        small_refund_value = round(72.0 + self._rng.uniform(-6.0, 12.0), 2)
        earbuds_value = round(155.0 + self._rng.uniform(-12.0, 18.0), 2)
        self._scenario = ScenarioConfig(
            task_name=self._scenario.task_name,
            difficulty=self._scenario.difficulty,
            time_limit_minutes=self._scenario.time_limit_minutes,
            budget_limit_usd=self._scenario.budget_limit_usd,
            step_limit=self._scenario.step_limit,
            inventory={
                "console-pro": 1,
                "earbuds-lite": 1 + self._rng.randint(0, 1),
            },
        )
        cases = [
            CaseInternal(
                case_id="HARD-1",
                title="Scarce console replacement for VIP wrong-item order",
                case_type=CaseType.WRONG_ITEM,
                customer_tier=CustomerTier.PLATINUM,
                priority=CasePriority.CRITICAL,
                order_value_usd=console_value,
                days_since_order=5 + self._rng.randint(0, 2),
                order_status=OrderStatus.DELIVERED,
                fraud_signal=FraudSignal.LOW,
                replacement_sku="console-pro",
                evidence_status=EvidenceStatus.RECEIVED,
                required_checks={"order", "inventory", "history"},
                preferred_resolution=ActionType.SHIP_REPLACEMENT,
                preferred_expedite=True,
                requires_note=True,
                order_details_text="Premium customer received the wrong console bundle; photo proof attached.",
                history_details_text="Customer history is clean and support promised priority make-good.",
                inventory_summary="console-pro inventory has exactly 1 unit available.",
                sla_minutes=48 + self._rng.randint(0, 12),
                customer_satisfaction_bonus=0.15,
            ),
            CaseInternal(
                case_id="HARD-2",
                title="Suspicious high-value refund request",
                case_type=CaseType.REFUND_REQUEST,
                customer_tier=CustomerTier.GOLD,
                priority=CasePriority.HIGH,
                order_value_usd=suspicious_refund_value,
                days_since_order=2 + self._rng.randint(0, 2),
                order_status=OrderStatus.DELIVERED,
                fraud_signal=FraudSignal.HIGH,
                requested_compensation_usd=suspicious_refund_value,
                required_checks={"order", "policy", "history"},
                needs_evidence=True,
                preferred_resolution=ActionType.ESCALATE_RISK,
                preferred_escalation_reason=EscalationReason.SUSPECTED_FRAUD,
                order_details_text="Delivery photo does not match the address on file.",
                policy_details_text="High-value mismatches must not be refunded without evidence and fraud review.",
                history_details_text="Account shows repeated refund disputes and mismatched IP regions.",
                evidence_response_text="Customer uploaded edited screenshots that do not verify non-delivery.",
                evidence_outcome=EvidenceStatus.INSUFFICIENT,
                sla_minutes=82 + self._rng.randint(0, 16),
                fraud_loss_if_bad_close_usd=320.0 + self._rng.uniform(0.0, 70.0),
                can_reopen=True,
                customer_satisfaction_bonus=0.04,
            ),
            CaseInternal(
                case_id="HARD-3",
                title="Second console complaint competing for the last unit",
                case_type=CaseType.WRONG_ITEM,
                customer_tier=CustomerTier.BRONZE,
                priority=CasePriority.MEDIUM,
                order_value_usd=round(suspicious_partial_refund / 0.35, 2),
                days_since_order=9 + self._rng.randint(0, 3),
                order_status=OrderStatus.DELIVERED,
                fraud_signal=FraudSignal.MEDIUM if self._rng.random() < 0.7 else FraudSignal.HIGH,
                replacement_sku="console-pro",
                required_checks={"order", "history"},
                preferred_resolution=ActionType.ISSUE_REFUND,
                refund_range=(max(90.0, suspicious_partial_refund - 18.0), suspicious_partial_refund + 18.0),
                order_details_text="Customer claims accessories are missing, but no photo evidence is attached.",
                history_details_text="Three prior replacements were granted on the same account this quarter.",
                sla_minutes=118 + self._rng.randint(0, 20),
                customer_satisfaction_bonus=0.05,
            ),
            CaseInternal(
                case_id="HARD-4",
                title="Delayed shipment awaiting carrier approval",
                case_type=CaseType.DELIVERY_ISSUE,
                customer_tier=CustomerTier.SILVER,
                priority=CasePriority.MEDIUM,
                order_value_usd=delayed_credit_value,
                days_since_order=6 + self._rng.randint(0, 3),
                order_status=OrderStatus.DELAYED,
                fraud_signal=FraudSignal.LOW,
                required_checks={"order", "policy"},
                needs_carrier_contact=True,
                preferred_resolution=ActionType.ISSUE_REFUND,
                refund_range=(max(35.0, delayed_credit_target - 12.0), delayed_credit_target + 12.0),
                order_details_text="Carrier missed two scans and the customer is asking for a service credit.",
                policy_details_text="Carrier must acknowledge service failure before compensation is issued.",
                carrier_response_text="Carrier accepted liability and approved a partial service refund.",
                carrier_outcome=CarrierStatus.APPROVED,
                sla_minutes=138 + self._rng.randint(0, 20),
                customer_satisfaction_bonus=0.07,
            ),
            CaseInternal(
                case_id="HARD-5",
                title="Fresh fraud alert on accessory order",
                case_type=CaseType.FRAUD_SIGNAL,
                customer_tier=CustomerTier.BRONZE,
                priority=CasePriority.HIGH,
                order_value_usd=210.0,
                days_since_order=1,
                order_status=OrderStatus.DELIVERED,
                fraud_signal=FraudSignal.HIGH,
                required_checks={"order", "history"},
                preferred_resolution=ActionType.ESCALATE_RISK,
                preferred_escalation_reason=EscalationReason.SUSPECTED_FRAUD,
                order_details_text="Geo-velocity and BIN mismatch triggered the fraud queue.",
                history_details_text="This is the first order on the account, but the card has multiple linked declines.",
                sla_minutes=96 + self._rng.randint(0, 18),
                customer_satisfaction_bonus=0.03,
            ),
            CaseInternal(
                case_id="HARD-6",
                title="Small standard refund still needs handling",
                case_type=CaseType.REFUND_REQUEST,
                customer_tier=CustomerTier.SILVER,
                priority=CasePriority.LOW,
                order_value_usd=small_refund_value,
                days_since_order=11 + self._rng.randint(0, 4),
                order_status=OrderStatus.DELIVERED,
                fraud_signal=FraudSignal.LOW,
                requested_compensation_usd=small_refund_value,
                required_checks={"order"},
                preferred_resolution=ActionType.ISSUE_REFUND,
                refund_range=(max(20.0, small_refund_value - 2.0), small_refund_value + 2.0),
                order_details_text="Straightforward return request within the return window.",
                sla_minutes=188 + self._rng.randint(0, 18),
                customer_satisfaction_bonus=0.05,
            ),
            CaseInternal(
                case_id="HARD-7",
                title="Lost earbuds shipment",
                case_type=CaseType.DELIVERY_ISSUE,
                customer_tier=CustomerTier.GOLD,
                priority=CasePriority.HIGH,
                order_value_usd=earbuds_value,
                days_since_order=8 + self._rng.randint(0, 3),
                order_status=OrderStatus.LOST,
                fraud_signal=FraudSignal.LOW,
                replacement_sku="earbuds-lite",
                required_checks={"order", "inventory"},
                preferred_resolution=ActionType.SHIP_REPLACEMENT,
                preferred_expedite=False,
                order_details_text="Carrier marked the parcel lost in transit after depot transfer.",
                inventory_summary="earbuds-lite inventory is scarce and may not support repeated replacements.",
                sla_minutes=88 + self._rng.randint(0, 16),
                customer_satisfaction_bonus=0.08,
            ),
        ]
        tail = cases[1:]
        self._rng.shuffle(tail)
        return [cases[0], *tail]

    def _validate_action(self, action: ShopopsAction) -> Optional[str]:
        if action.action_type == ActionType.SWITCH_CASE:
            if not action.case_id:
                return "case_id_required_for_switch"
            if self._case_by_id(action.case_id) is None:
                return "unknown_case_id"
        else:
            if action.case_id and action.case_id != self._active_case_id:
                return "action_targets_inactive_case"

        if action.action_type == ActionType.ISSUE_REFUND:
            if action.refund_amount_usd is None or action.refund_amount_usd <= 0:
                return "refund_amount_required"
        else:
            if action.refund_amount_usd is not None:
                return "refund_amount_not_allowed"

        if action.action_type == ActionType.ESCALATE_RISK:
            if action.escalation_reason is None:
                return "escalation_reason_required"
        else:
            if action.escalation_reason is not None:
                return "escalation_reason_not_allowed"

        if action.action_type == ActionType.ADD_INTERNAL_NOTE:
            if not action.note_code:
                return "note_code_required"
        else:
            if action.note_code is not None:
                return "note_code_not_allowed"

        if action.action_type not in {ActionType.SHIP_REPLACEMENT, ActionType.SWITCH_CASE} and action.expedite:
            return "expedite_not_allowed"

        case = self._active_case()
        if case.status == CaseStatus.CLOSED and action.action_type != ActionType.SWITCH_CASE:
            return "case_closed"
        if action.action_type == ActionType.CLOSE_CASE and self._blockers_for_case(case):
            return "cannot_close_with_blockers"
        if action.action_type == ActionType.SHIP_REPLACEMENT:
            if not case.replacement_sku:
                return "no_replacement_sku"
            units = self._inventory.get(case.replacement_sku, 0)
            if units <= 0:
                return "replacement_out_of_stock"
        return None

    def _handle_action(self, action: ShopopsAction, case: CaseInternal) -> ActionOutcome:
        handler = {
            ActionType.INSPECT_ORDER: self._inspect_order,
            ActionType.INSPECT_POLICY: self._inspect_policy,
            ActionType.INSPECT_INVENTORY: self._inspect_inventory,
            ActionType.INSPECT_CUSTOMER_HISTORY: self._inspect_customer_history,
            ActionType.REQUEST_EVIDENCE: self._request_evidence,
            ActionType.CONTACT_CARRIER: self._contact_carrier,
            ActionType.ISSUE_REFUND: self._issue_refund,
            ActionType.SHIP_REPLACEMENT: self._ship_replacement,
            ActionType.ESCALATE_RISK: self._escalate_risk,
            ActionType.ADD_INTERNAL_NOTE: self._add_internal_note,
            ActionType.CLOSE_CASE: self._close_case,
            ActionType.SWITCH_CASE: self._switch_case,
        }[action.action_type]
        return handler(case, action)

    def _inspect_order(self, case: CaseInternal, action: ShopopsAction) -> ActionOutcome:
        del action
        if "order" in case.completed_checks:
            return ActionOutcome(
                reward=-0.03,
                summary="Order details were already inspected.",
                details={"reward_breakdown": {"information_gain": -0.03}},
            )
        case.completed_checks.add("order")
        case.order_summary = case.order_details_text
        return ActionOutcome(
            reward=0.08,
            summary="Order details inspected.",
            details={
                "order_summary": case.order_summary,
                "reward_breakdown": {"information_gain": 0.08},
            },
        )

    def _inspect_policy(self, case: CaseInternal, action: ShopopsAction) -> ActionOutcome:
        del action
        if "policy" in case.completed_checks:
            return ActionOutcome(
                reward=-0.03,
                summary="Policy details were already inspected.",
                details={"reward_breakdown": {"information_gain": -0.03}},
            )
        case.completed_checks.add("policy")
        case.policy_summary = case.policy_details_text or "No special policy guidance for this case."
        reward = 0.08 if case.policy_details_text else 0.02
        return ActionOutcome(
            reward=reward,
            summary="Policy lookup completed.",
            details={
                "policy_summary": case.policy_summary,
                "reward_breakdown": {"information_gain": reward},
            },
        )

    def _inspect_inventory(self, case: CaseInternal, action: ShopopsAction) -> ActionOutcome:
        del action
        if "inventory" in case.completed_checks:
            return ActionOutcome(
                reward=-0.03,
                summary="Inventory was already inspected.",
                details={"reward_breakdown": {"information_gain": -0.03}},
            )
        case.completed_checks.add("inventory")
        sku = case.replacement_sku or "none"
        units = self._inventory.get(case.replacement_sku or "", 0)
        case.inventory_summary = case.inventory_summary or f"{sku} inventory has {units} unit(s) remaining."
        return ActionOutcome(
            reward=0.08,
            summary="Inventory lookup completed.",
            details={
                "inventory_summary": case.inventory_summary,
                "reward_breakdown": {"information_gain": 0.08},
            },
        )

    def _inspect_customer_history(self, case: CaseInternal, action: ShopopsAction) -> ActionOutcome:
        del action
        if "history" in case.completed_checks:
            return ActionOutcome(
                reward=-0.03,
                summary="Customer history was already inspected.",
                details={"reward_breakdown": {"information_gain": -0.03}},
            )
        case.completed_checks.add("history")
        case.history_summary = case.history_details_text or "No significant customer history was found."
        reward = 0.08 if case.history_details_text else 0.02
        return ActionOutcome(
            reward=reward,
            summary="Customer history lookup completed.",
            details={
                "history_summary": case.history_summary,
                "reward_breakdown": {"information_gain": reward},
            },
        )

    def _request_evidence(self, case: CaseInternal, action: ShopopsAction) -> ActionOutcome:
        del action
        if not case.needs_evidence:
            return ActionOutcome(
                reward=-0.05,
                summary="This case does not require customer evidence.",
                details={"reward_breakdown": {"workflow_progress": -0.05}},
            )
        if case.evidence_status == EvidenceStatus.REQUESTED:
            return ActionOutcome(
                reward=-0.03,
                summary="Evidence request is already pending.",
                details={"reward_breakdown": {"workflow_progress": -0.03}},
            )
        if case.evidence_status in {EvidenceStatus.RECEIVED, EvidenceStatus.INSUFFICIENT}:
            return ActionOutcome(
                reward=-0.02,
                summary="Evidence result is already available.",
                details={"reward_breakdown": {"workflow_progress": -0.02}},
            )
        case.evidence_status = EvidenceStatus.REQUESTED
        case.status = CaseStatus.WAITING_CUSTOMER
        summary = case.evidence_response_text or "Customer evidence arrived."
        case.pending_events.append("customer_evidence")
        self._pending_events.append(
            PendingEvent(
                case_id=case.case_id,
                event_type="customer_evidence",
                ready_step=self._state.step_count + 2,
                summary=summary,
                outcome=case.evidence_outcome.value,
            )
        )
        return ActionOutcome(
            reward=0.1,
            summary="Evidence request sent to the customer.",
            details={"reward_breakdown": {"workflow_progress": 0.1}},
        )

    def _contact_carrier(self, case: CaseInternal, action: ShopopsAction) -> ActionOutcome:
        del action
        if not case.needs_carrier_contact:
            return ActionOutcome(
                reward=-0.05,
                summary="Carrier contact is not needed for this case.",
                details={"reward_breakdown": {"workflow_progress": -0.05}},
            )
        if case.carrier_status == CarrierStatus.INVESTIGATING:
            return ActionOutcome(
                reward=-0.03,
                summary="Carrier investigation is already pending.",
                details={"reward_breakdown": {"workflow_progress": -0.03}},
            )
        if case.carrier_status in {CarrierStatus.APPROVED, CarrierStatus.DENIED}:
            return ActionOutcome(
                reward=-0.02,
                summary="Carrier result is already available.",
                details={"reward_breakdown": {"workflow_progress": -0.02}},
            )
        case.carrier_status = CarrierStatus.INVESTIGATING
        case.status = CaseStatus.WAITING_CARRIER
        case.pending_events.append("carrier_response")
        self._pending_events.append(
            PendingEvent(
                case_id=case.case_id,
                event_type="carrier_response",
                ready_step=self._state.step_count + 2,
                summary=case.carrier_response_text or "Carrier responded to the investigation.",
                outcome=case.carrier_outcome.value,
            )
        )
        return ActionOutcome(
            reward=0.1,
            summary="Carrier investigation started.",
            details={"reward_breakdown": {"workflow_progress": 0.1}},
        )

    def _issue_refund(self, case: CaseInternal, action: ShopopsAction) -> ActionOutcome:
        amount = float(action.refund_amount_usd or 0.0)
        case.resolution_action = ActionType.ISSUE_REFUND
        case.resolution_amount_usd = round(amount, 2)
        case.resolution_expedite = False
        case.resolution_escalation_reason = None
        case.status = CaseStatus.RESOLVED
        fit = self._refund_fit(case, amount)
        workflow = self._check_coverage(case)
        business = 0.22 + fit
        workflow_reward = 0.08 if workflow >= 1.0 else max(-0.08, 0.08 * (workflow - 1.0))
        if case.fraud_signal == FraudSignal.HIGH and case.evidence_status in {
            EvidenceStatus.NOT_REQUESTED,
            EvidenceStatus.REQUESTED,
        }:
            business -= 0.18
        reward = max(-0.2, business + workflow_reward)
        case.resolution_summary = f"Refund of ${amount:.2f} prepared."
        return ActionOutcome(
            reward=reward,
            summary=case.resolution_summary,
            details={
                "reward_breakdown": {
                    "workflow_progress": workflow_reward,
                    "business_outcome": reward - workflow_reward,
                }
            },
            budget_delta=amount,
        )

    def _ship_replacement(self, case: CaseInternal, action: ShopopsAction) -> ActionOutcome:
        sku = case.replacement_sku or ""
        units = self._inventory.get(sku, 0)
        if units <= 0:
            self._stockouts += 1
            return ActionOutcome(
                reward=-0.2,
                summary="Replacement failed because inventory is exhausted.",
                details={"reward_breakdown": {"business_outcome": -0.2}},
            )
        self._inventory[sku] = units - 1
        case.resolution_action = ActionType.SHIP_REPLACEMENT
        case.resolution_amount_usd = None
        case.resolution_expedite = action.expedite
        case.resolution_escalation_reason = None
        case.status = CaseStatus.RESOLVED
        workflow = self._check_coverage(case)
        expedite_bonus = 0.08 if action.expedite == case.preferred_expedite else -0.04
        resolution_bonus = 0.26 if case.preferred_resolution == ActionType.SHIP_REPLACEMENT else -0.12
        reward = max(-0.2, resolution_bonus + expedite_bonus + 0.06 * workflow)
        case.resolution_summary = (
            f"Replacement for {sku} queued{' with expedite' if action.expedite else ''}."
        )
        if self._inventory[sku] == 0:
            self._stockouts += 1
        return ActionOutcome(
            reward=reward,
            summary=case.resolution_summary,
            details={
                "reward_breakdown": {
                    "workflow_progress": 0.06 * workflow,
                    "business_outcome": reward - (0.06 * workflow),
                }
            },
            budget_delta=min(case.order_value_usd * 0.45, 320.0) + (25.0 if action.expedite else 0.0),
        )

    def _escalate_risk(self, case: CaseInternal, action: ShopopsAction) -> ActionOutcome:
        case.resolution_action = ActionType.ESCALATE_RISK
        case.resolution_amount_usd = None
        case.resolution_expedite = False
        case.resolution_escalation_reason = action.escalation_reason
        case.status = CaseStatus.RESOLVED
        correct_reason = action.escalation_reason == case.preferred_escalation_reason
        reward = 0.28 if correct_reason else 0.08
        reward += 0.05 * self._check_coverage(case)
        case.resolution_summary = f"Risk escalation filed as {action.escalation_reason.value}."
        return ActionOutcome(
            reward=reward,
            summary=case.resolution_summary,
            details={
                "reward_breakdown": {
                    "workflow_progress": 0.05 * self._check_coverage(case),
                    "business_outcome": reward - (0.05 * self._check_coverage(case)),
                }
            },
        )

    def _add_internal_note(self, case: CaseInternal, action: ShopopsAction) -> ActionOutcome:
        note_code = action.note_code or "general_note"
        if note_code in case.notes:
            return ActionOutcome(
                reward=-0.02,
                summary="That note already exists on the case.",
                details={"reward_breakdown": {"workflow_progress": -0.02}},
            )
        case.notes.append(note_code)
        reward = 0.05 if case.requires_note else 0.01
        return ActionOutcome(
            reward=reward,
            summary=f"Internal note '{note_code}' added.",
            details={"reward_breakdown": {"workflow_progress": reward}},
        )

    def _close_case(self, case: CaseInternal, action: ShopopsAction) -> ActionOutcome:
        del action
        blockers = self._blockers_for_case(case)
        quality = self._resolution_quality(case)
        reward = 0.12 + 0.22 * quality
        case.closed_at_step = self._state.step_count + 1
        case.status = CaseStatus.CLOSED
        if case.fraud_loss_if_bad_close_usd > 0 and quality < 0.3:
            self._fraud_loss_usd += case.fraud_loss_if_bad_close_usd
            reward -= 0.1
        self._customer_satisfaction = min(
            1.0,
            max(
                0.0,
                self._customer_satisfaction
                + (case.customer_satisfaction_bonus if quality >= 0.7 and not blockers else -0.04),
            ),
        )
        case.resolution_summary = case.resolution_summary or "Case closed."
        return ActionOutcome(
            reward=max(-0.25, reward),
            summary=f"Case {case.case_id} closed.",
            details={
                "reward_breakdown": {
                    "workflow_progress": 0.12,
                    "business_outcome": max(-0.37, reward - 0.12),
                },
                "closure_quality": round(quality, 4),
                "remaining_blockers": blockers,
            },
        )

    def _switch_case(self, case: CaseInternal, action: ShopopsAction) -> ActionOutcome:
        del case
        target = self._case_by_id(action.case_id or "")
        if target is None:
            return ActionOutcome(
                reward=-0.1,
                summary="Cannot switch because the target case does not exist.",
                details={"reward_breakdown": {"workflow_progress": -0.1}},
            )
        if target.case_id == self._active_case_id:
            return ActionOutcome(
                reward=-0.02,
                summary="The target case is already active.",
                details={"reward_breakdown": {"workflow_progress": -0.02}},
            )
        current = self._active_case()
        self._active_case_id = target.case_id
        better_target = (
            PRIORITY_SCORES[target.priority] > PRIORITY_SCORES[current.priority]
            or self._sla_remaining(target) < self._sla_remaining(current)
        )
        reward = 0.06 if better_target else 0.01
        return ActionOutcome(
            reward=reward,
            summary=f"Switched active case to {target.case_id}.",
            details={"reward_breakdown": {"workflow_progress": reward}},
        )

    def _advance_events(self) -> None:
        ready_events = [event for event in self._pending_events if event.ready_step <= self._state.step_count]
        if not ready_events:
            return
        remaining = [event for event in self._pending_events if event.ready_step > self._state.step_count]
        self._pending_events = remaining
        for event in ready_events:
            case = self._case_by_id(event.case_id)
            if case is None:
                continue
            if event.event_type == "customer_evidence":
                case.evidence_status = EvidenceStatus(event.outcome)
                case.pending_events = [name for name in case.pending_events if name != "customer_evidence"]
                case.status = CaseStatus.OPEN if case.status == CaseStatus.WAITING_CUSTOMER else case.status
                case.order_summary = self._append_summary(case.order_summary, event.summary)
            elif event.event_type == "carrier_response":
                case.carrier_status = CarrierStatus(event.outcome)
                case.pending_events = [name for name in case.pending_events if name != "carrier_response"]
                case.status = CaseStatus.OPEN if case.status == CaseStatus.WAITING_CARRIER else case.status
                case.policy_summary = self._append_summary(case.policy_summary, event.summary)
            elif event.event_type == "reopen_case":
                if case.status == CaseStatus.CLOSED:
                    case.status = CaseStatus.REOPENED
                    case.reopened_count += 1
                    self._reopened_cases += 1
                    self._customer_satisfaction = max(0.0, self._customer_satisfaction - 0.08)
                case.pending_events = [name for name in case.pending_events if name != "reopen_case"]
                case.resolution_summary = event.summary

    def _update_sla_breaches(self) -> None:
        for case in self._cases:
            if case.sla_breached:
                continue
            if case.status == CaseStatus.CLOSED:
                continue
            if self._time_used > case.sla_minutes:
                case.sla_breached = True
                self._sla_breaches += 1
                self._customer_satisfaction = max(0.0, self._customer_satisfaction - 0.04)

    def _apply_terminal_bonus(self) -> float:
        if self._terminal_bonus_applied:
            return 0.0
        self._terminal_bonus_applied = True
        summary = self._episode_summary()
        return float(summary["terminal_bonus"])

    def _episode_summary(self) -> Dict[str, object]:
        termination_reason = self._termination_reason()
        closed_cases = sum(1 for case in self._cases if case.status == CaseStatus.CLOSED)
        resolved_cases = sum(1 for case in self._cases if case.resolution_action is not None)
        quality_scores = [self._resolution_quality(case) for case in self._cases]
        policy_score = max(0.0, min(1.0, sum(max(score, 0.0) for score in quality_scores) / max(len(self._cases), 1)))
        closure_score = closed_cases / max(len(self._cases), 1)
        resolved_score = resolved_cases / max(len(self._cases), 1)
        sla_score = max(0.0, 1.0 - (self._sla_breaches / max(len(self._cases), 1)))
        stock_score = max(0.0, 1.0 - (self._stockouts / max(len(self._cases), 1)))
        fraud_score = max(0.0, 1.0 - (self._fraud_loss_usd / max(self._scenario.budget_limit_usd, 1.0)))
        customer_score = max(0.0, min(1.0, self._customer_satisfaction))
        invalid_penalty = min(1.0, self._invalid_count / INVALID_LIMIT)
        unresolved_ratio = 1.0 - closure_score
        business_score = (
            0.35 * policy_score
            + 0.15 * resolved_score
            + 0.05 * closure_score
            + 0.15 * sla_score
            + 0.15 * stock_score
            + 0.15 * fraud_score
            + 0.05 * customer_score
        )
        if termination_reason == "invalid_action_limit":
            business_score -= 0.35
        elif termination_reason == "step_limit":
            business_score -= 0.2 + 0.2 * unresolved_ratio
        business_score -= 0.35 * invalid_penalty
        business_score -= 0.25 * unresolved_ratio
        terminal_bonus = max(0.0, min(0.5, business_score * 0.5))
        final_score = max(SCORE_MIN, min(SCORE_MAX, business_score))
        return {
            "task": self._task_name,
            "difficulty": self._difficulty,
            "final_score": final_score,
            "terminal_bonus": round(terminal_bonus, 4),
            "closed_cases": closed_cases,
            "resolved_cases": resolved_cases,
            "reopened_cases": self._reopened_cases,
            "sla_breaches": self._sla_breaches,
            "fraud_loss_usd": round(self._fraud_loss_usd, 2),
            "stockouts": self._stockouts,
            "customer_satisfaction": round(customer_score, 4),
            "invalid_actions": self._invalid_count,
            "time_used_minutes": self._time_used,
            "budget_used_usd": round(self._budget_used, 2),
            "termination_reason": termination_reason,
        }

    def _resolution_quality(self, case: CaseInternal) -> float:
        score = 0.0
        if case.resolution_action is None:
            return 0.0
        if case.resolution_action == case.preferred_resolution:
            score += 0.5
        elif case.resolution_action == ActionType.ISSUE_REFUND and case.preferred_resolution == ActionType.SHIP_REPLACEMENT:
            score += 0.15
        else:
            score -= 0.15

        score += 0.2 * self._check_coverage(case)
        if case.preferred_resolution == ActionType.ISSUE_REFUND:
            score += self._refund_fit(case, case.resolution_amount_usd or 0.0)
        if case.preferred_resolution == ActionType.SHIP_REPLACEMENT:
            score += 0.12 if case.resolution_expedite == case.preferred_expedite else 0.0
        if case.preferred_resolution == ActionType.ESCALATE_RISK:
            score += (
                0.12
                if case.resolution_escalation_reason == case.preferred_escalation_reason
                else 0.0
            )
        if case.requires_note:
            score += 0.08 if case.notes else -0.08
        if case.needs_evidence and case.evidence_status == EvidenceStatus.REQUESTED:
            score -= 0.12
        if case.needs_carrier_contact and case.carrier_status == CarrierStatus.INVESTIGATING:
            score -= 0.12
        return max(0.0, min(1.0, score))

    def _refund_fit(self, case: CaseInternal, amount: float) -> float:
        if case.refund_range is None:
            return 0.0
        low, high = case.refund_range
        if low <= amount <= high:
            return 0.25
        if amount < low:
            delta = low - amount
        else:
            delta = amount - high
        if delta <= 15.0:
            return 0.1
        return -0.1

    def _check_coverage(self, case: CaseInternal) -> float:
        if not case.required_checks:
            return 1.0
        done = sum(1 for check in case.required_checks if check in case.completed_checks)
        return done / len(case.required_checks)

    def _blockers_for_case(self, case: CaseInternal) -> List[str]:
        blockers = []
        for check in sorted(case.required_checks):
            if check not in case.completed_checks:
                blockers.append(f"{check}_review_required")
        if case.needs_evidence and case.evidence_status in {
            EvidenceStatus.NOT_REQUESTED,
            EvidenceStatus.REQUESTED,
        }:
            blockers.append("customer_evidence_pending")
        if case.needs_carrier_contact and case.carrier_status in {
            CarrierStatus.NOT_CONTACTED,
            CarrierStatus.INVESTIGATING,
        }:
            blockers.append("carrier_confirmation_pending")
        if case.resolution_action is None:
            blockers.append("resolution_not_recorded")
        if case.requires_note and not case.notes:
            blockers.append("internal_note_required")
        return blockers

    def _build_observation(
        self,
        reward: Optional[float],
        done: bool,
        info: Dict[str, object],
    ) -> ShopopsObservation:
        active_case = self._active_case_or_empty()
        queue = [
            QueueItemView(
                case_id=case.case_id,
                title=case.title,
                case_type=case.case_type,
                status=case.status,
                priority=case.priority,
                customer_tier=case.customer_tier,
                sla_minutes_remaining=max(self._sla_remaining(case), 0),
                blocker_count=len(self._blockers_for_case(case)) if case.status != CaseStatus.CLOSED else 0,
            )
            for case in self._cases
        ]
        resources = Resources(
            time_remaining_minutes=max(self._scenario.time_limit_minutes - self._time_used, 0),
            budget_remaining_usd=round(max(self._scenario.budget_limit_usd - self._budget_used, 0.0), 2),
            time_used_minutes=self._time_used,
            budget_used_usd=round(self._budget_used, 2),
            inventory_units=copy.deepcopy(self._inventory),
        )
        metrics = BusinessMetrics(
            resolved_cases=sum(1 for case in self._cases if case.resolution_action is not None),
            closed_cases=sum(1 for case in self._cases if case.status == CaseStatus.CLOSED),
            reopened_cases=self._reopened_cases,
            sla_breaches=self._sla_breaches,
            fraud_loss_usd=round(self._fraud_loss_usd, 2),
            customer_satisfaction=round(max(0.0, min(1.0, self._customer_satisfaction)), 4),
            stockouts=self._stockouts,
        )
        metadata = {
            **info,
            "last_action_error": self._last_action_error,
            "task": self._task_name,
            "difficulty": self._difficulty,
            "max_steps": self._scenario.step_limit,
        }
        return ShopopsObservation(
            active_case=self._case_to_view(active_case),
            queue=queue,
            latest_tool_result=self._latest_tool_result,
            resources=resources,
            metrics=metrics,
            unresolved_blockers=self._blockers_for_case(active_case) if active_case.case_id else [],
            current_task=self._task_name,
            difficulty=self._difficulty,
            step_index=self._state.step_count,
            episode_id=self._state.episode_id or "",
            env_schema_version=ENV_SCHEMA_VERSION,
            reward=reward,
            done=done,
            metadata=metadata,
        )

    def _case_to_view(self, case: CaseInternal) -> CaseView:
        return CaseView(
            case_id=case.case_id,
            title=case.title,
            case_type=case.case_type,
            status=case.status,
            priority=case.priority,
            customer_tier=case.customer_tier,
            order_value_usd=case.order_value_usd,
            days_since_order=case.days_since_order,
            order_status=case.order_status,
            fraud_signal=case.fraud_signal,
            requested_compensation_usd=case.requested_compensation_usd,
            replacement_sku=case.replacement_sku,
            evidence_status=case.evidence_status,
            carrier_status=case.carrier_status,
            order_summary=case.order_summary,
            policy_summary=case.policy_summary,
            history_summary=case.history_summary,
            inventory_summary=case.inventory_summary,
            blockers=self._blockers_for_case(case) if case.case_id else [],
            pending_events=list(case.pending_events),
            completed_checks=sorted(case.completed_checks),
            notes=list(case.notes),
            resolution_action=case.resolution_action.value if case.resolution_action else None,
            resolution_summary=case.resolution_summary,
        )

    def _active_case(self) -> CaseInternal:
        if self._active_case_id is None:
            return self._active_case_or_empty()
        case = self._case_by_id(self._active_case_id)
        if case is None:
            raise RuntimeError(f"Active case not found: {self._active_case_id}")
        return case

    def _active_case_or_empty(self) -> CaseInternal:
        if self._active_case_id is not None:
            case = self._case_by_id(self._active_case_id)
            if case is not None:
                return case
        return CaseInternal(
            case_id="",
            title="No active case",
            case_type=CaseType.REFUND_REQUEST,
            customer_tier=CustomerTier.BRONZE,
            priority=CasePriority.LOW,
            order_value_usd=0.0,
            days_since_order=0,
            order_status=OrderStatus.DELIVERED,
            fraud_signal=FraudSignal.LOW,
        )

    def _case_by_id(self, case_id: str) -> Optional[CaseInternal]:
        for case in self._cases:
            if case.case_id == case_id:
                return case
        return None

    def _sla_remaining(self, case: CaseInternal) -> int:
        return case.sla_minutes - self._time_used

    def _append_summary(self, existing: Optional[str], addition: str) -> str:
        if not existing:
            return addition
        if addition in existing:
            return existing
        return f"{existing} {addition}"

    def _termination_reason(self) -> str:
        if self._invalid_count >= INVALID_LIMIT:
            return "invalid_action_limit"
        if self._budget_used > self._scenario.budget_limit_usd:
            return "budget_exhausted"
        if self._time_used > self._scenario.time_limit_minutes:
            return "time_exhausted"
        if self._state.step_count >= self._scenario.step_limit:
            return "step_limit"
        if self._all_cases_closed():
            return "cases_closed"
        return "unknown"

    def _all_cases_closed(self) -> bool:
        return all(case.status == CaseStatus.CLOSED for case in self._cases)

    def _is_done(self) -> bool:
        return (
            self._invalid_count >= INVALID_LIMIT
            or self._budget_used > self._scenario.budget_limit_usd
            or self._time_used > self._scenario.time_limit_minutes
            or self._state.step_count >= self._scenario.step_limit
            or self._all_cases_closed()
        )
