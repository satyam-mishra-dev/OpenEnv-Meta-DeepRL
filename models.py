from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


class CaseType(str, Enum):
    REFUND_REQUEST = "refund_request"
    DELIVERY_ISSUE = "delivery_issue"
    WRONG_ITEM = "wrong_item"
    FRAUD_SIGNAL = "fraud_signal"


class CustomerTier(str, Enum):
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"
    PLATINUM = "platinum"


class OrderStatus(str, Enum):
    DELIVERED = "delivered"
    IN_TRANSIT = "in_transit"
    DELAYED = "delayed"
    LOST = "lost"


class FraudSignal(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class CasePriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CaseStatus(str, Enum):
    OPEN = "open"
    WAITING_CUSTOMER = "waiting_customer"
    WAITING_CARRIER = "waiting_carrier"
    RESOLVED = "resolved"
    CLOSED = "closed"
    REOPENED = "reopened"


class EvidenceStatus(str, Enum):
    NOT_REQUESTED = "not_requested"
    REQUESTED = "requested"
    RECEIVED = "received"
    INSUFFICIENT = "insufficient"


class CarrierStatus(str, Enum):
    NOT_CONTACTED = "not_contacted"
    INVESTIGATING = "investigating"
    APPROVED = "approved"
    DENIED = "denied"


class EscalationReason(str, Enum):
    SUSPECTED_FRAUD = "suspected_fraud"
    POLICY_EXCEPTION = "policy_exception"
    SLA_RISK = "sla_risk"
    VIP_RECOVERY = "vip_recovery"


class ActionType(str, Enum):
    INSPECT_ORDER = "inspect_order"
    INSPECT_POLICY = "inspect_policy"
    INSPECT_INVENTORY = "inspect_inventory"
    INSPECT_CUSTOMER_HISTORY = "inspect_customer_history"
    REQUEST_EVIDENCE = "request_evidence"
    CONTACT_CARRIER = "contact_carrier"
    ISSUE_REFUND = "issue_refund"
    SHIP_REPLACEMENT = "ship_replacement"
    ESCALATE_RISK = "escalate_risk"
    ADD_INTERNAL_NOTE = "add_internal_note"
    CLOSE_CASE = "close_case"
    SWITCH_CASE = "switch_case"


class QueueItemView(BaseModel):
    case_id: str = Field(..., description="Unique case identifier")
    title: str = Field(..., description="Short case summary")
    case_type: CaseType = Field(..., description="Type of case")
    status: CaseStatus = Field(..., description="Current case status")
    priority: CasePriority = Field(..., description="Priority label")
    customer_tier: CustomerTier = Field(..., description="Customer tier")
    sla_minutes_remaining: int = Field(..., ge=0, description="Minutes before SLA breach")
    blocker_count: int = Field(..., ge=0, description="Outstanding blocker count")


class ToolResult(BaseModel):
    action_type: Optional[ActionType] = Field(default=None, description="Action just executed")
    target_case_id: Optional[str] = Field(default=None, description="Target case for the action")
    outcome: str = Field(default="", description="Human-readable action result")
    details: Dict[str, object] = Field(default_factory=dict, description="Structured tool payload")


class Resources(BaseModel):
    time_remaining_minutes: int = Field(..., ge=0)
    budget_remaining_usd: float = Field(..., ge=0)
    time_used_minutes: int = Field(..., ge=0)
    budget_used_usd: float = Field(..., ge=0)
    inventory_units: Dict[str, int] = Field(
        default_factory=dict,
        description="Replacement inventory counts by SKU",
    )


class BusinessMetrics(BaseModel):
    resolved_cases: int = Field(..., ge=0)
    closed_cases: int = Field(..., ge=0)
    reopened_cases: int = Field(..., ge=0)
    sla_breaches: int = Field(..., ge=0)
    fraud_loss_usd: float = Field(..., ge=0)
    customer_satisfaction: float = Field(..., ge=0, le=1.0)
    stockouts: int = Field(..., ge=0)


class CaseView(BaseModel):
    case_id: str = Field(..., description="Unique case identifier")
    title: str = Field(..., description="Short case title")
    case_type: CaseType = Field(..., description="Case category")
    status: CaseStatus = Field(..., description="Current state of the case")
    priority: CasePriority = Field(..., description="Priority label")
    customer_tier: CustomerTier = Field(..., description="Customer tier")
    order_value_usd: float = Field(..., ge=0, description="Order value in USD")
    days_since_order: int = Field(..., ge=0, description="Days since order creation")
    order_status: OrderStatus = Field(..., description="Current shipment/order state")
    fraud_signal: FraudSignal = Field(..., description="Fraud risk estimate")
    requested_compensation_usd: Optional[float] = Field(
        default=None,
        ge=0,
        description="Customer-requested refund amount when applicable",
    )
    replacement_sku: Optional[str] = Field(
        default=None,
        description="Replacement SKU if a shipment can be sent",
    )
    evidence_status: EvidenceStatus = Field(..., description="Status of customer evidence collection")
    carrier_status: CarrierStatus = Field(..., description="Status of carrier investigation")
    order_summary: Optional[str] = Field(default=None, description="Persistent order inspection result")
    policy_summary: Optional[str] = Field(default=None, description="Persistent policy lookup result")
    history_summary: Optional[str] = Field(default=None, description="Persistent customer history summary")
    inventory_summary: Optional[str] = Field(default=None, description="Persistent inventory lookup result")
    blockers: List[str] = Field(default_factory=list, description="Open blockers before closure")
    pending_events: List[str] = Field(default_factory=list, description="Known incoming delayed events")
    completed_checks: List[str] = Field(default_factory=list, description="Checks already performed")
    notes: List[str] = Field(default_factory=list, description="Internal note codes added to the case")
    resolution_action: Optional[str] = Field(default=None, description="Recorded resolution action")
    resolution_summary: Optional[str] = Field(default=None, description="Summary of the current resolution")


class ShopopsAction(Action):
    action_type: ActionType = Field(..., description="Tool or workflow action to execute")
    case_id: Optional[str] = Field(
        default=None,
        description="Target case identifier. Required for switch_case; otherwise must match the active case if present.",
    )
    refund_amount_usd: Optional[float] = Field(
        default=None,
        ge=0,
        description="Refund amount for issue_refund",
    )
    expedite: bool = Field(
        default=False,
        description="Whether replacement shipping should be expedited",
    )
    escalation_reason: Optional[EscalationReason] = Field(
        default=None,
        description="Reason for escalate_risk",
    )
    note_code: Optional[str] = Field(
        default=None,
        description="Structured note code for add_internal_note",
    )


class ShopopsObservation(Observation):
    active_case: CaseView = Field(..., description="Current active case")
    queue: List[QueueItemView] = Field(..., description="Visible work queue")
    latest_tool_result: Optional[ToolResult] = Field(
        default=None,
        description="Result of the last action executed",
    )
    resources: Resources = Field(..., description="Budget, time, and inventory snapshot")
    metrics: BusinessMetrics = Field(..., description="Business outcome snapshot")
    unresolved_blockers: List[str] = Field(
        default_factory=list,
        description="Blockers for the active case after the last action",
    )
    current_task: str = Field(..., description="Task/scenario identifier")
    difficulty: str = Field(..., description="Difficulty label")
    step_index: int = Field(..., ge=0, description="Action count in the episode")
    episode_id: str = Field(..., description="Episode identifier")
    env_schema_version: str = Field(..., description="Environment schema version")
    metadata: Dict[str, object] = Field(default_factory=dict, description="Extended debug metadata")
