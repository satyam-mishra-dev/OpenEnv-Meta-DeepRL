# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the ShopOps Environment.

The ShopOps environment simulates e-commerce support operations.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Optional

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


class DeliveryStatus(str, Enum):
    DELIVERED = "delivered"
    IN_TRANSIT = "in_transit"
    DELAYED = "delayed"
    LOST = "lost"


class IssueSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class FraudSignal(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ItemCategory(str, Enum):
    APPAREL = "apparel"
    ELECTRONICS = "electronics"
    HOME = "home"
    BEAUTY = "beauty"
    GROCERY = "grocery"


class ActionType(str, Enum):
    REFUND = "refund"
    REPLACE = "replace"
    ESCALATE = "escalate"
    REJECT = "reject"


class EscalationReason(str, Enum):
    SUSPECTED_FRAUD = "suspected_fraud"
    HIGH_VALUE = "high_value"
    POLICY_EXCEPTION = "policy_exception"
    SAFETY_ISSUE = "safety_issue"


class CaseView(BaseModel):
    """
    A partially observable view of a case presented to the agent.

    Fields may be hidden or bucketed depending on difficulty tier.
    """

    case_id: str = Field(..., description="Unique case identifier")
    case_type: CaseType = Field(..., description="Type of customer issue")
    customer_tier: CustomerTier = Field(..., description="Customer tier")
    issue_severity: IssueSeverity = Field(..., description="Issue severity")
    fraud_signal: FraudSignal = Field(..., description="Coarse fraud signal")
    item_category: ItemCategory = Field(..., description="Item category")

    order_value_usd: Optional[float] = Field(
        default=None, description="Order value in USD (may be hidden)"
    )
    order_value_bucket: Optional[str] = Field(
        default=None, description="Order value bucket when exact value is hidden"
    )
    days_since_order: Optional[int] = Field(
        default=None, description="Days since order (may be hidden)"
    )
    order_age_bucket: Optional[str] = Field(
        default=None, description="Order age bucket when exact days are hidden"
    )
    delivery_status: Optional[DeliveryStatus] = Field(
        default=None, description="Delivery status for delivery-related issues"
    )
    return_window_open: Optional[bool] = Field(
        default=None, description="Whether return window is open (may be hidden)"
    )
    evidence_provided: Optional[bool] = Field(
        default=None, description="Whether evidence was provided (may be hidden)"
    )
    prior_refund_count_bucket: Optional[str] = Field(
        default=None, description="Bucketed prior refund count (may be hidden)"
    )


class Resources(BaseModel):
    """Resource tracking for the current episode."""

    time_remaining_minutes: int = Field(..., ge=0)
    budget_remaining_usd: float = Field(..., ge=0)
    time_used_minutes: int = Field(..., ge=0)
    budget_used_usd: float = Field(..., ge=0)


class ShopopsAction(Action):
    """Action for the ShopOps environment."""

    action_type: ActionType = Field(..., description="Primary action type")
    refund_amount_usd: Optional[float] = Field(
        default=None, ge=0, description="Refund amount in USD (refund only)"
    )
    replacement_expedite: bool = Field(
        default=False, description="Expedite replacement shipping (replace only)"
    )
    escalation_reason: Optional[EscalationReason] = Field(
        default=None, description="Reason for escalation (escalate only)"
    )
    note_code: Optional[str] = Field(
        default=None, description="Optional structured note code"
    )


class ShopopsObservation(Observation):
    """Observation from the ShopOps environment."""

    case: CaseView = Field(..., description="Current case view")
    resources: Resources = Field(..., description="Resource state")
    case_index: int = Field(..., ge=0, description="Index of current case")
    step_index: int = Field(..., ge=0, description="Step count in episode")
    episode_id: str = Field(..., description="Episode identifier")
    tier: str = Field(..., description="Difficulty tier")
    env_schema_version: str = Field(..., description="Environment schema version")
    metadata: Dict[str, object] = Field(
        default_factory=dict, description="Per-step info and metrics"
    )
