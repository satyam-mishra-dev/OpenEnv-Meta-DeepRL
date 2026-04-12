from __future__ import annotations

from shopOps.models import (
    ActionType,
    EscalationReason,
    ShopopsAction,
)
from shopOps.server.shopOps_environment import ShopopsEnvironment


def test_validation_missing_refund_amount() -> None:
    env = ShopopsEnvironment(debug_mode=True)
    env.reset(seed=1, task="refund_policy_recovery")

    obs = env.step(ShopopsAction(action_type=ActionType.ISSUE_REFUND))
    assert obs.metadata["last_action_error"] == "refund_amount_required"
    assert obs.metadata["invalid_count"] == 1


def test_resource_budget_updates_after_refund() -> None:
    env = ShopopsEnvironment(debug_mode=True)
    env.reset(seed=1, task="refund_policy_recovery")
    env.step(ShopopsAction(action_type=ActionType.INSPECT_ORDER))
    env.step(ShopopsAction(action_type=ActionType.INSPECT_POLICY))

    obs = env.step(ShopopsAction(action_type=ActionType.ISSUE_REFUND, refund_amount_usd=92.0))
    assert obs.resources.budget_used_usd == 92.0
    assert obs.reward is not None
    assert obs.reward > 0.2


def test_out_of_stock_validation() -> None:
    env = ShopopsEnvironment(debug_mode=True)
    env.reset(seed=1, task="fraud_stockout_cascade")

    env.step(ShopopsAction(action_type=ActionType.INSPECT_ORDER))
    env.step(ShopopsAction(action_type=ActionType.INSPECT_CUSTOMER_HISTORY))
    env.step(ShopopsAction(action_type=ActionType.INSPECT_INVENTORY))
    env.step(ShopopsAction(action_type=ActionType.SHIP_REPLACEMENT, expedite=True))
    env.step(ShopopsAction(action_type=ActionType.ADD_INTERNAL_NOTE, note_code="vip_makegood"))
    env.step(ShopopsAction(action_type=ActionType.CLOSE_CASE))

    env.step(ShopopsAction(action_type=ActionType.SWITCH_CASE, case_id="HARD-3"))
    obs = env.step(ShopopsAction(action_type=ActionType.SHIP_REPLACEMENT))
    assert obs.metadata["last_action_error"] == "replacement_out_of_stock"


def test_premature_close_is_rejected() -> None:
    env = ShopopsEnvironment(debug_mode=True)
    env.reset(seed=1, task="fraud_stockout_cascade")

    env.step(ShopopsAction(action_type=ActionType.SWITCH_CASE, case_id="HARD-2"))
    env.step(ShopopsAction(action_type=ActionType.INSPECT_ORDER))
    env.step(ShopopsAction(action_type=ActionType.INSPECT_POLICY))
    env.step(ShopopsAction(action_type=ActionType.INSPECT_CUSTOMER_HISTORY))
    env.step(ShopopsAction(action_type=ActionType.ISSUE_REFUND, refund_amount_usd=640.0))
    obs = env.step(ShopopsAction(action_type=ActionType.CLOSE_CASE))

    reopened = env._case_by_id("HARD-2")
    assert reopened is not None
    assert reopened.status.value == "resolved"
    assert obs.metadata["last_action_error"] == "cannot_close_with_blockers"


def test_escalation_requires_reason() -> None:
    env = ShopopsEnvironment(debug_mode=True)
    env.reset(seed=1, task="fraud_stockout_cascade")
    env.step(ShopopsAction(action_type=ActionType.SWITCH_CASE, case_id="HARD-5"))

    obs = env.step(ShopopsAction(action_type=ActionType.ESCALATE_RISK))
    assert obs.metadata["last_action_error"] == "escalation_reason_required"
