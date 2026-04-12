from __future__ import annotations

import pytest

from shopOps.models import ActionType, ShopopsAction
from shopOps.server.shopOps_environment import ShopopsEnvironment


def test_observation_schema_fields() -> None:
    env = ShopopsEnvironment(debug_mode=True)
    obs = env.reset(seed=123, task="refund_policy_recovery")

    assert obs.active_case.case_id == "RPR-1"
    assert obs.queue
    assert obs.resources.time_remaining_minutes > 0
    assert obs.metrics.customer_satisfaction >= 0.0
    assert obs.current_task == "refund_policy_recovery"
    assert obs.difficulty == "easy"
    assert obs.step_index == 0
    assert obs.episode_id


def test_invalid_action_enum_rejected() -> None:
    with pytest.raises(ValueError):
        ShopopsAction(action_type="fly_to_moon")  # type: ignore[arg-type]


def test_action_schema_supports_tool_arguments() -> None:
    action = ShopopsAction(
        action_type=ActionType.ISSUE_REFUND,
        refund_amount_usd=42.0,
    )
    payload = action.model_dump(exclude_none=True)
    assert payload["action_type"] == "issue_refund"
    assert payload["refund_amount_usd"] == 42.0
