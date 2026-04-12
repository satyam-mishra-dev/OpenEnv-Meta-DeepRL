from __future__ import annotations

from shopOps.eval import baseline_policy, run_episode
from shopOps.models import ActionType, ShopopsAction
from shopOps.server.shopOps_environment import ShopopsEnvironment


def test_refund_policy_recovery_happy_path() -> None:
    env = ShopopsEnvironment(debug_mode=True)
    obs = env.reset(seed=1, task="refund_policy_recovery")

    actions = [
        ShopopsAction(action_type=ActionType.INSPECT_ORDER),
        ShopopsAction(action_type=ActionType.INSPECT_POLICY),
        ShopopsAction(action_type=ActionType.ISSUE_REFUND, refund_amount_usd=92.0),
        ShopopsAction(action_type=ActionType.ADD_INTERNAL_NOTE, note_code="service_credit"),
        ShopopsAction(action_type=ActionType.CLOSE_CASE),
    ]

    for action in actions:
        obs = env.step(action)

    summary = obs.metadata["episode_summary"]
    assert obs.done is True
    assert obs.metadata["termination_reason"] == "cases_closed"
    assert summary["final_score"] > 0.9


def test_sla_queue_baseline_closes_all_cases() -> None:
    result = run_episode(seed=1, task="sla_queue_juggle", debug_mode=True)
    summary = result["episode_summary"]
    assert result["termination_reason"] == "cases_closed"
    assert summary["closed_cases"] == 5
    assert summary["final_score"] > 0.85


def test_hard_task_baseline_finishes_without_invalid_actions() -> None:
    result = run_episode(seed=1, task="fraud_stockout_cascade", debug_mode=True)
    summary = result["episode_summary"]
    assert result["termination_reason"] == "cases_closed"
    assert summary["invalid_actions"] == 0
    assert summary["final_score"] > 0.75


def test_switching_away_from_waiting_case_is_supported() -> None:
    env = ShopopsEnvironment(debug_mode=True)
    obs = env.reset(seed=1, task="sla_queue_juggle")
    env.step(ShopopsAction(action_type=ActionType.SWITCH_CASE, case_id="SLA-5"))
    obs = env.step(ShopopsAction(action_type=ActionType.CONTACT_CARRIER))
    assert obs.active_case.status.value == "waiting_carrier"

    obs = env.step(ShopopsAction(action_type=ActionType.SWITCH_CASE, case_id="SLA-1"))
    assert obs.active_case.case_id == "SLA-1"


def test_rule_baseline_is_reproducible_on_hard_task() -> None:
    env = ShopopsEnvironment(debug_mode=True)
    obs = env.reset(seed=3, task="fraud_stockout_cascade")
    total_reward = 0.0
    while True:
        obs = env.step(baseline_policy(obs))
        total_reward += float(obs.reward or 0.0)
        if obs.done:
            break
    assert total_reward > 5.0
