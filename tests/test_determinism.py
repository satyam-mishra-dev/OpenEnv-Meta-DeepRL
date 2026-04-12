from __future__ import annotations

from shopOps.eval import baseline_policy
from shopOps.server.shopOps_environment import ShopopsEnvironment


def _snapshot(env: ShopopsEnvironment) -> list[tuple[str, float, int]]:
    return [
        (case.case_id, round(case.order_value_usd, 2), case.sla_minutes)
        for case in env._cases
    ]


def test_determinism_same_seed_same_task() -> None:
    env1 = ShopopsEnvironment(debug_mode=True)
    env2 = ShopopsEnvironment(debug_mode=True)

    obs1 = env1.reset(seed=42, task="fraud_stockout_cascade")
    obs2 = env2.reset(seed=42, task="fraud_stockout_cascade")

    assert _snapshot(env1) == _snapshot(env2)
    assert obs1.active_case.case_id == obs2.active_case.case_id


def test_determinism_different_seed_changes_case_values() -> None:
    env1 = ShopopsEnvironment(debug_mode=True)
    env2 = ShopopsEnvironment(debug_mode=True)

    env1.reset(seed=41, task="refund_policy_recovery")
    env2.reset(seed=42, task="refund_policy_recovery")

    assert _snapshot(env1) != _snapshot(env2)


def test_reproducible_trajectory() -> None:
    def run(seed: int) -> list[float]:
        env = ShopopsEnvironment(debug_mode=True)
        obs = env.reset(seed=seed, task="sla_queue_juggle")
        rewards = []
        while True:
            action = baseline_policy(obs)
            obs = env.step(action)
            rewards.append(round(float(obs.reward or 0.0), 6))
            if obs.done:
                break
        return rewards

    assert run(7) == run(7)
