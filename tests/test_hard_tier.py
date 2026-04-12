from __future__ import annotations

from shopOps.server.shopOps_environment import ShopopsEnvironment


def test_hard_task_has_scarce_inventory() -> None:
    env = ShopopsEnvironment(debug_mode=True)
    obs = env.reset(seed=1, task="fraud_stockout_cascade")

    assert obs.resources.inventory_units["console-pro"] == 1
    assert obs.resources.inventory_units["earbuds-lite"] == 1


def test_hard_task_contains_risky_cases() -> None:
    env = ShopopsEnvironment(debug_mode=True)
    env.reset(seed=1, task="fraud_stockout_cascade")

    hard2 = env._case_by_id("HARD-2")
    hard3 = env._case_by_id("HARD-3")
    assert hard2 is not None and hard2.needs_evidence is True
    assert hard2.fraud_loss_if_bad_close_usd > 0.0
    assert hard3 is not None and hard3.replacement_sku == "console-pro"
