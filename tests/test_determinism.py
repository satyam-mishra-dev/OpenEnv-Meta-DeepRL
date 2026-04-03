# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from typing import List

from shopOps.server.shopOps_environment import ShopopsEnvironment

logger = logging.getLogger(__name__)


def _snapshot_episode(env: ShopopsEnvironment) -> List[dict]:
    snapshot = []
    for case in env._cases:
        snapshot.append(
            {
                "case_id": case.case_id,
                "case_type": case.case_type.value,
                "order_value": case.order_value_usd,
                "days_since_order": case.days_since_order,
                "delivery_status": case.delivery_status.value if case.delivery_status else None,
                "issue_severity": case.issue_severity.value,
                "fraud_score": round(case.fraud_score, 4),
                "hidden_fields": sorted(case.hidden_fields),
                "adversarial": case.adversarial,
            }
        )
    return snapshot


def test_determinism_same_seed():
    env1 = ShopopsEnvironment(debug_mode=True)
    env2 = ShopopsEnvironment(debug_mode=True)

    obs1 = env1.reset(seed=42, tier="hard", split="test")
    obs2 = env2.reset(seed=42, tier="hard", split="test")

    snap1 = _snapshot_episode(env1)
    snap2 = _snapshot_episode(env2)

    logger.info("snapshot1[0]=%s", snap1[0])
    logger.info("snapshot2[0]=%s", snap2[0])

    assert snap1 == snap2
    assert obs1.case.case_id == obs2.case.case_id


def test_determinism_different_seed():
    env1 = ShopopsEnvironment(debug_mode=True)
    env2 = ShopopsEnvironment(debug_mode=True)

    env1.reset(seed=42, tier="medium", split="train")
    env2.reset(seed=43, tier="medium", split="train")

    snap1 = _snapshot_episode(env1)
    snap2 = _snapshot_episode(env2)

    assert snap1 != snap2


def test_reproducible_trajectory():
    def run_episode(seed: int) -> list[float]:
        env = ShopopsEnvironment(debug_mode=True)
        obs = env.reset(seed=seed, tier="easy", split="train")
        rewards = []
        while True:
            case = env._cases[env._case_index]
            action = env._expected_action(case)
            obs = env.step(action)
            rewards.append(float(obs.reward or 0.0))
            if obs.done:
                break
        return rewards

    r1 = run_episode(101)
    r2 = run_episode(101)
    logger.info("trajectory_rewards=%s", r1)
    assert r1 == r2
