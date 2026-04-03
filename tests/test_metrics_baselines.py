# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging

from shopOps.eval import aggregate_results, run_episode
from shopOps.models import ActionType, ShopopsAction
from shopOps.server.shopOps_environment import ShopopsEnvironment

logger = logging.getLogger(__name__)


def test_episode_summary_schema():
    env = ShopopsEnvironment(debug_mode=True)
    obs = env.reset(seed=7, tier="easy", split="train")
    while True:
        case = env._cases[env._case_index]
        action = env._expected_action(case)
        obs = env.step(action)
        if obs.done:
            summary = obs.metadata.get("episode_summary", {})
            logger.info("episode_summary=%s", summary)
            assert "final_score" in summary
            assert "cases_completed" in summary
            assert "success_rate_by_case_type" in summary
            assert "adversarial_case_count" in summary
            break


def test_eval_aggregate_metrics():
    results = [
        run_episode(seed=1, tier="easy", split="train", debug_mode=True),
        run_episode(seed=2, tier="easy", split="train", debug_mode=True),
    ]
    summary = aggregate_results(results)
    logger.info("aggregate_summary=%s", summary)
    assert "avg_final_score" in summary
    assert "avg_total_reward" in summary
    assert "avg_success_rate_by_case_type" in summary


def test_baseline_separation_easy_vs_hard():
    def run_policy(seed: int, tier: str, action: ShopopsAction) -> float:
        env = ShopopsEnvironment(debug_mode=True)
        obs = env.reset(seed=seed, tier=tier, split="train")
        total_reward = 0.0
        while True:
            obs = env.step(action)
            total_reward += float(obs.reward or 0.0)
            if obs.done:
                break
        return total_reward

    seeds = [10, 11, 12]
    easy_refund = sum(
        run_policy(seed, "easy", ShopopsAction(action_type=ActionType.REFUND, refund_amount_usd=10.0))
        for seed in seeds
    ) / len(seeds)
    hard_refund = sum(
        run_policy(seed, "hard", ShopopsAction(action_type=ActionType.REFUND, refund_amount_usd=10.0))
        for seed in seeds
    ) / len(seeds)

    logger.info("baseline_easy_refund=%s", easy_refund)
    logger.info("baseline_hard_refund=%s", hard_refund)

    assert easy_refund > hard_refund
