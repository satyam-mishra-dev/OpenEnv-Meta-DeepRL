# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Baseline evaluation runner for ShopOps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Dict, List

from .models import (
    ActionType,
    CustomerTier,
    EscalationReason,
    FraudSignal,
    ShopopsAction,
    ShopopsObservation,
)
from .server.shopOps_environment import ShopopsEnvironment

OUTPUT_DIR = Path("outputs/evals")
DEFAULT_SPLIT_SEED = 1337


def generate_seed_split(total: int, train_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    seeds = list(range(1, total + 1))
    rng.shuffle(seeds)
    split_idx = int(total * train_ratio)
    return seeds[:split_idx], seeds[split_idx:]


def _bucket_value_to_estimate(bucket: str | None) -> float:
    mapping = {
        "low": 40.0,
        "medium": 150.0,
        "high": 350.0,
        "very_high": 700.0,
    }
    return mapping.get(bucket or "medium", 150.0)


def baseline_policy(obs: ShopopsObservation) -> ShopopsAction:
    case = obs.case
    order_value = case.order_value_usd or _bucket_value_to_estimate(case.order_value_bucket)
    return_window_open = case.return_window_open
    if return_window_open is None:
        return_window_open = case.order_age_bucket in {"recent", "normal"}

    if case.case_type.value == "fraud_signal":
        if case.fraud_signal == FraudSignal.HIGH:
            return ShopopsAction(action_type=ActionType.ESCALATE, escalation_reason=EscalationReason.SUSPECTED_FRAUD)
        if case.fraud_signal == FraudSignal.MEDIUM:
            return ShopopsAction(action_type=ActionType.REJECT)
        return ShopopsAction(action_type=ActionType.REFUND, refund_amount_usd=round(order_value * 0.5, 2))

    if case.case_type.value == "refund_request":
        if not return_window_open:
            return ShopopsAction(action_type=ActionType.REJECT)
        if case.fraud_signal == FraudSignal.HIGH:
            return ShopopsAction(action_type=ActionType.ESCALATE, escalation_reason=EscalationReason.SUSPECTED_FRAUD)
        if order_value >= 500:
            return ShopopsAction(action_type=ActionType.ESCALATE, escalation_reason=EscalationReason.HIGH_VALUE)
        return ShopopsAction(action_type=ActionType.REFUND, refund_amount_usd=round(order_value, 2))

    if case.case_type.value == "delivery_issue":
        if case.delivery_status and case.delivery_status.value == "lost":
            return ShopopsAction(action_type=ActionType.REPLACE, replacement_expedite=case.issue_severity.value == "high")
        if case.delivery_status and case.delivery_status.value == "delayed":
            return ShopopsAction(action_type=ActionType.REFUND, refund_amount_usd=round(order_value * 0.2, 2))
        if case.delivery_status and case.delivery_status.value == "in_transit":
            return ShopopsAction(action_type=ActionType.ESCALATE, escalation_reason=EscalationReason.POLICY_EXCEPTION)
        return ShopopsAction(action_type=ActionType.REJECT)

    if case.case_type.value == "wrong_item":
        if case.evidence_provided:
            return ShopopsAction(action_type=ActionType.REPLACE, replacement_expedite=case.issue_severity.value == "high")
        if case.customer_tier in {CustomerTier.GOLD, CustomerTier.PLATINUM}:
            return ShopopsAction(action_type=ActionType.REPLACE)
        return ShopopsAction(action_type=ActionType.REJECT)

    return ShopopsAction(action_type=ActionType.REJECT)


def run_episode(seed: int, tier: str, debug_mode: bool = False) -> Dict[str, object]:
    env = ShopopsEnvironment(debug_mode=debug_mode)
    obs = env.reset(seed=seed, tier=tier)
    total_reward = 0.0
    steps = 0

    while True:
        action = baseline_policy(obs)
        obs = env.step(action)
        total_reward += float(obs.reward or 0.0)
        steps += 1
        if obs.done:
            summary = obs.metadata.get("episode_summary", {})
            return {
                "seed": seed,
                "tier": tier,
                "steps": steps,
                "total_reward": round(total_reward, 4),
                "termination_reason": obs.metadata.get("termination_reason"),
                "episode_summary": summary,
            }


def aggregate_results(results: List[Dict[str, object]]) -> Dict[str, object]:
    if not results:
        return {}
    total_score = 0.0
    total_reward = 0.0
    termination_counts: Dict[str, int] = {}
    success_rate_totals: Dict[str, float] = {}
    success_rate_counts: Dict[str, int] = {}

    for result in results:
        summary = result.get("episode_summary", {})
        total_score += float(summary.get("final_score", 0.0))
        total_reward += float(result.get("total_reward", 0.0))
        for case_type, rate in (summary.get("success_rate_by_case_type", {}) or {}).items():
            success_rate_totals[case_type] = success_rate_totals.get(case_type, 0.0) + float(rate)
            success_rate_counts[case_type] = success_rate_counts.get(case_type, 0) + 1
        reason = result.get("termination_reason") or "unknown"
        termination_counts[reason] = termination_counts.get(reason, 0) + 1

    count = len(results)
    avg_success_rate = {
        case_type: round(success_rate_totals[case_type] / success_rate_counts[case_type], 4)
        for case_type in success_rate_totals
    }
    return {
        "episodes": count,
        "avg_final_score": round(total_score / count, 4),
        "avg_total_reward": round(total_reward / count, 4),
        "termination_reasons": termination_counts,
        "avg_success_rate_by_case_type": avg_success_rate,
    }


def run_eval(split: str, tier: str, total_seeds: int, train_ratio: float, seed: int) -> Dict[str, object]:
    train_seeds, test_seeds = generate_seed_split(total_seeds, train_ratio, seed)
    seeds = train_seeds if split == "train" else test_seeds

    results = [run_episode(seed=value, tier=tier, debug_mode=True) for value in seeds]
    return {
        "split": split,
        "tier": tier,
        "seed_count": len(seeds),
        "results": results,
        "summary": aggregate_results(results),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ShopOps baseline evaluation")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--tier", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--total-seeds", type=int, default=200)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = run_eval(args.split, args.tier, args.total_seeds, args.train_ratio, args.seed)
    out_path = OUTPUT_DIR / f"shopops_eval_{args.split}_{args.tier}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
