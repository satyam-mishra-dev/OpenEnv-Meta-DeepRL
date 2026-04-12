from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
import sys
from typing import Callable, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shopOps.eval import TASKS, baseline_policy
from shopOps.models import ActionType, ShopopsAction, ShopopsObservation
from shopOps.server.shopOps_environment import ShopopsEnvironment


OUTPUT_PATH = Path("outputs/evals/meta_review_eval.json")
NORMALIZATION_CAPS = {
    "refund_policy_recovery": 2.0,
    "sla_queue_juggle": 5.4,
    "fraud_stockout_cascade": 7.6,
}


@dataclass
class EpisodeStats:
    total_reward: float
    normalized_reward: float
    final_score: float
    termination_reason: str
    steps: int


PolicyFn = Callable[[ShopopsObservation], ShopopsAction]


def _run_policy(task: str, seed: int, policy: PolicyFn) -> EpisodeStats:
    env = ShopopsEnvironment(debug_mode=True)
    obs = env.reset(seed=seed, task=task)
    total_reward = 0.0
    steps = 0

    while True:
        action = policy(obs)
        obs = env.step(action)
        total_reward += float(obs.reward or 0.0)
        steps += 1
        if obs.done:
            summary = obs.metadata.get("episode_summary", {})
            return EpisodeStats(
                total_reward=round(total_reward, 4),
                normalized_reward=round(
                    max(0.0, min(1.0, total_reward / NORMALIZATION_CAPS[task])),
                    4,
                ),
                final_score=float(summary.get("final_score", 0.0)),
                termination_reason=str(obs.metadata.get("termination_reason") or "unknown"),
                steps=steps,
            )


def _close_only(_: ShopopsObservation) -> ShopopsAction:
    return ShopopsAction(action_type=ActionType.CLOSE_CASE)


def _inspect_only(_: ShopopsObservation) -> ShopopsAction:
    return ShopopsAction(action_type=ActionType.INSPECT_ORDER)


def _switch_only(obs: ShopopsObservation) -> ShopopsAction:
    for item in obs.queue:
        if item.case_id != obs.active_case.case_id:
            return ShopopsAction(action_type=ActionType.SWITCH_CASE, case_id=item.case_id)
    return ShopopsAction(action_type=ActionType.CLOSE_CASE)


def _snapshot_signature(task: str, seed: int) -> str:
    env = ShopopsEnvironment(debug_mode=True)
    env.reset(seed=seed, task=task)
    signature = [
        (
            case.case_id,
            round(case.order_value_usd, 2),
            case.priority.value,
            case.order_status.value,
            case.fraud_signal.value,
            case.sla_minutes,
        )
        for case in env._cases
    ]
    return json.dumps(signature)


def _policy_summary(task: str, policy_name: str, policy: PolicyFn, seeds: range) -> Dict[str, object]:
    runs = [_run_policy(task, seed, policy) for seed in seeds]
    return {
        "policy": policy_name,
        "avg_total_reward": round(mean(run.total_reward for run in runs), 4),
        "avg_normalized_reward": round(mean(run.normalized_reward for run in runs), 4),
        "avg_final_score": round(mean(run.final_score for run in runs), 4),
        "termination_reasons": sorted({run.termination_reason for run in runs}),
        "avg_steps": round(mean(run.steps for run in runs), 2),
    }


def _risk_flags(task: str, diversity_count: int, close_summary: Dict[str, object], baseline_summary: Dict[str, object]) -> List[str]:
    flags: List[str] = []
    if diversity_count <= 1:
        flags.append("seed_diversity_missing")
    if float(close_summary["avg_final_score"]) >= 0.5:
        flags.append("terminal_score_too_high_for_close_only")
    if float(close_summary["avg_normalized_reward"]) >= 0.15:
        flags.append("reward_too_high_for_close_only")
    if float(baseline_summary["avg_normalized_reward"]) >= 0.9:
        flags.append("baseline_too_close_to_ceiling")
    return flags


def main() -> None:
    seeds = range(1, 6)
    report: Dict[str, object] = {"tasks": {}}

    for task in TASKS:
        diversity_signatures = {_snapshot_signature(task, seed) for seed in seeds}
        baseline_summary = _policy_summary(task, "baseline", baseline_policy, seeds)
        close_summary = _policy_summary(task, "close_only", _close_only, seeds)
        inspect_summary = _policy_summary(task, "inspect_only", _inspect_only, seeds)
        switch_summary = _policy_summary(task, "switch_only", _switch_only, seeds)

        report["tasks"][task] = {
            "unique_seed_snapshots": len(diversity_signatures),
            "baseline": baseline_summary,
            "close_only": close_summary,
            "inspect_only": inspect_summary,
            "switch_only": switch_summary,
            "flags": _risk_flags(task, len(diversity_signatures), close_summary, baseline_summary),
        }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2))
    print(f"Wrote {OUTPUT_PATH}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
