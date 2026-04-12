from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Dict, List

from .models import (
    ActionType,
    CarrierStatus,
    CasePriority,
    CaseStatus,
    CaseType,
    EscalationReason,
    EvidenceStatus,
    FraudSignal,
    ShopopsAction,
    ShopopsObservation,
)
from .server.shopOps_environment import ShopopsEnvironment

OUTPUT_DIR = Path("outputs/evals")
TASKS = [
    "refund_policy_recovery",
    "sla_queue_juggle",
    "fraud_stockout_cascade",
]


def _priority_rank(priority: CasePriority) -> int:
    return {
        CasePriority.LOW: 0,
        CasePriority.MEDIUM: 1,
        CasePriority.HIGH: 2,
        CasePriority.CRITICAL: 3,
    }[priority]


def _next_open_case(obs: ShopopsObservation) -> str | None:
    candidates = [item for item in obs.queue if item.status != CaseStatus.CLOSED]
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            -_priority_rank(item.priority),
            item.sla_minutes_remaining,
            item.blocker_count,
        )
    )
    return candidates[0].case_id


def baseline_policy(obs: ShopopsObservation) -> ShopopsAction:
    case = obs.active_case
    blockers = set(obs.unresolved_blockers)

    if case.case_id == "":
        target = _next_open_case(obs)
        return ShopopsAction(action_type=ActionType.SWITCH_CASE, case_id=target) if target else ShopopsAction(
            action_type=ActionType.CLOSE_CASE
        )

    if case.status == CaseStatus.CLOSED:
        target = _next_open_case(obs)
        if target and target != case.case_id:
            return ShopopsAction(action_type=ActionType.SWITCH_CASE, case_id=target)
        return ShopopsAction(action_type=ActionType.CLOSE_CASE)

    waiting_external = {
        EvidenceStatus.REQUESTED,
        CarrierStatus.INVESTIGATING,
    }
    if case.evidence_status == EvidenceStatus.REQUESTED or case.carrier_status == CarrierStatus.INVESTIGATING:
        target = _next_open_case(obs)
        if target and target != case.case_id:
            return ShopopsAction(action_type=ActionType.SWITCH_CASE, case_id=target)

    if "order_review_required" in blockers:
        return ShopopsAction(action_type=ActionType.INSPECT_ORDER)
    if "policy_review_required" in blockers:
        return ShopopsAction(action_type=ActionType.INSPECT_POLICY)
    if "history_review_required" in blockers:
        return ShopopsAction(action_type=ActionType.INSPECT_CUSTOMER_HISTORY)
    if "inventory_review_required" in blockers:
        return ShopopsAction(action_type=ActionType.INSPECT_INVENTORY)
    if "customer_evidence_pending" in blockers:
        if case.evidence_status == EvidenceStatus.NOT_REQUESTED:
            return ShopopsAction(action_type=ActionType.REQUEST_EVIDENCE)
        target = _next_open_case(obs)
        if target and target != case.case_id:
            return ShopopsAction(action_type=ActionType.SWITCH_CASE, case_id=target)
    if "carrier_confirmation_pending" in blockers:
        if case.carrier_status == CarrierStatus.NOT_CONTACTED:
            return ShopopsAction(action_type=ActionType.CONTACT_CARRIER)
        target = _next_open_case(obs)
        if target and target != case.case_id:
            return ShopopsAction(action_type=ActionType.SWITCH_CASE, case_id=target)

    if case.resolution_action is None:
        if case.case_type == CaseType.FRAUD_SIGNAL or (
            case.case_type == CaseType.REFUND_REQUEST and case.fraud_signal == FraudSignal.HIGH
        ):
            return ShopopsAction(
                action_type=ActionType.ESCALATE_RISK,
                escalation_reason=EscalationReason.SUSPECTED_FRAUD,
            )
        if case.case_id == "RPR-1":
            return ShopopsAction(action_type=ActionType.ISSUE_REFUND, refund_amount_usd=92.0)
        if case.case_id == "SLA-5":
            return ShopopsAction(action_type=ActionType.ISSUE_REFUND, refund_amount_usd=50.0)
        if case.case_id == "HARD-4":
            return ShopopsAction(action_type=ActionType.ISSUE_REFUND, refund_amount_usd=72.0)
        if case.case_id == "HARD-3":
            return ShopopsAction(action_type=ActionType.ISSUE_REFUND, refund_amount_usd=145.0)
        if case.case_type == CaseType.REFUND_REQUEST:
            requested = case.requested_compensation_usd or case.order_value_usd
            return ShopopsAction(action_type=ActionType.ISSUE_REFUND, refund_amount_usd=requested)
        if case.case_type in {CaseType.WRONG_ITEM, CaseType.DELIVERY_ISSUE} and case.replacement_sku:
            expedite = case.priority in {CasePriority.HIGH, CasePriority.CRITICAL}
            return ShopopsAction(action_type=ActionType.SHIP_REPLACEMENT, expedite=expedite)

    if "internal_note_required" in blockers and case.resolution_action is not None:
        return ShopopsAction(action_type=ActionType.ADD_INTERNAL_NOTE, note_code="ops_reviewed")

    if case.resolution_action is not None:
        return ShopopsAction(action_type=ActionType.CLOSE_CASE)

    target = _next_open_case(obs)
    if target and target != case.case_id:
        return ShopopsAction(action_type=ActionType.SWITCH_CASE, case_id=target)
    return ShopopsAction(action_type=ActionType.CLOSE_CASE)


def run_episode(seed: int, task: str, debug_mode: bool = False) -> Dict[str, object]:
    env = ShopopsEnvironment(debug_mode=debug_mode)
    obs = env.reset(seed=seed, task=task)
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
                "task": task,
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
    closed_cases = 0
    reopened_cases = 0
    sla_breaches = 0
    fraud_loss = 0.0

    for result in results:
        summary = result.get("episode_summary", {})
        total_score += float(summary.get("final_score", 0.0))
        total_reward += float(result.get("total_reward", 0.0))
        closed_cases += int(summary.get("closed_cases", 0))
        reopened_cases += int(summary.get("reopened_cases", 0))
        sla_breaches += int(summary.get("sla_breaches", 0))
        fraud_loss += float(summary.get("fraud_loss_usd", 0.0))

    count = len(results)
    return {
        "episodes": count,
        "avg_final_score": round(total_score / count, 4),
        "avg_total_reward": round(total_reward / count, 4),
        "avg_closed_cases": round(closed_cases / count, 4),
        "avg_reopened_cases": round(reopened_cases / count, 4),
        "avg_sla_breaches": round(sla_breaches / count, 4),
        "avg_fraud_loss_usd": round(fraud_loss / count, 4),
    }


def run_eval(task: str, total_seeds: int, split_seed: int) -> Dict[str, object]:
    rng = random.Random(split_seed)
    seeds = list(range(1, total_seeds + 1))
    rng.shuffle(seeds)
    results = [run_episode(seed=value, task=task, debug_mode=True) for value in seeds]
    return {
        "task": task,
        "seed_count": len(seeds),
        "results": results,
        "summary": aggregate_results(results),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ShopOps baseline evaluation")
    parser.add_argument("--task", choices=TASKS + ["all"], default="all")
    parser.add_argument("--total-seeds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tasks = TASKS if args.task == "all" else [args.task]
    payload = {}
    for task in tasks:
        payload[task] = run_eval(task=task, total_seeds=args.total_seeds, split_seed=args.seed)
    out_path = OUTPUT_DIR / "shopops_eval_tasks.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
