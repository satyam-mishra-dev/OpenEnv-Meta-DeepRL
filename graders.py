from __future__ import annotations

from typing import Any, Dict, List


def _grade_with_cap(trajectory: List[Dict[str, Any]], max_total_reward: float) -> float:
    if not trajectory or max_total_reward <= 0:
        return 0.0
    total_reward = sum(float(step.get("reward") or 0.0) for step in trajectory)
    score = total_reward / max_total_reward
    return float(max(0.0, min(1.0, score)))


class RefundPolicyRecoveryGrader:
    def grade(self, trajectory: List[Dict[str, Any]]) -> float:
        return _grade_with_cap(trajectory, max_total_reward=1.7)


class SlaQueueJuggleGrader:
    def grade(self, trajectory: List[Dict[str, Any]]) -> float:
        return _grade_with_cap(trajectory, max_total_reward=5.4)


class FraudStockoutCascadeGrader:
    def grade(self, trajectory: List[Dict[str, Any]]) -> float:
        return _grade_with_cap(trajectory, max_total_reward=7.6)
