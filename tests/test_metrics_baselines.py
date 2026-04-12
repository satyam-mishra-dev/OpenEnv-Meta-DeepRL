from __future__ import annotations

from shopOps.eval import TASKS, aggregate_results, run_episode


def test_episode_summary_schema() -> None:
    result = run_episode(seed=1, task="refund_policy_recovery", debug_mode=True)
    summary = result["episode_summary"]
    assert "final_score" in summary
    assert "closed_cases" in summary
    assert "reopened_cases" in summary
    assert "fraud_loss_usd" in summary


def test_eval_aggregate_metrics() -> None:
    results = [
        run_episode(seed=1, task="refund_policy_recovery", debug_mode=True),
        run_episode(seed=2, task="refund_policy_recovery", debug_mode=True),
    ]
    summary = aggregate_results(results)
    assert "avg_final_score" in summary
    assert "avg_total_reward" in summary
    assert "avg_closed_cases" in summary


def test_baseline_scores_are_monotonic_by_difficulty_seed_1() -> None:
    scores = [
        run_episode(seed=1, task=task, debug_mode=True)["episode_summary"]["final_score"]
        for task in TASKS
    ]
    assert scores[0] >= scores[1] >= scores[2]
