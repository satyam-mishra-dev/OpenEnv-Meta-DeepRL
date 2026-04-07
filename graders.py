# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Graders for the ShopOps OpenEnv tasks.

Each grader receives the full episode trajectory (list of step dicts, each
containing at least a "reward" key) and returns a normalised score strictly
in (0.0, 1.0) — i.e. scores are always > 0 and < 1 as required by the spec.
"""

from __future__ import annotations

from typing import Any, Dict, List

# Must match MAX_CASES in shopOps_environment.py.
# Used as the theoretical maximum total reward (1.0 per step × 20 steps).
_MAX_STEPS = 20

# Scores must be strictly between 0 and 1 (exclusive) per the OpenEnv spec.
_SCORE_MIN = 1e-9
_SCORE_MAX = 1.0 - 1e-9


class ScoreGrader:
    """
    Trajectory-quality grader for ShopOps.

    Scoring formula
    ---------------
    Per-step rewards are in the range [-1.0, 1.0]:
      * +0.0 – 1.0  for valid actions  (weighted correctness + efficiency + priority)
      * -1.0         for invalid actions

    The grader sums all rewards and divides by the theoretical maximum
    (_MAX_STEPS × 1.0 = 20.0), then clamps the result to the open interval
    (_SCORE_MIN, _SCORE_MAX) so the returned value is always strictly between
    0 and 1 (never exactly 0.0 or 1.0):

        score = clamp(sum(rewards) / _MAX_STEPS, _SCORE_MIN, _SCORE_MAX)

    This means:
      * A perfect agent that scores 1.0 every step → score ≈ 1.0 (capped at _SCORE_MAX)
      * An agent that always rejects correctly     → score ≈ 0.45–0.75 (task-dependent)
      * An agent that triggers the invalid limit   → score ≈ 0.0 (floored at _SCORE_MIN)

    The grader is deterministic: identical trajectories always yield the same score.
    """

    def grade(self, trajectory: List[Dict[str, Any]]) -> float:
        """
        Score a completed episode.

        Args:
            trajectory: List of step dicts.  Each dict must contain a "reward"
                        key whose value is a float (or None, treated as 0.0).

        Returns:
            Normalised score strictly in (0.0, 1.0).
        """
        if not trajectory:
            return _SCORE_MIN

        total_reward = sum(float(step.get("reward") or 0.0) for step in trajectory)
        score = total_reward / _MAX_STEPS
        return float(max(_SCORE_MIN, min(_SCORE_MAX, score)))
