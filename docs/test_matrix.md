# ShopOps vNext Test Matrix

This document formalizes the review-and-test framework into a matrix format suitable for implementation and CI tracking.

## Legend
- **Priority**: P0 (must), P1 (should), P2 (nice)
- **Layer**: A (Schema), B (Determinism), C (Validation), D (Resources), E (Reward), F (Hard Tier), G (Episode), H (Metrics), S (Stress)

---

## A. Schema & Contract Tests

| Test ID | Priority | Setup | Expected | Failure Mode |
| --- | --- | --- | --- | --- |
| A1 | P0 | `reset(seed=123)` | Observation contains only documented fields (case data, indices, episode_id, tier, resources) | Missing field, extra ground-truth leak, wrong types |
| A2 | P0 | Valid action then invalid action | Valid passes; invalid rejected/penalized per policy | Invalid enum accepted; missing fields silently accepted |
| A3 | P0 | Step one case | `info` includes reward breakdown + correctness + resource usage; termination reason only on done | Info unstable or leaks expected_action in normal mode |

## B. Determinism Tests

| Test ID | Priority | Setup | Expected | Failure Mode |
| --- | --- | --- | --- | --- |
| B1 | P0 | `reset(seed=42)` twice | Identical 20-case queue, hidden fields, observations | Any content differs |
| B2 | P1 | Compare seed 42 vs 43 | At least one meaningful case difference | Episodes identical or trivial diff |
| B3 | P0 | Fixed policy, same seed, run twice | Identical step rewards and final metrics | Score drift or step divergence |

## C. Validation & Safety Tests

| Test ID | Priority | Setup | Expected | Failure Mode |
| --- | --- | --- | --- | --- |
| C1 | P0 | `action_type="fly_to_moon"` | Strong penalty + invalid count | Action accepted |
| C2 | P0 | Refund > order value | Invalid action handling triggers | Silent clip or acceptance |
| C3 | P0 | Refund without amount | Invalid action handling triggers | Guessed refund or accepted |
| C4 | P1 | Escalate without reason | Invalid action handling triggers | Escalation allowed |
| C5 | P0 | Three invalid actions | Episode terminates with reason | No termination or wrong reason |

## D. Resource Accounting Tests

| Test ID | Priority | Setup | Expected | Failure Mode |
| --- | --- | --- | --- | --- |
| D1 | P0 | Refund $120 | Budget decreases by $120 | Budget mismatch |
| D2 | P0 | Replace / Escalate | Time decreases per action cost table | Time drift |
| D3 | P0 | Spend past budget | Terminate with `budget_exhausted` | Continues w/ negative budget |
| D4 | P0 | Spend past time | Terminate with `time_exhausted` | Continues w/ negative time |
| D5 | P1 | Sum 5 actions externally | Internal resources match external sum | Drift after multiple steps |

## E. Reward Logic Tests

| Test ID | Priority | Setup | Expected | Failure Mode |
| --- | --- | --- | --- | --- |
| E1 | P0 | Compare correct vs wrong action | Correct action > wrong action | Wrong action scores similarly |
| E2 | P1 | Two correct actions, diff cost | Lower cost wins | Cost ignored |
| E3 | P1 | High-priority vs low-priority | Priority bonus applied | No priority effect |
| E4 | P2 | Slightly worse action | Reward drops smoothly | Binary reward only |
| E5 | P0 | Inspect breakdown | Reward equals weighted sum | Mismatch between components and reward |

## F. Hard-Tier Tests

| Test ID | Priority | Setup | Expected | Failure Mode |
| --- | --- | --- | --- | --- |
| F1 | P0 | Compare easy vs hard | Hard hides/buckets more fields | Hard same as easy |
| F2 | P0 | Hard test split seeds | Adversarial cases only in hard/test | Adversarial leaks to train/easy |
| F3 | P1 | Run simple heuristic | Hard score noticeably lower than easy | No degradation |
| F4 | P1 | Investigation action (if enabled) | Hidden info revealed w/ time cost | No effect or free reveal |
| F5 | P0 | Normal mode | No expected_action leakage | Ground-truth leaks |

## G. Episode-Level Behavior

| Test ID | Priority | Setup | Expected | Failure Mode |
| --- | --- | --- | --- | --- |
| G1 | P0 | Perfect policy on easy seed | High final score, low invalids | Score inconsistent |
| G2 | P1 | Random policy | Low score vs baseline | Random too strong |
| G3 | P1 | Always refund | Penalized on cost/prioritization | Competitive score |
| G4 | P1 | Always escalate | Penalized on time/accuracy | Competitive score |
| G5 | P0 | Compare baselines | Clear ordering: random < greedy < heuristic | Scores cluster |

## H. Metrics & Reporting

| Test ID | Priority | Setup | Expected | Failure Mode |
| --- | --- | --- | --- | --- |
| H1 | P0 | Eval across tiers | Per-tier averages reported | Single aggregate only |
| H2 | P0 | Eval across case types | Success rates per case type | Missing breakdown |
| H3 | P0 | Eval smoke | JSON artifact written | Missing or malformed JSON |
| H4 | P1 | 10+ episodes | Mean + variance reported | Only a single score |
| H5 | P0 | Hard validation set | Stable scores across reruns | Drifting scores |

## S. Architecture Stress Tests

| Test ID | Priority | Setup | Expected | Failure Mode |
| --- | --- | --- | --- | --- |
| S1 | P1 | Heuristics sweep | Heuristics underperform strong baseline | Heuristics too strong |
| S2 | P2 | Pattern exploitation attempt | No easy shortcut (seed/order) | Exploitable pattern |
| S3 | P2 | Investigation effectiveness | Investigate improves score when needed | Mechanic useless |
| S4 | P1 | Medium vs hard | Hard materially harder than medium | No difficulty gap |

---

## Pass/Fail Targets

- Determinism: 100% reproducible with same seed
- Invalid action handling: 100% rejection/penalty as designed
- Baseline separation: 15–25% gap between random and strong baseline
- Hard-tier degradation: noticeable drop vs easy
- Reward consistency: 100% formula match in tests
- Episode termination: correct end on 20 cases/time/budget/invalids
- No leakage: 0 ground-truth fields in normal mode
