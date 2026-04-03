---
title: ShopOps Environment Server
emoji: 🎥
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# ShopOps Environment

ShopOps is a realistic OpenEnv environment that simulates daily operations of an
E‑commerce support and operations team. Each episode represents a 20‑case workday
where an agent must choose actions (refund, replace, escalate, reject), manage
limited resources (time + budget), and prioritize urgent or high‑value cases.

## Quick Start

```python
from shopOps import ShopopsAction, ShopopsEnv, ActionType, EscalationReason

try:
    env = ShopopsEnv.from_docker_image("shopOps-env:latest")
    result = env.reset(seed=42, tier="medium")

    # Take a step
    action = ShopopsAction(
        action_type=ActionType.REFUND,
        refund_amount_usd=50.0,
    )
    result = env.step(action)
    print(result.observation.case.case_type)
    print(result.observation.metadata["reward_breakdown"])

finally:
    env.close()
```

## Environment Overview

### Episode
- Fixed length: **20 cases** per episode
- Done conditions:
  - All 20 cases processed
  - Time or budget exhausted
  - 3 invalid actions

### Resources
- Time limit: **480 minutes**
- Budget limit: **$2,000**

### Difficulty Tiers
- `easy`: full detail
- `medium`: partial detail
- `hard`: bucketed values, hidden fields, and adversarial cases (test split only)

## Action Space

**ShopopsAction** fields:
- `action_type`: `refund | replace | escalate | reject`
- `refund_amount_usd` (refund only)
- `replacement_expedite` (replace only)
- `escalation_reason` (escalate only)

Invalid actions are rejected and count toward the **3‑strike limit**.

### Action Costs (Explicit)

```
ACTION_COSTS = {
  refund:  time=2,  cost=refund_amount
  replace: time=5,  cost=min(order_value*0.6, 300) (+2 min, +$20 if expedited)
  escalate: time=10, cost=0
  reject:  time=1,  cost=0
}
```

## Observations

Each observation includes:
- `case`: structured case data (partially observable by tier)
- `resources`: time/budget remaining
- `case_index`, `step_index`, `episode_id`, `tier`
- `metadata`: per‑step info and reward breakdown

### Partial Observability Example
Instead of `fraud_score`, the agent receives:
- `fraud_signal`: `low | medium | high`

## Rewards

Dense reward per step:
```
reward = 0.6 * correctness
       + 0.25 * (cost_efficiency * correctness)
       + 0.15 * prioritization
```

Reward breakdown is returned in `observation.metadata["reward_breakdown"]`.
Per-step rewards are normalized to roughly `[-1, 1]` (invalid actions yield `-1`).

## Info & Debugging

The environment returns rich per‑step info via `observation.metadata`:
- reward breakdown
- correctness flag
- budget/time used
- cumulative score
- invalid action count

**Expected action is NOT exposed** unless `debug_mode=True` or in offline eval.

## Evaluation + Metrics

A baseline evaluation runner is provided:

```bash
python -m shopOps.eval --split test --tier medium --total-seeds 200
```

Outputs JSON metrics to `outputs/evals/` with:
- final score
- per‑tier score
- success rate per case type

### Train/Test Split
- Deterministic split (80/20)
- Controlled via `--seed` in the eval runner
- Hard-tier validation seeds available via `--validation`

## Test Results

Latest scenario test report:

```
outputs/test_report.txt
```

## Building the Docker Image

```bash
docker build -t shopOps-env:latest -f server/Dockerfile .
```

## Running Locally

```bash
uvicorn server.app:app --reload
```

## Deploying to Hugging Face Spaces

```bash
openenv push
```

## Project Structure

```
shopOps/
├── openenv.yaml           # OpenEnv manifest (includes schema_version)
├── client.py              # ShopOpsEnv client
├── models.py              # Action/observation models
├── eval.py                # Baseline evaluation runner
└── server/
    ├── shopOps_environment.py  # Environment logic
    ├── app.py                  # FastAPI server
    └── Dockerfile
```
