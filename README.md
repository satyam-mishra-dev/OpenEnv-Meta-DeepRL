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

## Inference Script (Hackathon Compliance)

The repo includes `inference.py` at the project root. It uses the OpenAI client
and emits strict `[START]`, `[STEP]`, `[END]` logs.

Required environment variables:
- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`
- `ENV_URL` (optional, defaults to `http://localhost:8000`)

Example:
```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o"
export HF_TOKEN="<your_key>"
export ENV_URL="http://localhost:8000"
python inference.py
```

## Submission Checklist

Run these before submitting:

1. **HF Space ping**  
   Confirm your Space responds:  
   `curl -s -o /dev/null -w "%{http_code}" -X POST "$PING_URL/reset"` → `200`

2. **Docker build**  
   `docker build -t shopops-env:latest -f server/Dockerfile .`

3. **OpenEnv validate**  
   `openenv validate`

4. **Inference script**  
   `set -a; source .env; set +a; python inference.py`  
   Ensure `[START]`, `[STEP]`, `[END]` lines are emitted and the script exits cleanly.

5. **Graded tasks**  
   Run your 3+ tasks/graders and verify all scores are in `[0.0, 1.0]`.

### Validator Script

If provided by the hackathon, run:

```bash
./scripts/validate-submission.sh <ping_url> .
```

## Test Results

Latest scenario test report:

```
outputs/test_report.txt
outputs/test_report_full.txt
```

## Baseline Scores

Rule-based baseline policy on test split (total-seeds=200 → 40 test episodes).

| Tier | Model | Avg Final Score |
| --- | --- | --- |
| easy | baseline_policy | 15.7861 |
| medium | baseline_policy | 14.3358 |
| hard | baseline_policy | 9.0594 |

## Model Benchmarks (Inference Script)

Inference-based benchmarks using `inference.py` against the local server, `MAX_STEPS=20`, 10 seeds.

| Model | Avg Score | Success Rate | Avg Steps | Seeds |
| --- | --- | --- | --- | --- |
| gpt-4o | 0.2825 | 100.0% | 20.0 | 10 |
| gpt-4.1 | 0.2825 | 100.0% | 20.0 | 10 |
| gpt-4.1-mini | 0.2825 | 100.0% | 20.0 | 10 |
| gpt-4o-mini | 0.2825 | 100.0% | 20.0 | 10 |

Score is computed as average reward per step (`sum(rewards) / MAX_STEPS`), since the HTTP API does not expose `episode_summary`.

### Reproduce Benchmarks

These steps reproduce all metrics above on any machine with the repo:

1. **Install dependencies**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r server/requirements.txt
pip install -e .
```

2. **Start the environment server**
```bash
PORT=8000 python -m shopOps.server.app
```

3. **Set required env vars**
```bash
export API_BASE_URL="https://api.openai.com/v1"
export HF_TOKEN="<your_api_key>"
export ENV_URL="http://localhost:8000"
```

4. **Run the benchmark script**
```bash
cd shopOps
BENCH_MODELS="gpt-4o,gpt-4.1,gpt-4.1-mini,gpt-4o-mini" \\
BENCH_SEEDS="1,2,3,4,5,6,7,8,9,10" \\
python scripts/benchmark_models.py
```

The script prints a markdown table that matches the benchmark table above.

## Building the Docker Image

```bash
docker build -t shopOps-env:latest -f server/Dockerfile .
```

## Running Locally

```bash
uvicorn shopOps.server.app:app --reload --host 0.0.0.0 --port 8000
```

Or:

```bash
python -m shopOps.server.app
```

## Deploying to Hugging Face Spaces

```bash
openenv push
```

## Project Structure

```
shopOps/
├── openenv.yaml           # OpenEnv manifest
├── client.py              # ShopOpsEnv client
├── models.py              # Action/observation models
├── eval.py                # Baseline evaluation runner
└── server/
    ├── shopOps_environment.py  # Environment logic
    ├── app.py                  # FastAPI server
    └── Dockerfile
```
