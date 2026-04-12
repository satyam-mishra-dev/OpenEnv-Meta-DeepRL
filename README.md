---
title: ShopOps Environment Server
emoji: 🛒
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
---

# ShopOps 2.0

ShopOps is an OpenEnv environment for **real customer-operations work**. The agent is not picking a final label from a tiny action set anymore. It has to operate a support queue, inspect policies and customer history, manage scarce replacement inventory, wait for delayed carrier or evidence responses, and close cases without creating downstream damage.

This is designed to evaluate long-horizon business operations behavior:
- tool use instead of single-shot classification
- coupled state across multiple cases
- delayed consequences from premature closure
- real tradeoffs between SLA, budget, fraud loss, and stock availability

## Environment Overview

Each episode is a deterministic task scenario exposed through the standard OpenEnv API:
- `reset()` returns the first observation for the selected task
- `step(action)` applies one tool/action and returns observation, reward, done, and info
- `state` returns the current `episode_id` and `step_count`

The environment is implemented with typed Pydantic models and ships with:
- `openenv.yaml`
- deterministic graders for all tasks
- `inference.py` in the repo root
- a Dockerfile for local and Hugging Face Space deployment

## Action Space

`ShopopsAction` is a typed tool invocation with these fields:
- `action_type`
- `case_id`
- `refund_amount_usd`
- `expedite`
- `escalation_reason`
- `note_code`

Supported `action_type` values:
- `inspect_order`
- `inspect_policy`
- `inspect_inventory`
- `inspect_customer_history`
- `request_evidence`
- `contact_carrier`
- `issue_refund`
- `ship_replacement`
- `escalate_risk`
- `add_internal_note`
- `close_case`
- `switch_case`

Key constraints:
- `issue_refund` requires `refund_amount_usd`
- `ship_replacement` may set `expedite`
- `escalate_risk` requires `escalation_reason`
- `add_internal_note` requires `note_code`
- non-switch actions must target the active case

## Observation Space

Each `ShopopsObservation` contains:
- `active_case`: full working view of the active case
- `queue`: visible queue summary for all cases
- `latest_tool_result`: persistent result from the last action
- `resources`: time, budget, and inventory state
- `metrics`: resolved cases, reopened cases, SLA breaches, fraud loss, satisfaction, stockouts
- `unresolved_blockers`: blockers still preventing safe closure
- `current_task`, `difficulty`, `step_index`, `episode_id`, `env_schema_version`

The active case includes persistent tool-discovered summaries:
- `order_summary`
- `policy_summary`
- `history_summary`
- `inventory_summary`

This lets an agent build working memory from prior inspection actions instead of re-querying everything every step.

## Tasks

### 1. `refund_policy_recovery` — easy
Single-case recovery task.

The agent must:
- inspect order facts
- inspect policy
- choose a compliant partial refund
- add the required internal note
- close the case cleanly

### 2. `sla_queue_juggle` — medium
Five-case queue with mixed urgency.

The agent must:
- switch cases intentionally
- prioritize SLA-critical work
- inspect inventory and history where needed
- avoid wasting budget
- close all five cases

### 3. `fraud_stockout_cascade` — hard
Seven-case scenario with coupled consequences.

The agent must:
- preserve scarce inventory for the right case
- avoid refunding suspicious orders before evidence arrives
- handle fraud escalation correctly
- juggle delayed carrier/evidence events
- prevent reopen cascades and fraud loss

## Reward Design

Reward is dense over the trajectory and combines:
- information gain from useful inspections
- workflow progress from moving a case forward correctly
- business outcome from the quality of the chosen resolution

Undesirable behavior is penalized:
- invalid tool calls
- duplicate inspections
- unnecessary external requests
- refunds without required review
- premature closure that causes reopen or fraud loss

The terminal episode summary tracks:
- `final_score`
- `closed_cases`
- `reopened_cases`
- `sla_breaches`
- `fraud_loss_usd`
- `stockouts`
- `customer_satisfaction`

## Baseline

`shopOps.eval` contains a deterministic rule baseline that uses the same public observation space as an agent.

10-seed baseline results:

| Task | Avg final score | Avg total reward |
| --- | ---: | ---: |
| `refund_policy_recovery` | `0.9840` | `1.5920` |
| `sla_queue_juggle` | `0.9360` | `5.0384` |
| `fraud_stockout_cascade` | `0.9246` | `7.1421` |

Reproduce:

```bash
./venv/bin/python -m shopOps.eval --task all --total-seeds 10
```

## Inference Script

The required root-level `inference.py` uses the OpenAI client and emits strict:
- `[START]`
- `[STEP]`
- `[END]`

Required environment variables:
- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Optional:
- `ENV_URL` default `http://localhost:8000`

Example:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="<your_token>"
export ENV_URL="http://localhost:8000"
python inference.py
```

## Local Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r server/requirements.txt
pip install -e .
```

Run tests:

```bash
../venv/bin/python -m pytest -q
```

Run local server:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Validate OpenEnv packaging:

```bash
../venv/bin/openenv validate
```

## Docker

Build:

```bash
docker build -t shopops-env:latest .
```

Run:

```bash
docker run -p 8000:8000 shopops-env:latest
```

## Files

Important entrypoints:
- `server/shopOps_environment.py`
- `models.py`
- `graders.py`
- `eval.py`
- `inference.py`
- `openenv.yaml`
