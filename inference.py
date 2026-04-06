"""
ShopOps Inference Script
========================
Runs the LLM agent against all three difficulty tiers (easy, medium, hard)
and emits strict [START] / [STEP] / [END] logs to stdout.

Required environment variables:
    API_BASE_URL  – LLM API endpoint  (default: https://router.huggingface.co/v1)
    MODEL_NAME    – model identifier  (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN      – Hugging Face / API key  (required)

Optional:
    ENV_URL       – environment server URL (default: http://localhost:8000)
    MAX_STEPS     – max steps per episode  (default: 20)
    SEED          – random seed for reproducibility (default: 42)
"""

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")

BENCHMARK = "shopops"
MAX_STEPS = int(os.getenv("MAX_STEPS", "20"))
SEED = int(os.getenv("SEED", "42"))
TIERS = ["easy", "medium", "hard"]

# Max theoretical reward per step is 1.0 (correctness=1, efficiency=1, priority=1).
# Normalise cumulative reward against this ceiling so score stays in [0, 1].
MAX_REWARD_PER_EPISODE = float(MAX_STEPS)

_SYSTEM_PROMPT = (
    "You are an e-commerce support agent. Analyse the case and return ONLY a valid JSON object "
    "with exactly these four keys: action_type, refund_amount_usd, replacement_expedite, escalation_reason.\n\n"
    "action_type choices:\n"
    "  refund    – set refund_amount_usd to a positive float <= order value\n"
    "  replace   – set replacement_expedite to true/false\n"
    "  escalate  – set escalation_reason to one of: suspected_fraud | high_value | policy_exception | safety_issue\n"
    "  reject    – no extra fields needed (set others to null/false)\n\n"
    "Decision rules:\n"
    "  fraud_signal=high                          → escalate, suspected_fraud\n"
    "  fraud_signal=medium                        → reject\n"
    "  refund_request + return window closed      → reject\n"
    "  delivery lost                              → replace\n"
    "  delivery delayed                           → refund 20% of order value\n"
    "  delivery in_transit                        → escalate, policy_exception\n"
    "  wrong_item with evidence                   → replace\n"
    "  wrong_item gold/platinum, few refunds      → replace\n"
    "  default                                    → reject\n"
)


def _require_env() -> None:
    if not HF_TOKEN:
        print("Missing required env var: HF_TOKEN", flush=True)
        sys.exit(2)


def _parse_action(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def _safe_action() -> Dict[str, Any]:
    return {
        "action_type": "reject",
        "refund_amount_usd": None,
        "replacement_expedite": False,
        "escalation_reason": None,
    }


def _log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def _log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    done_val = str(done).lower()
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def _log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def _get_action(client: OpenAI, obs: Dict[str, Any]) -> Dict[str, Any]:
    """Call the LLM to decide an action; fall back to reject on any error."""
    user_msg = (
        f"Case: {json.dumps(obs.get('case', {}))}\n"
        f"Resources: {json.dumps(obs.get('resources', {}))}\n"
        f"Tier: {obs.get('tier', 'unknown')}\n\n"
        "Return ONLY the JSON object."
    )
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=150,
        )
        text = (response.choices[0].message.content or "").strip()
        return _parse_action(text)
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return _safe_action()


def _run_tier(client: OpenAI, tier: str) -> None:
    """Run one full episode for the given tier, emitting START / STEP / END logs."""
    _log_start(task=tier, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    try:
        reset_resp = requests.post(
            f"{ENV_URL}/reset",
            json={"seed": SEED, "tier": tier},
            timeout=30,
        )
        reset_resp.raise_for_status()
        payload = reset_resp.json()
        obs = payload.get("observation", {})
        episode_id = obs.get("episode_id", "unknown")
        done = payload.get("done", False)

        step = 1
        while not done and step <= MAX_STEPS:
            action = _get_action(client, obs)

            step_resp = requests.post(
                f"{ENV_URL}/step",
                json={"action": action, "episode_id": episode_id},
                timeout=30,
            )
            error: Optional[str] = None
            if step_resp.status_code == 200:
                step_payload = step_resp.json()
                reward = float(step_payload.get("reward") or 0.0)
                done = bool(step_payload.get("done", False))
                error = (
                    (step_payload.get("observation") or {})
                    .get("metadata", {})
                    .get("validation_error")
                )
                obs = step_payload.get("observation", obs)
            else:
                try:
                    err_payload = step_resp.json()
                    error = err_payload.get("detail") or str(err_payload)
                except Exception:
                    error = step_resp.text or f"http_{step_resp.status_code}"
                reward = 0.0
                done = True

            rewards.append(reward)
            steps_taken = step
            _log_step(
                step=step,
                action=json.dumps(action, separators=(",", ":")),
                reward=reward,
                done=done,
                error=error,
            )
            step += 1

        # Normalise: max reward per step = 1.0, so dividing by MAX_STEPS maps [0, 20] → [0, 1].
        # Negative rewards are clamped to 0.
        score = sum(rewards) / MAX_REWARD_PER_EPISODE if MAX_REWARD_PER_EPISODE > 0 else 0.0
        score = max(0.0, min(1.0, score))
        success = score > 0.0
    finally:
        _log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    _require_env()
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    for tier in TIERS:
        _run_tier(client, tier)


if __name__ == "__main__":
    main()
