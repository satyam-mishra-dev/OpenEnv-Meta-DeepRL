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

TASK_NAME = os.getenv("TASK_NAME", "shopops")
BENCHMARK = os.getenv("BENCHMARK", "shopops")
MAX_STEPS = int(os.getenv("MAX_STEPS", "20"))

REQUIRED_VARS = {
    "API_BASE_URL": API_BASE_URL,
    "MODEL_NAME": MODEL_NAME,
    "HF_TOKEN": HF_TOKEN,
}


def _require_env() -> None:
    missing = [key for key, value in REQUIRED_VARS.items() if not value]
    if missing:
        print("Missing required env vars: " + ", ".join(missing))
        sys.exit(2)


def _parse_action(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
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


def main() -> None:
    _require_env()
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    seed = int(os.getenv("SEED", "42"))
    _log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    try:
        reset_resp = requests.post(f"{ENV_URL}/reset", json={"seed": seed})
        reset_resp.raise_for_status()
        payload = reset_resp.json()
        obs = payload["observation"]
        episode_id = obs.get("episode_id", "unknown")

        step = 1
        done = payload.get("done", False)

        while not done and step <= MAX_STEPS:
            prompt = (
                "You are an e-commerce ops agent. Return ONLY JSON with keys: "
                "action_type, refund_amount_usd, replacement_expedite, escalation_reason. "
                f"Observation: {json.dumps(obs)}"
            )

            try:
                response = client.responses.create(
                    model=MODEL_NAME,
                    input=prompt,
                    text={"format": {"type": "json_object"}},
                )
                action = _parse_action(response.output_text)
            except Exception as exc:
                action = _safe_action()

            step_resp = requests.post(
                f"{ENV_URL}/step",
                json={"action": action, "episode_id": episode_id},
            )
            step_payload = {}
            if step_resp.status_code == 200:
                step_payload = step_resp.json()
                reward = float(step_payload.get("reward") or 0.0)
                done = bool(step_payload.get("done", False))
                error = (
                    (step_payload.get("observation") or {})
                    .get("metadata", {})
                    .get("validation_error")
                )
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

            if step_payload:
                obs = step_payload["observation"]
            step += 1

        # HTTP API does not include episode_summary, so compute a normalized score.
        # This keeps score within [0, 1] for logging.
        score = sum(rewards) / float(MAX_STEPS) if MAX_STEPS > 0 else 0.0
        score = max(0.0, min(1.0, score))
        success = score > 0.0
    finally:
        _log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
