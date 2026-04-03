import json
import os
import re
import sys
from typing import Any, Dict

import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")

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


def main() -> None:
    _require_env()
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    seed = int(os.getenv("SEED", "42"))

    print("[START]")
    print(f"episode_id=unknown")
    print(f"seed={seed}")
    print(f"model={MODEL_NAME}")
    print(f"env_url={ENV_URL}")

    reset_resp = requests.post(f"{ENV_URL}/reset", json={"seed": seed})
    reset_resp.raise_for_status()
    payload = reset_resp.json()
    obs = payload["observation"]
    episode_id = obs.get("episode_id", "unknown")
    print(f"episode_id={episode_id}")

    step = 0
    done = payload.get("done", False)

    while not done:
        prompt = (
            "You are an e-commerce ops agent. Return ONLY JSON with keys: "
            "action_type, refund_amount_usd, replacement_expedite, escalation_reason. "
            f"Observation: {json.dumps(obs)}"
        )

        try:
            response = client.responses.create(
                model=MODEL_NAME,
                input=prompt,
            )
            action = _parse_action(response.output_text)
        except Exception:
            action = _safe_action()

        step_resp = requests.post(
            f"{ENV_URL}/step",
            json={"action": action, "episode_id": episode_id},
        )
        step_resp.raise_for_status()
        step_payload = step_resp.json()

        print("[STEP]")
        print(f"step={step}")
        print(f"action={json.dumps(action)}")
        print(f"reward={step_payload.get('reward')}")
        print(f"done={step_payload.get('done')}")

        obs = step_payload["observation"]
        done = step_payload.get("done", False)
        step += 1
        if step >= 20:
            break

    final_score = (
        obs.get("metadata", {})
        .get("episode_summary", {})
        .get("final_score")
    )
    print("[END]")
    print(f"final_score={final_score}")


if __name__ == "__main__":
    main()
