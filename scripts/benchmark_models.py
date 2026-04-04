#!/usr/bin/env python3
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List

END_RE = re.compile(r"^\[END\] success=(true|false) steps=(\d+) score=([0-9.]+) rewards=(.*)$")

@dataclass
class RunResult:
    success: bool
    steps: int
    score: float


def run_once(root: str, env: dict) -> RunResult:
    proc = subprocess.run(
        [sys.executable, os.path.join(root, "inference.py")],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"inference.py failed (rc={proc.returncode})\n{proc.stdout}")

    end_line = None
    for line in proc.stdout.splitlines():
        if line.startswith("[END]"):
            end_line = line
    if not end_line:
        raise RuntimeError("Missing [END] in inference output")

    m = END_RE.match(end_line.strip())
    if not m:
        raise RuntimeError(f"[END] line did not match format: {end_line}")

    success = m.group(1) == "true"
    steps = int(m.group(2))
    score = float(m.group(3))
    return RunResult(success=success, steps=steps, score=score)


def avg(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> int:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    api_base_url = os.getenv("API_BASE_URL")
    if not api_base_url:
        print("API_BASE_URL is required", file=sys.stderr)
        return 2

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("HF_TOKEN is required", file=sys.stderr)
        return 2

    env_url = os.getenv("ENV_URL", "http://localhost:8000")

    models = os.getenv("BENCH_MODELS", "gpt-4o,gpt-4.1,gpt-4.1-mini,gpt-4o-mini").split(",")
    seeds_env = os.getenv("BENCH_SEEDS", "1,2,3,4,5,6,7,8,9,10")
    seeds = [int(s.strip()) for s in seeds_env.split(",") if s.strip()]

    print(f"Benchmarking {len(models)} model(s) across {len(seeds)} seed(s)", flush=True)
    print(f"ENV_URL={env_url}", flush=True)

    summary: Dict[str, dict] = {}

    for model in models:
        model = model.strip()
        if not model:
            continue
        scores: List[float] = []
        successes: List[int] = []
        steps: List[int] = []

        for seed in seeds:
            print(f"{model}: seed={seed} ...", flush=True)
            env = os.environ.copy()
            env["API_BASE_URL"] = api_base_url
            env["MODEL_NAME"] = model
            env["HF_TOKEN"] = hf_token
            env["ENV_URL"] = env_url
            env["SEED"] = str(seed)

            r = run_once(root, env)
            scores.append(r.score)
            successes.append(1 if r.success else 0)
            steps.append(r.steps)

        avg_score = avg(scores)
        success_rate = avg(successes) * 100.0
        avg_steps = avg(steps)

        summary[model] = {
            "avg_score": avg_score,
            "success_rate": success_rate,
            "avg_steps": avg_steps,
            "seeds": len(seeds),
        }

        print(
            f"{model}: avg_score={avg_score:.4f} success_rate={success_rate:.1f}% avg_steps={avg_steps:.1f}",
            flush=True,
        )

    print("\nREADME_TABLE_START", flush=True)
    print("| Model | Avg Score | Success Rate | Avg Steps | Seeds |", flush=True)
    print("| --- | --- | --- | --- | --- |", flush=True)
    for model, stats in summary.items():
        print(
            f"| {model} | {stats['avg_score']:.4f} | {stats['success_rate']:.1f}% | {stats['avg_steps']:.1f} | {stats['seeds']} |",
            flush=True,
        )
    print("README_TABLE_END", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
