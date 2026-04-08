"""
Baseline inference script for Clinical Triage Coordinator.
Uses OpenAI-compatible client (works with Groq, OpenAI, OpenRouter).

Environment variables:
  API_BASE_URL : LLM API base URL (default: https://api.groq.com/openai/v1)
  MODEL_NAME   : Model identifier (default: llama-3.1-8b-instant)
  HF_TOKEN     : API key for LLM provider (Groq key goes here)
  ENV_URL      : Running environment URL (default: http://localhost:8000)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

import httpx
from openai import AsyncOpenAI


API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
HF_TOKEN = os.getenv("HF_TOKEN", "")
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000").rstrip("/")

BENCHMARK = "clinical-triage-env"
MAX_STEPS = 15
MAX_TOTAL_REWARD = 15.0
SUCCESS_SCORE_THRESHOLD = 0.4
TASKS = ["easy", "medium", "hard"]


def log_start(task: str, env: str, model: str) -> None:
    print(json.dumps({
        "type": "START",
        "task": task,
        "env": env,
        "model": model,
    }), flush=True)


def log_step(step: int, action: str, reward: float,
             done: bool, error=None) -> None:
    print(json.dumps({
        "type": "STEP",
        "step": step,
        "action": str(action),
        "reward": round(float(reward), 4),
        "done": bool(done),
        "error": error,
    }), flush=True)


def log_end(success: bool, steps: int,
            score: float, rewards: list) -> None:
    print(json.dumps({
        "type": "END",
        "success": bool(success),
        "steps": steps,
        "score": round(float(score), 4),
        "rewards": [round(float(r), 4) for r in rewards],
    }), flush=True)


def get_fallback_action(patient_queue: list) -> dict | None:
    """
    Rule-based fallback when LLM is unavailable or errors.
    Always picks the patient with highest MEWS score.
    Uses MEWS thresholds to determine correct action.
    """
    if not patient_queue:
        return None

    patient = max(patient_queue, key=lambda p: p.get("mews_score", 0))
    mews = patient.get("mews_score", 0)

    if mews >= 7:
        return {
            "patient_id": patient["patient_id"],
            "assigned_severity": 1,
            "assigned_ward": 1,
            "treatment_protocol": 1,
            "resource_action": 1,
        }
    elif mews >= 5:
        return {
            "patient_id": patient["patient_id"],
            "assigned_severity": 2,
            "assigned_ward": 2,
            "treatment_protocol": 2,
            "resource_action": 2,
        }
    elif mews >= 3:
        return {
            "patient_id": patient["patient_id"],
            "assigned_severity": 3,
            "assigned_ward": 3,
            "treatment_protocol": 2,
            "resource_action": 4,
        }
    elif mews >= 1:
        return {
            "patient_id": patient["patient_id"],
            "assigned_severity": 4,
            "assigned_ward": 3,
            "treatment_protocol": 2,
            "resource_action": 5,
        }
    else:
        return {
            "patient_id": patient["patient_id"],
            "assigned_severity": 5,
            "assigned_ward": 4,
            "treatment_protocol": 4,
            "resource_action": 5,
        }


async def get_model_action(
    client: AsyncOpenAI,
    patient_queue: list,
    icu_beds: int,
    staff_units: int,
    step: int,
    history: list,
) -> dict | None:
    """
    Ask LLM to pick the best triage action.
    Falls back to rule-based if LLM fails or no API key.
    """
    if not HF_TOKEN:
        return get_fallback_action(patient_queue)

    system_prompt = """You are an expert medical triage coordinator for an Indian district hospital.

Pick the MOST CRITICAL patient from the queue and decide:
1. assigned_severity: 1=Critical, 2=Emergency, 3=Urgent, 4=Semi-urgent, 5=Non-urgent
2. assigned_ward: 1=ICU, 2=Emergency, 3=General, 4=Discharge
3. treatment_protocol: 1=Stabilize, 2=Medicate, 3=Refer, 4=Discharge
4. resource_action: 1=Assign ICU Bed, 2=Assign General Bed, 3=Schedule Lab, 4=Allocate Staff, 5=Hold

MEWS guide (use this to decide):
MEWS>=7  -> severity=1, ward=1, treatment=1, resource=1
MEWS 5-6 -> severity=2, ward=2, treatment=2, resource=2
MEWS 3-4 -> severity=3, ward=3, treatment=2, resource=4
MEWS 1-2 -> severity=4, ward=3, treatment=2, resource=5
MEWS=0   -> severity=5, ward=4, treatment=4, resource=5

Respond with ONLY valid JSON, no markdown, no explanation:
{
  "patient_id": "<id>",
  "assigned_severity": <1-5>,
  "assigned_ward": <1-4>,
  "treatment_protocol": <1-4>,
  "resource_action": <1-5>
}"""

    queue_summary = []
    for p in patient_queue[:10]:
        queue_summary.append({
            "patient_id": p.get("patient_id"),
            "mews_score": p.get("mews_score"),
            "heart_rate": p.get("heart_rate"),
            "systolic_bp": p.get("systolic_bp"),
            "spo2": p.get("spo2"),
            "time_in_queue": p.get("time_in_queue"),
        })

    user_message = (
        f"Step {step}\n"
        f"ICU beds available: {icu_beds}\n"
        f"Staff units free: {staff_units}\n"
        f"Patient queue ({len(patient_queue)} patients):\n"
        f"{json.dumps(queue_summary, indent=2)}\n"
    )
    if history:
        user_message += f"\nRecent history:\n" + "\n".join(history[-3:])

    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=200,
            temperature=0.1,
        )
        text = (response.choices[0].message.content or "").strip()

        # Strip markdown code blocks if present
        if "```" in text:
            lines = text.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            text = "\n".join(lines).strip()

        action = json.loads(text)

        # Validate required fields exist
        required = [
            "patient_id",
            "assigned_severity",
            "assigned_ward",
            "treatment_protocol",
            "resource_action",
        ]
        for field in required:
            if field not in action:
                raise ValueError(f"Missing field: {field}")

        return action

    except Exception:
        # Fallback to rule-based on any error
        return get_fallback_action(patient_queue)


async def run_task(
    task_name: str,
    client: AsyncOpenAI,
    http_client: httpx.AsyncClient,
) -> float:
    """Run one full task episode and return score 0.0-1.0."""
    _ = sys.version_info
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    rewards = []
    history = []
    steps_taken = 0
    success = False
    score = 0.0
    done = False

    try:
        # Reset with specific task
        response = await http_client.post(
            f"{ENV_URL}/reset/{task_name}",
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        obs = data["observation"]
        done = data.get("done", False)

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            patient_queue = obs.get("patient_queue", [])
            if not patient_queue:
                break

            icu_beds = obs.get("icu_beds_available", 2)
            staff_units = obs.get("staff_units_free", 5)

            # Get action from LLM or fallback
            action = await get_model_action(
                client, patient_queue, icu_beds,
                staff_units, step, history,
            )

            if action is None:
                break

            # Execute step
            try:
                step_response = await http_client.post(
                    f"{ENV_URL}/step",
                    json=action,
                    timeout=30.0,
                )
                step_response.raise_for_status()
                step_data = step_response.json()
                obs = step_data["observation"]
                reward = float(step_data.get("reward", 0.0))
                reward = max(0.0, min(1.0, reward))
                done = bool(step_data.get("done", False))
                error = None
            except Exception as e:
                reward = 0.0
                done = False
                error = str(e)

            log_step(
                step=step,
                action=action.get("patient_id", "unknown"),
                reward=reward,
                done=done,
                error=error,
            )

            rewards.append(reward)
            steps_taken = step
            history.append(
                f"Step {step}: patient={action.get('patient_id')} "
                f"severity={action.get('assigned_severity')} "
                f"reward={reward:.3f}"
            )

            if done:
                break

        # Get task grade
        try:
            grade_response = await http_client.post(
                f"{ENV_URL}/grade",
                timeout=10.0,
            )
            grade_response.raise_for_status()
            grade_data = grade_response.json()
            score = float(grade_data.get("score", 0.0))
            score = max(0.0, min(1.0, score))
        except Exception:
            # Fallback: average step rewards
            score = sum(rewards) / MAX_TOTAL_REWARD if rewards else 0.0
            score = max(0.0, min(1.0, score))

        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(json.dumps({
            "type": "ERROR",
            "task": task_name,
            "error": str(e),
        }), flush=True)
        score = 0.0
        success = False

    log_end(
        success=success,
        steps=steps_taken,
        score=score,
        rewards=rewards,
    )
    return score


async def main() -> None:
    """Run all 3 tasks and print summary."""
    # Create OpenAI-compatible client pointing to Groq
    client = AsyncOpenAI(
        api_key=HF_TOKEN if HF_TOKEN else "dummy-key-fallback",
        base_url=API_BASE_URL,
    )

    all_scores = []

    async with httpx.AsyncClient() as http_client:
        for task_name in TASKS:
            try:
                score = await run_task(task_name, client, http_client)
                all_scores.append(score)
            except Exception as e:
                print(json.dumps({
                    "type": "ERROR",
                    "task": task_name,
                    "error": str(e),
                }), flush=True)
                all_scores.append(0.0)

    # Final summary
    mean_score = round(sum(all_scores) / len(all_scores), 4)
    print(json.dumps({
        "type": "SUMMARY",
        "tasks": TASKS,
        "scores": [round(s, 4) for s in all_scores],
        "mean_score": mean_score,
        "all_passed": all(s >= SUCCESS_SCORE_THRESHOLD for s in all_scores),
    }), flush=True)


if __name__ == "__main__":
    asyncio.run(main())