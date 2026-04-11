"""
Baseline inference script — Clinical Triage Coordinator.

Runs an LLM agent against all 3 tasks and emits structured logs.

Environment variables:
  OPENAI_API_KEY : API key (works with Groq, OpenAI, OpenRouter)
  API_BASE_URL   : LLM endpoint (default: https://api.groq.com/openai/v1)
  MODEL_NAME     : Model name (default: llama-3.1-8b-instant)
  HF_TOKEN       : Alternative name for API key (fallback)
  ENV_URL        : Running server URL (default: http://localhost:8000)
"""

from __future__ import annotations
import asyncio
import json
import os
import sys
from typing import Optional
import httpx
from openai import AsyncOpenAI

OPENAI_API_KEY = (
    os.getenv("OPENAI_API_KEY")
    or os.getenv("HF_TOKEN")
    or ""
)
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:8000").rstrip("/")

BENCHMARK             = "clinical-triage-env"
MAX_STEPS             = 15
MAX_TOTAL_REWARD      = 15.0
SUCCESS_THRESHOLD     = 0.4
TASKS                 = ["easy", "medium", "hard"]


# ── Logging — exact format required by validator ──────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str] = None,
) -> None:
    done_s  = "true" if done else "false"
    error_s = str(error) if error else "null"
    print(
        f"[STEP] step={step} action={action} "
        f"reward={round(float(reward), 4)} "
        f"done={done_s} error={error_s}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: list[float],
) -> None:
    success_s  = "true" if success else "false"
    rewards_s  = ",".join(str(round(float(r), 4)) for r in rewards)
    print(
        f"[END] success={success_s} steps={steps} "
        f"score={round(float(score), 4)} "
        f"rewards=[{rewards_s}]",
        flush=True,
    )


# ── Rule-based fallback (no API key needed) ───────────────────────────────────

def get_fallback_action(patient_queue: list[dict]) -> Optional[dict]:
    """Pick most critical patient using MEWS thresholds."""
    if not patient_queue:
        return None
    patient = max(patient_queue, key=lambda p: p.get("mews_score", 0))
    mews    = patient.get("mews_score", 0)
    pid     = patient["patient_id"]
    if mews >= 7:
        return dict(patient_id=pid, assigned_severity=1,
                    assigned_ward=1, treatment_protocol=1, resource_action=1)
    elif mews >= 5:
        return dict(patient_id=pid, assigned_severity=2,
                    assigned_ward=2, treatment_protocol=2, resource_action=2)
    elif mews >= 3:
        return dict(patient_id=pid, assigned_severity=3,
                    assigned_ward=3, treatment_protocol=2, resource_action=4)
    elif mews >= 1:
        return dict(patient_id=pid, assigned_severity=4,
                    assigned_ward=3, treatment_protocol=2, resource_action=5)
    else:
        return dict(patient_id=pid, assigned_severity=5,
                    assigned_ward=4, treatment_protocol=4, resource_action=5)


# ── LLM action (with fallback) ────────────────────────────────────────────────

async def get_model_action(
    client: AsyncOpenAI,
    patient_queue: list[dict],
    icu_beds: int,
    staff_free: int,
    step: int,
    history: list[str],
) -> Optional[dict]:
    if not OPENAI_API_KEY:
        return get_fallback_action(patient_queue)

    system = (
        "You are an expert medical triage coordinator for an Indian "
        "district hospital.\n\n"
        "Pick the MOST CRITICAL patient and respond with ONLY valid JSON "
        "(no markdown, no explanation):\n"
        '{"patient_id":"<id>","assigned_severity":<1-5>,'
        '"assigned_ward":<1-4>,"treatment_protocol":<1-4>,'
        '"resource_action":<1-5>}\n\n'
        "MEWS guide:\n"
        ">=7  -> severity=1 ward=1 treatment=1 resource=1\n"
        "5-6  -> severity=2 ward=2 treatment=2 resource=2\n"
        "3-4  -> severity=3 ward=3 treatment=2 resource=4\n"
        "1-2  -> severity=4 ward=3 treatment=2 resource=5\n"
        "0    -> severity=5 ward=4 treatment=4 resource=5"
    )

    brief_queue = [
        {"patient_id": p["patient_id"],
         "mews_score": p["mews_score"],
         "time_in_queue": p.get("time_in_queue", 0)}
        for p in patient_queue[:8]
    ]
    user_msg = (
        f"Step {step} | ICU beds={icu_beds} | Staff={staff_free}\n"
        f"Queue: {json.dumps(brief_queue)}"
    )
    if history:
        user_msg += "\nHistory: " + "; ".join(history[-2:])

    try:
        resp = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=150,
            temperature=0.0,
        )
        text = (resp.choices[0].message.content or "").strip()
        # Strip markdown fences
        for fence in ("```json", "```"):
            text = text.replace(fence, "")
        text = text.strip()
        action = json.loads(text)
        required = ["patient_id","assigned_severity","assigned_ward",
                    "treatment_protocol","resource_action"]
        if not all(k in action for k in required):
            raise ValueError("missing fields")
        return action
    except Exception:
        return get_fallback_action(patient_queue)


# ── Run one task ──────────────────────────────────────────────────────────────

async def run_task(
    task_name: str,
    llm_client: AsyncOpenAI,
    http: httpx.AsyncClient,
) -> float:
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    rewards:     list[float] = []
    history:     list[str]   = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False

    try:
        # Reset environment with specific task
        r = await http.post(f"{ENV_URL}/reset/{task_name}", timeout=30.0)
        r.raise_for_status()
        data = r.json()
        obs  = data["observation"]
        done = bool(data.get("done", False))

        for step in range(1, MAX_STEPS + 1):
            if done:
                break
            queue = obs.get("patient_queue", [])
            if not queue:
                break

            action = await get_model_action(
                llm_client,
                queue,
                obs.get("icu_beds_available", 2),
                obs.get("staff_units_free", 5),
                step,
                history,
            )
            if action is None:
                break

            error_s: Optional[str] = None
            try:
                sr = await http.post(
                    f"{ENV_URL}/step",
                    json=action,
                    timeout=30.0,
                )
                sr.raise_for_status()
                sd     = sr.json()
                obs    = sd["observation"]
                reward = float(sd.get("reward", 0.0))
                done   = bool(sd.get("done", False))
            except Exception as e:
                reward  = 0.0
                done    = False
                error_s = str(e)[:80]

            log_step(step, action.get("patient_id", "unknown"),
                     reward, done, error_s)
            rewards.append(reward)
            steps_taken = step
            history.append(
                f"s{step}:{action.get('patient_id')} r={reward:.2f}"
            )
            if done:
                break

        # Get task grade
        try:
            gr    = await http.post(f"{ENV_URL}/grade", timeout=10.0)
            gdata = gr.json()
            score = float(gdata.get("score", 0.0))
        except Exception:
            score = (
                sum(rewards) / MAX_TOTAL_REWARD if rewards else 0.0
            )
        score   = max(0.0, min(1.0, score))
        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task {task_name} error: {e}", flush=True)
        score   = 0.0
        success = False

    log_end(success=success, steps=steps_taken,
            score=score, rewards=rewards)
    return score


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    llm_client = AsyncOpenAI(
        api_key  = OPENAI_API_KEY if OPENAI_API_KEY else "dummy",
        base_url = API_BASE_URL,
    )

    all_scores: list[float] = []

    async with httpx.AsyncClient() as http:
        for task_name in TASKS:
            try:
                s = await run_task(task_name, llm_client, http)
            except Exception as e:
                print(f"[DEBUG] {task_name} failed: {e}", flush=True)
                s = 0.0
            all_scores.append(s)

    # Close LLM client
    try:
        await llm_client.close()
    except Exception:
        pass

    mean = round(sum(all_scores) / len(all_scores), 4)
    print(
        f"[SUMMARY] tasks={TASKS} scores={all_scores} "
        f"mean={mean} "
        f"all_passed={all(s >= SUCCESS_THRESHOLD for s in all_scores)}",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
