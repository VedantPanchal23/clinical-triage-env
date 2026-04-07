"""Comprehensive end-to-end tests for the Clinical Triage Coordinator environment."""

from __future__ import annotations

import os
import sys
from typing import Any

import httpx

try:
    import pytest  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - enables direct script execution without pytest installed
    class _MarkShim:
        @staticmethod
        def asyncio(func):
            return func

    class _PytestShim:
        mark = _MarkShim()

    pytest = _PytestShim()  # type: ignore[assignment]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from client import TriageEnv
from models import TriageAction
from server.llm_grader import LLMGrader
from server.mews_scorer import compute_mews, score_triage_decision

BASE_URL = "http://localhost:8000"
TIMEOUT_S = 30.0


def _severity_from_mews(mews: int) -> int:
    if mews >= 7:
        return 1
    if mews >= 5:
        return 2
    if mews >= 3:
        return 3
    if mews >= 1:
        return 4
    return 5


def _action_for_mews(patient: dict[str, Any]) -> dict[str, int]:
    mews = int(patient["mews_score"])
    if mews >= 7:
        return {
            "assigned_severity": 1,
            "assigned_ward": 1,
            "treatment_protocol": 1,
            "resource_action": 1,
        }
    if mews >= 5:
        return {
            "assigned_severity": 2,
            "assigned_ward": 2,
            "treatment_protocol": 1,
            "resource_action": 2,
        }
    if mews >= 3:
        return {
            "assigned_severity": 3,
            "assigned_ward": 3,
            "treatment_protocol": 2,
            "resource_action": 2,
        }
    if mews >= 1:
        return {
            "assigned_severity": 4,
            "assigned_ward": 3,
            "treatment_protocol": 2,
            "resource_action": 2,
        }
    return {
        "assigned_severity": 5,
        "assigned_ward": 4,
        "treatment_protocol": 4,
        "resource_action": 5,
    }


async def _reset(client: httpx.AsyncClient) -> dict[str, Any]:
    response = await client.post(f"{BASE_URL}/reset")
    assert response.status_code == 200
    payload = response.json()
    assert "observation" in payload
    return payload["observation"]


async def _step(client: httpx.AsyncClient, payload: dict[str, Any]) -> dict[str, Any]:
    response = await client.post(f"{BASE_URL}/step", json={"action": payload})
    if response.status_code == 422:
        # Backward-compatible fallback for deployments that accept unwrapped action bodies.
        response = await client.post(f"{BASE_URL}/step", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert "observation" in result
    return result["observation"]


@pytest.mark.asyncio
async def test_health_endpoint() -> None:
    async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
        response = await client.get(f"{BASE_URL}/health")

    assert response.status_code == 200
    payload = response.json()
    assert "status" in payload
    print("✓ Health endpoint OK")


@pytest.mark.asyncio
async def test_reset_endpoint() -> None:
    async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
        obs = await _reset(client)

    assert "patient_queue" in obs
    assert len(obs["patient_queue"]) == 8
    assert obs["icu_beds_available"] == 2
    assert obs["general_beds_available"] == 10
    assert obs["staff_units_free"] == 5
    assert obs["episode_done"] is False
    assert obs["step_count"] == 0
    print("✓ Reset endpoint OK: 8 patients in queue")


@pytest.mark.asyncio
async def test_patient_has_mews_score() -> None:
    async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
        obs = await _reset(client)

    patient = obs["patient_queue"][0]
    assert "mews_score" in patient
    mews_score = int(patient["mews_score"])
    assert mews_score >= 0

    if "true_severity" in patient:
        assert patient["true_severity"] in [1, 2, 3, 4, 5]
    else:
        assert _severity_from_mews(mews_score) in [1, 2, 3, 4, 5]

    print("✓ Patient MEWS scoring OK")


@pytest.mark.asyncio
async def test_step_critical_patient() -> None:
    async with TriageEnv(base_url=BASE_URL) as client:
        reset_result = await client.reset()
        critical_patient = max(
            reset_result.observation.patient_queue,
            key=lambda p: p["mews_score"],
        )
        action = TriageAction(
            patient_id=critical_patient["patient_id"],
            assigned_severity=1,
            assigned_ward=1,
            treatment_protocol=1,
            resource_action=1,
        )
        step_result = await client.step(action)
        next_obs = step_result.observation

    assert next_obs.step_count == 1
    assert next_obs.step_reward > 0
    assert next_obs.patients_stabilized >= 1
    assert next_obs.icu_beds_available == 1
    print(f"✓ Critical patient step OK: reward={next_obs.step_reward}")


@pytest.mark.asyncio
async def test_step_noncritical_patient() -> None:
    async with TriageEnv(base_url=BASE_URL) as client:
        reset_result = await client.reset()
        patient_queue = reset_result.observation.patient_queue
        zero_mews = [p for p in patient_queue if p["mews_score"] == 0]
        noncritical_patient = zero_mews[0] if zero_mews else min(patient_queue, key=lambda p: p["mews_score"])

        action = TriageAction(
            patient_id=noncritical_patient["patient_id"],
            assigned_severity=5,
            assigned_ward=4,
            treatment_protocol=4,
            resource_action=5,
        )
        step_result = await client.step(action)
        next_obs = step_result.observation

    assert next_obs.step_reward > 0
    print("✓ Non-critical patient step OK")


@pytest.mark.asyncio
async def test_state_endpoint() -> None:
    async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
        await _reset(client)
        response = await client.get(f"{BASE_URL}/state")

    assert response.status_code == 200
    payload = response.json()
    assert "episode_id" in payload
    assert payload.get("step_count", 0) == 0
    assert payload.get("icu_beds_available", 2) == 2
    print("✓ State endpoint OK")


@pytest.mark.asyncio
async def test_multiple_steps() -> None:
    async with TriageEnv(base_url=BASE_URL) as client:
        reset_result = await client.reset()
        obs = reset_result.observation
        for expected_step in range(1, 6):
            patient = obs.patient_queue[0]
            action = _action_for_mews(patient)
            step_action = TriageAction(
                patient_id=patient["patient_id"],
                assigned_severity=action["assigned_severity"],
                assigned_ward=action["assigned_ward"],
                treatment_protocol=action["treatment_protocol"],
                resource_action=action["resource_action"],
            )
            step_result = await client.step(step_action)
            obs = step_result.observation

            assert obs.step_count == expected_step
            assert isinstance(float(obs.step_reward), float)

    assert obs.step_count == 5
    print("✓ Multiple steps OK: 5 steps completed")


@pytest.mark.asyncio
async def test_client_async() -> None:
    async with TriageEnv(base_url=BASE_URL) as client:
        result = await client.reset()
        assert result.observation is not None
        assert len(result.observation.patient_queue) == 8

        p = result.observation.patient_queue[0]
        action = TriageAction(
            patient_id=p["patient_id"],
            assigned_severity=1 if p["mews_score"] >= 7 else 5,
            assigned_ward=1 if p["mews_score"] >= 7 else 4,
            treatment_protocol=1 if p["mews_score"] >= 7 else 4,
            resource_action=1 if p["mews_score"] >= 7 else 5,
        )
        result2 = await client.step(action)

    assert result2.reward is not None
    assert isinstance(result2.reward, float)
    print("✓ Async client OK")


@pytest.mark.asyncio
async def test_mews_grader_direct() -> None:
    mews = compute_mews(140, 70, 32, 39.5, 82.0, 2)
    assert mews.total >= 7
    assert mews.severity_label == "CRITICAL"
    assert mews.recommended_ward == "ICU"

    reward, feedback = score_triage_decision(1, 1, mews.total)
    assert reward > 0
    assert "Correct" in feedback

    reward2, _feedback2 = score_triage_decision(5, 4, mews.total)
    assert reward2 < 0

    print("✓ MEWS grader direct test OK")


@pytest.mark.asyncio
async def test_llm_grader_fallback() -> None:
    grader = LLMGrader()

    sample_log = [
        {
            "step": 1,
            "patient_id": "PT-001",
            "mews_score": 15,
            "true_severity": 1,
            "assigned_severity": 1,
            "assigned_ward": 1,
            "treatment_protocol": 1,
            "resource_action": 1,
            "step_reward": 3.5,
            "feedback": "Correct severity",
        },
        {
            "step": 2,
            "patient_id": "PT-002",
            "mews_score": 0,
            "true_severity": 5,
            "assigned_severity": 5,
            "assigned_ward": 4,
            "treatment_protocol": 4,
            "resource_action": 5,
            "step_reward": 1.5,
            "feedback": "Correct severity",
        },
    ]

    result = await grader.grade_episode(sample_log)
    assert 0 <= result.score <= 10
    assert result.total_decisions == 2
    assert result.grade in ["Dangerous", "Poor", "Acceptable", "Good", "Excellent"]
    assert result.justification != ""
    print(f"✓ LLM grader fallback OK: score={result.score}")


if __name__ == "__main__":
    import asyncio

    async def run_all() -> None:
        print("Running end-to-end tests...")
        print("Make sure server is running: python -m uvicorn server.app:app --port 8000")
        print()
        await test_health_endpoint()
        await test_reset_endpoint()
        await test_patient_has_mews_score()
        await test_step_critical_patient()
        await test_step_noncritical_patient()
        await test_state_endpoint()
        await test_multiple_steps()
        await test_client_async()
        await test_mews_grader_direct()
        await test_llm_grader_fallback()
        print()
        print("ALL TESTS PASSED")

    asyncio.run(run_all())
