# Clinical Triage Coordinator v1.3.1 - force redeploy
"""FastAPI app entrypoint for the Clinical Triage OpenEnv environment."""

import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import HTTPException
from pydantic import ValidationError
import uvicorn
from models import TriageAction, TriageObservation
from server.triage_environment import TASKS, TriageEnvironment


def _get_reward(observation: TriageObservation) -> float:
	return observation.step_reward


def _get_done(observation: TriageObservation) -> bool:
	return observation.episode_done


# OpenEnv expects reward/done attributes on observation models.
TriageObservation.reward = property(_get_reward)  # type: ignore[attr-defined]
TriageObservation.done = property(_get_done)  # type: ignore[attr-defined]

env = TriageEnvironment()
env.reset(task="medium")

os.environ["ENABLE_WEB_INTERFACE"] = "true"

try:
	from openenv.core.env_server import create_web_interface_app
	try:
		app = create_web_interface_app(env, TriageAction, TriageObservation)
	except TypeError:
		app = create_web_interface_app(TriageEnvironment, TriageAction, TriageObservation)
	print("Web interface enabled at /web", flush=True)
except (ImportError, Exception):
	from openenv.core.env_server import create_fastapi_app
	try:
		app = create_fastapi_app(env, TriageAction, TriageObservation)
	except TypeError:
		app = create_fastapi_app(TriageEnvironment, TriageAction, TriageObservation)
	print("Web interface not available, using standard app", flush=True)

app.state.max_concurrent_envs = 4


def _is_openenv_simulation_route(route) -> bool:
	path = getattr(route, "path", "")
	methods = getattr(route, "methods", set())
	if path == "/reset" and "POST" in methods:
		return True
	if path == "/step" and "POST" in methods:
		return True
	if path == "/state" and "GET" in methods:
		return True
	if path == "/metadata" and "GET" in methods:
		return True
	if path == "/mcp" and "POST" in methods:
		return True
	return False


app.router.routes = [
	route for route in app.router.routes if not _is_openenv_simulation_route(route)
]


@app.get("/")
async def root():
	return {
		"name": "Clinical Triage Coordinator",
		"description": "Multi-agent RL environment for Indian district hospital triage",
		"version": "1.0.0",
		"tasks": ["easy", "medium", "hard"],
		"agents": ["triage_nurse", "specialist_doctor", "resource_allocator"],
		"endpoints": {
			"reset": "POST /reset or POST /reset/{task}",
			"step": "POST /step",
			"state": "GET /state",
			"grade": "POST /grade",
			"tasks": "GET /tasks",
			"docs": "GET /docs",
			"health": "GET /health",
		},
		"rfc_compliance": ["RFC001", "RFC002", "RFC004", "RFC005"],
		"live_demo": "https://vedantpanchal23-clinical-triage-env.hf.space/docs",
		"github": "https://github.com/VedantPanchal23/clinical-triage-env",
	}


@app.get("/metadata")
async def get_metadata():
	return {
		"name": "clinical-triage-env",
		"version": "1.0.0",
		"description": (
			"Multi-agent RL environment for Indian district hospital triage. "
			"Three agents coordinate to minimize patient deterioration."
		),
		"action_space": {
			"patient_id": "str - target patient ID",
			"assigned_severity": "int 1-5 (1=Critical, 5=Non-urgent)",
			"assigned_ward": "int 1-4 (1=ICU, 2=Emergency, 3=General, 4=Discharge)",
			"treatment_protocol": "int 1-4 (1=Stabilize, 2=Medicate, 3=Refer, 4=Discharge)",
			"resource_action": "int 1-5 (1=ICU Bed, 2=General Bed, 3=Lab, 4=Staff, 5=Hold)",
		},
		"observation_space": {
			"patient_queue": "list[dict] - up to 30 patients with vitals and MEWS",
			"icu_beds_available": "int 0-2",
			"general_beds_available": "int 0-10",
			"lab_queue_length": "int 0-5",
			"staff_units_free": "int 0-5",
			"step_reward": "float 0.0-1.0",
			"episode_done": "bool",
		},
		"reward_range": [0.0, 1.0],
		"tasks": [
			{
				"name": "easy",
				"difficulty": "easy",
				"description": "Basic MEWS-guided triage of 5 patients",
				"max_steps": 10,
				"success_threshold": 0.7,
			},
			{
				"name": "medium",
				"difficulty": "medium",
				"description": "Resource-constrained triage of 10 patients",
				"max_steps": 30,
				"success_threshold": 0.6,
			},
			{
				"name": "hard",
				"difficulty": "hard",
				"description": "Mass casualty incident with 20 patients",
				"max_steps": 60,
				"success_threshold": 0.5,
			},
		],
		"rfc_compliance": {
			"RFC001": "reset/step/state baseline API",
			"RFC002": "tool discoverability via observation",
			"RFC004": "delayed trajectory rewards",
			"RFC005": "agentic harness with decision log",
		},
		"clinical_grounding": "Modified Early Warning Score (MEWS) as ground truth",
		"domain": "Healthcare - Indian District Hospital Triage",
		"agents": 3,
	}


@app.post("/reset")
async def reset_default_episode(seed: Optional[int] = None):
	obs = env.reset(task="medium", seed=seed)
	return {
		"observation": obs.model_dump(),
		"reward": 0.0,
		"done": False,
	}


@app.post("/step")
async def step_episode(payload: dict):
	action_payload = payload.get("action", payload)
	try:
		action = TriageAction.model_validate(action_payload)
	except ValidationError as exc:
		raise HTTPException(status_code=422, detail=exc.errors()) from exc

	obs = env.step(action)
	return {
		"observation": obs.model_dump(),
		"reward": obs.step_reward,
		"done": obs.episode_done,
		"info": env.info,
	}


@app.get("/capacity")
async def get_capacity():
	return {
		"max_concurrent_envs": 4,
		"current_sessions": 1,
		"status": "available",
	}


@app.get("/state")
async def get_state():
	return env.state.to_dict()


@app.get("/tasks")
async def list_tasks():
	return {"tasks": TASKS}


@app.post("/grade")
async def grade_current_episode():
	result = env.grade_task()
	# Ensure score is strictly between 0 and 1.
	if "score" in result:
		result["score"] = round(max(0.01, min(0.99, float(result["score"]))), 4)
	return result


@app.get("/grade/{task_name}")
async def grade_task_by_name(task_name: str):
	"""Grade the current episode for a specific task difficulty."""
	result = env.grade_task()
	# Ensure score is strictly between 0 and 1.
	if "score" in result:
		result["score"] = round(max(0.01, min(0.99, float(result["score"]))), 4)
	result["requested_task"] = task_name
	return result


@app.post("/reset/{task_name}")
async def reset_with_task(task_name: str, seed: Optional[int] = None):
	obs = env.reset(task=task_name, seed=seed)
	return {
		"observation": obs.model_dump(),
		"reward": 0.0,
		"done": False,
		"task": env._current_task,
	}


def main() -> None:
	uvicorn.run("server.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
	main()