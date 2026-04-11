# Clinical Triage Coordinator v1.3.1 - force redeploy
"""FastAPI app entrypoint for the Clinical Triage OpenEnv environment."""

import os
import sys
from typing import Optional
from uuid import uuid4

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from pydantic import ValidationError
import uvicorn
from models import TriageAction, TriageObservation
from server.web_interface import get_dashboard_html
from server.triage_environment import TASKS, TriageEnvironment


def _get_reward(observation: TriageObservation) -> float:
	return observation.step_reward


def _get_done(observation: TriageObservation) -> bool:
	return observation.episode_done


# OpenEnv expects reward/done attributes on observation models.
TriageObservation.reward = property(_get_reward)  # type: ignore[attr-defined]
TriageObservation.done = property(_get_done)  # type: ignore[attr-defined]


MAX_CONCURRENT_ENVS = 4
SESSION_COOKIE_NAME = "triage_session_id"

def make_env():
	"""Factory - creates a fresh TriageEnvironment per session."""
	return TriageEnvironment()


_fallback_env = TriageEnvironment()
_fallback_env.reset(task="medium")
_sessions: dict[str, TriageEnvironment] = {"default": _fallback_env}


def _active_session_count() -> int:
	return max(0, len(_sessions) - 1)


def _resolve_session_id(request: Request) -> str:
	return (
		request.headers.get("X-Session-ID")
		or request.query_params.get("session_id")
		or request.cookies.get(SESSION_COOKIE_NAME)
		or "default"
	)


def _attach_session_cookie(response: Response, session_id: str) -> None:
	response.set_cookie(
		key=SESSION_COOKIE_NAME,
		value=session_id,
		httponly=True,
		samesite="lax",
	)


def _get_or_create_session_env(session_id: str) -> TriageEnvironment:
	env = _sessions.get(session_id)
	if env is not None:
		return env

	if _active_session_count() >= MAX_CONCURRENT_ENVS:
		raise HTTPException(
			status_code=429,
			detail=(
				f"Maximum concurrent sessions reached ({MAX_CONCURRENT_ENVS}). "
				"Reuse an existing session_id or wait for a session to end."
			),
		)

	env = make_env()
	_sessions[session_id] = env
	return env


def get_env(request: Request) -> tuple[str, TriageEnvironment]:
	session_id = _resolve_session_id(request)
	env = _sessions.get(session_id)
	if env is None:
		raise HTTPException(
			status_code=404,
			detail="Unknown session_id. Call /reset first to create a session.",
		)
	return session_id, env


os.environ["ENABLE_WEB_INTERFACE"] = "true"

try:
    from openenv.core.env_server import create_web_interface_app
    try:
        app = create_web_interface_app(make_env, TriageAction, TriageObservation)
        print("Web interface enabled at /web (session factory)", flush=True)
    except TypeError:
        from openenv.core.env_server import create_fastapi_app

        try:
            app = create_fastapi_app(make_env, TriageAction, TriageObservation)
            print("Session-isolated env factory registered", flush=True)
        except TypeError:
            app = create_fastapi_app(_fallback_env, TriageAction, TriageObservation)
            print("Single env instance (factory not supported)", flush=True)
except (ImportError, Exception):
    from openenv.core.env_server import create_fastapi_app

    try:
        app = create_fastapi_app(make_env, TriageAction, TriageObservation)
        print("Session-isolated env factory registered", flush=True)
    except TypeError:
        app = create_fastapi_app(_fallback_env, TriageAction, TriageObservation)
        print("Single env instance (factory not supported)", flush=True)

app.state.max_concurrent_envs = MAX_CONCURRENT_ENVS


def _is_openenv_simulation_route(route) -> bool:
	path = getattr(route, "path", "")
	methods = getattr(route, "methods", set())
	if path == "/":
		return True
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
	if path == "/web" or path.startswith("/web/"):
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
			"web_ui": "GET /web - Interactive visual dashboard",
			"docs": "GET /docs",
			"health": "GET /health",
		},
		"rfc_compliance": ["RFC001", "RFC002", "RFC004", "RFC005"],
		"live_demo": "https://vedantpanchal23-clinical-triage-env.hf.space/docs",
		"github": "https://github.com/VedantPanchal23/clinical-triage-env",
	}


@app.get("/web", response_class=HTMLResponse)
async def web_dashboard():
	return get_dashboard_html()


@app.get("/web/", response_class=HTMLResponse)
async def web_dashboard_slash():
	return get_dashboard_html()


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
async def reset_default_episode(request: Request, response: Response, seed: Optional[int] = None):
	requested_session_id = request.headers.get("X-Session-ID") or request.query_params.get("session_id")
	session_id = requested_session_id or str(uuid4())
	env = _get_or_create_session_env(session_id)
	obs = env.reset(task="medium", seed=seed)
	_attach_session_cookie(response, session_id)
	return {
		"observation": obs.model_dump(),
		"reward": 0.0,
		"done": False,
		"session_id": session_id,
	}


@app.post("/step")
async def step_episode(request: Request, response: Response, payload: dict):
	action_payload = payload.get("action", payload)
	try:
		action = TriageAction.model_validate(action_payload)
	except ValidationError as exc:
		raise HTTPException(status_code=422, detail=exc.errors()) from exc

	session_id, env = get_env(request)
	obs = env.step(action)
	_attach_session_cookie(response, session_id)
	return {
		"observation": obs.model_dump(),
		"reward": obs.step_reward,
		"done": obs.episode_done,
		"info": env.info,
		"session_id": session_id,
	}


@app.get("/capacity")
async def get_capacity():
	return {
		"max_concurrent_envs": MAX_CONCURRENT_ENVS,
		"current_sessions": _active_session_count(),
		"status": "available",
	}


@app.get("/state")
async def get_state(request: Request, response: Response):
	session_id, env = get_env(request)
	state = env.state.to_dict()
	state["session_id"] = session_id
	_attach_session_cookie(response, session_id)
	return state


@app.get("/tasks")
async def list_tasks():
	return {"tasks": TASKS}


@app.post("/grade")
async def grade_current_episode(request: Request, response: Response):
	session_id, env = get_env(request)
	result = env.grade_task()
	# Ensure score is strictly between 0 and 1.
	if "score" in result:
		result["score"] = round(max(0.01, min(0.99, float(result["score"]))), 4)
	result["session_id"] = session_id
	_attach_session_cookie(response, session_id)
	return result


@app.get("/grade/{task_name}")
async def grade_task_by_name(task_name: str, request: Request, response: Response):
	"""Grade the current episode for a specific task difficulty."""
	session_id, env = get_env(request)
	result = env.grade_task()
	# Ensure score is strictly between 0 and 1.
	if "score" in result:
		result["score"] = round(max(0.01, min(0.99, float(result["score"]))), 4)
	result["requested_task"] = task_name
	result["session_id"] = session_id
	_attach_session_cookie(response, session_id)
	return result


@app.post("/reset/{task_name}")
async def reset_with_task(task_name: str, request: Request, response: Response, seed: Optional[int] = None):
	requested_session_id = request.headers.get("X-Session-ID") or request.query_params.get("session_id")
	session_id = requested_session_id or str(uuid4())
	env = _get_or_create_session_env(session_id)
	obs = env.reset(task=task_name, seed=seed)
	_attach_session_cookie(response, session_id)
	return {
		"observation": obs.model_dump(),
		"reward": 0.0,
		"done": False,
		"task": env._current_task,
		"session_id": session_id,
	}


def main() -> None:
	uvicorn.run("server.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
	main()