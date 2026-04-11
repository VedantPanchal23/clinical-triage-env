"""FastAPI app entrypoint for the Clinical Triage OpenEnv environment."""

import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import HTTPException
from fastapi.responses import RedirectResponse
from openenv.core.env_server import create_fastapi_app
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

app = create_fastapi_app(TriageEnvironment, TriageAction, TriageObservation)
env = TriageEnvironment()
env.reset(task="medium")


def _is_openenv_simulation_route(route) -> bool:
	path = getattr(route, "path", "")
	methods = getattr(route, "methods", set())
	if path == "/reset" and "POST" in methods:
		return True
	if path == "/step" and "POST" in methods:
		return True
	if path == "/state" and "GET" in methods:
		return True
	if path == "/mcp" and "POST" in methods:
		return True
	return False


app.router.routes = [
	route for route in app.router.routes if not _is_openenv_simulation_route(route)
]


@app.get("/", include_in_schema=False)
async def root_redirect():
	return RedirectResponse(url="/docs")


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
	}


@app.get("/state")
async def get_state():
	return env.state.to_dict()


@app.get("/tasks")
async def list_tasks():
	return {"tasks": TASKS}


@app.post("/grade")
async def grade_current_episode():
	return env.grade_task()


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