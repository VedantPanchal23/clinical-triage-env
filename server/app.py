"""FastAPI app entrypoint for the Clinical Triage OpenEnv environment."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server import create_fastapi_app
from models import TriageAction, TriageObservation
from server.triage_environment import TriageEnvironment


def _get_reward(observation: TriageObservation) -> float:
	return observation.step_reward


def _get_done(observation: TriageObservation) -> bool:
	return observation.episode_done


# OpenEnv expects reward/done attributes on observation models.
TriageObservation.reward = property(_get_reward)  # type: ignore[attr-defined]
TriageObservation.done = property(_get_done)  # type: ignore[attr-defined]

env = TriageEnvironment
app = create_fastapi_app(env, TriageAction, TriageObservation)