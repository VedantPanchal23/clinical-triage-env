"""OpenEnv client for interacting with the Clinical Triage Coordinator server."""

from __future__ import annotations

from typing import Any

import httpx
from openenv.core import EnvClient

try:
	from openenv.core import StepResult
except ImportError:  # openenv-core 0.2.3 fallback
	from openenv.core.env_client import StepResult

from models import TriageAction, TriageObservation, TriageState


class TriageEnv(EnvClient[TriageAction, TriageObservation, TriageState]):
	"""OpenEnv HTTP client for the Clinical Triage Coordinator environment.

	This client translates typed action/state/observation objects to and from
	the OpenEnv server payload format so training loops can interact with the
	environment safely.

	It coordinates three collaborative agent roles per step:
	1. Triage nurse: assigns severity and destination ward.
	2. Doctor: selects a treatment protocol.
	3. Resource allocator: assigns scarce hospital resources.

	Async example:
		from client import TriageEnv
		from models import TriageAction, ResourceAction

		async with TriageEnv(base_url="http://localhost:8000") as client:
			result = await client.reset()
			obs = result.observation
			first_patient = obs.patient_queue[0]
			action = TriageAction(
				patient_id=first_patient["patient_id"],
				assigned_severity=1,
				assigned_ward=1,
				treatment_protocol=1,
				resource_action=ResourceAction.ASSIGN_ICU_BED,
			)
			result2 = await client.step(action)
			print(result2.reward)

	Sync example:
		from client import TriageEnv
		from models import TriageAction, ResourceAction

		with TriageEnv(base_url="http://localhost:8000").sync() as client:
			result = client.reset()
			obs = result.observation
			first_patient = obs.patient_queue[0]
			action = TriageAction(
				patient_id=first_patient["patient_id"],
				assigned_severity=1,
				assigned_ward=1,
				treatment_protocol=1,
				resource_action=ResourceAction.ASSIGN_ICU_BED,
			)
			result2 = client.step(action)
			print(result2.reward)
	"""

	_last_observation: TriageObservation | None = None

	def _step_payload(self, action: TriageAction) -> dict[str, Any]:
		"""Serialize a typed action into the JSON payload expected by /step."""
		return action.model_dump()

	def _parse_result(self, payload: dict[str, Any]) -> StepResult[TriageObservation]:
		"""Parse /reset or /step response payload into a typed StepResult."""
		observation = TriageObservation(**payload["observation"])
		self._last_observation = observation
		return StepResult(
			observation=observation,
			reward=payload["reward"],
			done=payload["done"],
		)

	def _parse_state(self, payload: dict[str, Any]) -> TriageState:
		"""Parse /state payload into TriageState dataclass fields."""
		return TriageState(
			episode_id=payload.get("episode_id", ""),
			step_count=payload.get("step_count", 0),
			max_steps=payload.get("max_steps", 100),
			patients_stabilized=payload.get("patients_stabilized", 0),
			patients_deteriorated=payload.get("patients_deteriorated", 0),
			patients_deceased=payload.get("patients_deceased", 0),
			queue_overflow_count=payload.get("queue_overflow_count", 0),
			icu_beds_available=payload.get("icu_beds_available", 2),
			general_beds_available=payload.get("general_beds_available", 10),
			lab_queue_length=payload.get("lab_queue_length", 0),
			staff_units_free=payload.get("staff_units_free", 5),
			trajectory_reward=payload.get("trajectory_reward", 0.0),
		)

	def _http_state_url(self) -> str:
		"""Convert internal websocket URL to the HTTP /state endpoint URL."""
		ws_base = self._ws_url.removesuffix("/ws")
		if ws_base.startswith("ws://"):
			http_base = f"http://{ws_base[len('ws://'):] }"
		elif ws_base.startswith("wss://"):
			http_base = f"https://{ws_base[len('wss://'):] }"
		else:
			http_base = ws_base
		return f"{http_base}/state"

	async def state(self) -> TriageState:
		"""Get state via WebSocket, with HTTP fallback for OpenEnv compatibility."""
		try:
			return await super().state()
		except RuntimeError as exc:
			if "Server error:" not in str(exc):
				raise

		async with httpx.AsyncClient(timeout=10.0) as client:
			response = await client.get(self._http_state_url())
			response.raise_for_status()
			payload = response.json()

		state = self._parse_state(payload)
		if self._last_observation is not None and state.step_count == 0:
			state.step_count = self._last_observation.step_count
			state.patients_stabilized = self._last_observation.patients_stabilized
			state.patients_deteriorated = self._last_observation.patients_deteriorated
			state.patients_deceased = self._last_observation.patients_deceased
			state.icu_beds_available = self._last_observation.icu_beds_available
			state.general_beds_available = self._last_observation.general_beds_available
			state.lab_queue_length = self._last_observation.lab_queue_length
			state.staff_units_free = self._last_observation.staff_units_free
		return state
