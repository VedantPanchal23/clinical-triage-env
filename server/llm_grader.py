"""LLM-based medical justifiability grader for triage episode decision logs."""

from __future__ import annotations

import json
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Any

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


SYSTEM_PROMPT = """You are a medical triage expert evaluating AI agent decisions
in an Indian district hospital simulation.

Score the following triage decisions on a scale of 0-10:
- 0-2: Dangerous decisions, multiple critical errors,
		patients sent to wrong wards
- 3-4: Poor decisions, significant mismatches with MEWS scores
- 5-6: Acceptable decisions, some errors but no dangerous mistakes
- 7-8: Good decisions, mostly correct severity and ward assignments
- 9-10: Excellent decisions, all critical patients correctly
		 identified and routed, optimal resource use

For each decision evaluate:
1. Was severity correctly assigned based on MEWS score?
2. Was ward routing appropriate?
3. Was treatment protocol suitable?
4. Were resources allocated efficiently?

Respond in this EXACT JSON format:
{
  "score": <float 0-10>,
  "grade": "<Dangerous|Poor|Acceptable|Good|Excellent>",
  "critical_errors": <int>,
  "correct_decisions": <int>,
  "justification": "<2-3 sentence summary>",
  "recommendations": "<1-2 sentences on improvement>"
}

Respond with JSON only. No other text."""


@dataclass
class GradeResult:
	"""Structured output of LLM or fallback episode grading."""

	score: float
	grade: str
	critical_errors: int
	correct_decisions: int
	total_decisions: int
	justification: str
	recommendations: str
	llm_used: bool

	def to_dict(self) -> dict[str, Any]:
		"""Convert grade result to a JSON-serializable dictionary."""
		return {
			"score": self.score,
			"grade": self.grade,
			"critical_errors": self.critical_errors,
			"correct_decisions": self.correct_decisions,
			"total_decisions": self.total_decisions,
			"justification": self.justification,
			"recommendations": self.recommendations,
			"llm_used": self.llm_used,
		}


class LLMGrader:
	"""Grades episode decision quality using an LLM rubric with safe fallback."""

	def __init__(self, model: str = "claude-haiku-4-5-20251001") -> None:
		self.model = model
		self.api_key = os.getenv("ANTHROPIC_API_KEY")
		self.enabled = bool(self.api_key)
		if not self.enabled:
			warnings.warn(
				"ANTHROPIC_API_KEY not set. LLM grading disabled; using fallback grader.",
				RuntimeWarning,
			)

	async def grade_episode(self, decision_log: list[dict[str, Any]]) -> GradeResult:
		"""Grade one full episode from its decision transcript."""
		if not decision_log:
			return GradeResult(
				score=0.0,
				grade="Dangerous",
				critical_errors=0,
				correct_decisions=0,
				total_decisions=0,
				justification="No decisions were provided for grading.",
				recommendations="Provide episode decisions before requesting a grade.",
				llm_used=False,
			)

		if not self.enabled:
			fallback = self._build_fallback_grade(decision_log)
			fallback.justification = "LLM grading disabled - no API key"
			fallback.recommendations = (
				"Set ANTHROPIC_API_KEY to enable rubric-based LLM grading."
			)
			return fallback

		transcript = self._build_transcript(decision_log)
		try:
			llm_payload = await self._call_llm(transcript)
			score = float(llm_payload.get("score", 5.0))
			score = max(0.0, min(10.0, score))
			return GradeResult(
				score=score,
				grade=str(llm_payload.get("grade", self._score_to_grade(score))),
				critical_errors=int(llm_payload.get("critical_errors", 0)),
				correct_decisions=int(llm_payload.get("correct_decisions", 0)),
				total_decisions=len(decision_log),
				justification=str(
					llm_payload.get(
						"justification",
						"LLM returned no justification.",
					)
				),
				recommendations=str(
					llm_payload.get(
						"recommendations",
						"No recommendations returned.",
					)
				),
				llm_used=True,
			)
		except Exception:
			return self._build_fallback_grade(decision_log)

	def _build_transcript(self, decision_log: list[dict[str, Any]]) -> str:
		"""Convert decision log dictionaries into a readable triage transcript."""
		lines: list[str] = []
		for entry in decision_log:
			step = entry.get("step", "?")
			patient_id = entry.get("patient_id", "UNKNOWN")
			mews_score = entry.get("mews_score", "?")
			true_severity = entry.get("true_severity", "?")
			assigned_severity = entry.get("assigned_severity", "?")
			assigned_ward = entry.get("assigned_ward", "?")
			treatment_protocol = entry.get("treatment_protocol", "?")
			resource_action = entry.get("resource_action", "?")
			step_reward = entry.get("step_reward", 0)
			feedback = entry.get("feedback", "")

			line = (
				f"Step {step} | Patient {patient_id} | MEWS={mews_score} "
				f"(expected severity={true_severity}) | Assigned severity={assigned_severity}, "
				f"ward={assigned_ward}, treatment={treatment_protocol}, resource={resource_action} | "
				f"Reward={step_reward} | Feedback: {feedback}"
			)
			lines.append(line)

		return "\n".join(lines)

	async def _call_llm(self, transcript: str) -> dict[str, Any]:
		"""Call Anthropic Messages API and parse the grader JSON response."""
		if not self.api_key:
			raise RuntimeError("ANTHROPIC_API_KEY missing")

		url = "https://api.anthropic.com/v1/messages"
		headers = {
			"x-api-key": self.api_key,
			"anthropic-version": "2023-06-01",
			"content-type": "application/json",
		}
		body = {
			"model": self.model,
			"max_tokens": 500,
			"system": SYSTEM_PROMPT,
			"messages": [{"role": "user", "content": transcript}],
		}

		async with httpx.AsyncClient(timeout=30.0) as client:
			response = await client.post(url, headers=headers, json=body)
			response.raise_for_status()
			data = response.json()

		try:
			text = data["content"][0]["text"]
			return json.loads(text)
		except (KeyError, IndexError, TypeError, json.JSONDecodeError):
			return {
				"score": 5.0,
				"grade": "Acceptable",
				"critical_errors": 0,
				"correct_decisions": 0,
				"justification": "LLM response could not be parsed as JSON.",
				"recommendations": "Ensure model returns strict JSON per prompt.",
			}

	def _build_fallback_grade(self, decision_log: list[dict[str, Any]]) -> GradeResult:
		"""Compute a deterministic grade when LLM scoring is unavailable."""
		total = len(decision_log)
		if total == 0:
			return GradeResult(
				score=0.0,
				grade="Dangerous",
				critical_errors=0,
				correct_decisions=0,
				total_decisions=0,
				justification="No decisions were provided for fallback grading.",
				recommendations="Collect episode decisions to evaluate quality.",
				llm_used=False,
			)

		correct = sum(1 for entry in decision_log if float(entry.get("step_reward", 0)) > 0)
		score = (correct / total) * 10.0
		grade = self._score_to_grade(score)

		return GradeResult(
			score=round(score, 2),
			grade=grade,
			critical_errors=0,
			correct_decisions=correct,
			total_decisions=total,
			justification=(
				f"Fallback grading used positive step rewards: {correct} of {total} "
				"decisions were rewarded."
			),
			recommendations=(
				"Improve consistency on severity assignment and ward routing for high-risk "
				"patients."
			),
			llm_used=False,
		)

	@staticmethod
	def _score_to_grade(score: float) -> str:
		"""Map numeric score to categorical grade."""
		if score >= 9.0:
			return "Excellent"
		if score >= 7.0:
			return "Good"
		if score >= 5.0:
			return "Acceptable"
		if score >= 3.0:
			return "Poor"
		return "Dangerous"


async def grade_episode(
	decision_log: list[dict[str, Any]],
	model: str = "claude-haiku-4-5-20251001",
) -> GradeResult:
	"""Convenience function to grade one episode without manual class wiring."""
	grader = LLMGrader(model=model)
	return await grader.grade_episode(decision_log)
