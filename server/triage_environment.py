"""
Triage Environment — Clinical Triage Coordinator
Main OpenEnv Environment implementation.

Multi-agent design (turn-based, single step() call):
  Each step() receives one TriageAction targeting one patient.
  The action encodes decisions from all 3 agents simultaneously:
    - Triage Nurse   → assigned_severity + assigned_ward
    - Doctor         → treatment_protocol
    - Resource Alloc → resource_action

RFC 004: Delayed trajectory reward computed at episode end.
RFC 002: Available tools exposed via observation feedback.
RFC 005: Decision log maintained for LLM grader.
"""

from __future__ import annotations
import uuid
import random
from typing import Optional

from openenv.core.env_server import Environment

from models import (
    PatientState,
    TriageAction,
    TriageObservation,
    TriageState,
    Severity,
    Ward,
    TreatmentProtocol,
    ResourceAction,
)
from server.patient_generator import generate_initial_queue, generate_patient
from server.mews_scorer import (
    score_triage_decision,
    check_sla_breach,
    mews_to_severity_int,
)


TASKS = {
    "easy": {
        "name": "Basic Triage",
        "description": (
            "Correctly triage 5 patients by assigning severity scores "
            "that match MEWS ground truth. All patients have clear-cut "
            "vital signs. Succeed by achieving >= 0.7 average triage accuracy."
        ),
        "max_steps": 10,
        "target_patients": 5,
        "success_threshold": 0.7,
        "difficulty": "easy",
    },
    "medium": {
        "name": "Resource Constrained Triage",
        "description": (
            "Triage 10 patients while managing scarce resources: "
            "only 2 ICU beds and 5 staff units. Correctly prioritize "
            "critical patients for ICU. Succeed with zero patient deaths "
            "and >= 0.6 stabilization rate."
        ),
        "max_steps": 30,
        "target_patients": 10,
        "success_threshold": 0.6,
        "difficulty": "medium",
    },
    "hard": {
        "name": "Mass Casualty Incident",
        "description": (
            "Handle a mass casualty scenario with 20 simultaneous "
            "patients including multiple critical cases arriving "
            "continuously every step. Strict resource limits: only "
            "1 ICU bed available. Agents must prioritize correctly "
            "or patients die. Succeed with >= 0.5 stabilization "
            "rate and < 2 deaths."
        ),
        "max_steps": 40,
        "target_patients": 20,
        "success_threshold": 0.5,
        "difficulty": "hard",
        "icu_beds": 1,
        "new_patient_prob": 0.6,
        "deterioration_steps": 1,
    },
}


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_QUEUE_SIZE      = 30
INITIAL_QUEUE_SIZE  = 8
MAX_STEPS           = 100
NEW_PATIENT_PROB    = 0.3    # probability a new patient arrives each step
ICU_BEDS_TOTAL      = 2
GENERAL_BEDS_TOTAL  = 10
STAFF_UNITS_TOTAL   = 5
LAB_SLOTS_TOTAL     = 5

# Deterioration: untreated critical patient worsens every N steps
DETERIORATION_STEPS = 3


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class TriageEnvironment(Environment):
    """
    OpenEnv Environment for Indian district hospital triage coordination.
    Implements reset(), step(), and state property.
    """

    def __init__(self) -> None:
        super().__init__()
        self._state = TriageState()
        self._queue: list[PatientState] = []
        self._current_task = TASKS["medium"]
        self._last_info: dict = {}
        self._hard_mode = False

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self, task: str = "medium", seed: Optional[int] = None) -> TriageObservation:
        """Start a new episode with a fresh patient queue."""
        if seed is not None:
            import random
            random.seed(seed)

        task_config = TASKS.get(task)
        if task_config is None:
            # Backward-compatible default used by legacy clients and tests.
            self._current_task = TASKS["medium"]
            initial_queue_size = INITIAL_QUEUE_SIZE
            max_steps = MAX_STEPS
        else:
            self._current_task = task_config
            difficulty = self._current_task["difficulty"]
            if difficulty == "hard":
                initial_queue_size = 20
            elif difficulty == "easy":
                initial_queue_size = 5
            else:
                initial_queue_size = 10
            max_steps = self._current_task["max_steps"]

        self._state = TriageState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            max_steps=max_steps,
            icu_beds_available=ICU_BEDS_TOTAL,
            general_beds_available=GENERAL_BEDS_TOTAL,
            staff_units_free=STAFF_UNITS_TOTAL,
            lab_queue_length=0,
        )

        if self._current_task["difficulty"] == "hard":
            self._state.icu_beds_available = self._current_task.get("icu_beds", 1)
            self._hard_mode = True
        else:
            self._state.icu_beds_available = ICU_BEDS_TOTAL
            self._hard_mode = False

        self._queue = generate_initial_queue(initial_queue_size)
        self._last_info = self._build_info(
            "RESET",
            "Episode started. Triage coordinator active.",
        )

        return self._build_observation(
            last_patient_id="RESET",
            valid=True,
            feedback="Episode started. Triage coordinator active.",
            step_reward=0.0,
        )

    def reset_default(self) -> TriageObservation:
        """Legacy default reset used by /reset endpoint for compatibility tests."""
        return self.reset(task="default")

    def step(self, action: TriageAction) -> TriageObservation:
        """
        Execute one triage action and advance the simulation by one step.

        Flow:
          1. Validate action
          2. Apply triage nurse decision (severity + ward)
          3. Apply doctor decision (treatment)
          4. Apply resource allocator decision
          5. Compute step reward
          6. Advance simulation (queue aging, deterioration, new arrivals)
          7. Check episode termination
          8. Return observation
        """
        self._state.step_count += 1
        step_reward = 0.0
        feedback_parts = []

        # --- 1. Find patient ---
        patient = self._find_patient(action.patient_id)
        if patient is None:
            invalid_feedback = f"Patient {action.patient_id} not found in queue."
            self._last_info = self._build_info(action.patient_id, invalid_feedback)
            return self._build_observation(
                last_patient_id=action.patient_id,
                valid=False,
                feedback=invalid_feedback,
                step_reward=self._normalize_reward(-0.5),
            )

        # --- 2. Triage nurse: severity + ward ---
        triage_reward, triage_feedback = score_triage_decision(
            assigned_severity=action.assigned_severity,
            assigned_ward=action.assigned_ward,
            mews_total=patient.mews_score,
        )
        step_reward += triage_reward
        feedback_parts.append(triage_feedback)

        patient.assigned_ward = action.assigned_ward

        # --- 3. Doctor: treatment protocol ---
        if action.treatment_protocol is not None:
            treatment_reward, treatment_feedback = self._apply_treatment(
                patient, action.treatment_protocol
            )
            step_reward += treatment_reward
            feedback_parts.append(treatment_feedback)
        else:
            feedback_parts.append("No treatment assigned yet")

        # --- 4. Resource allocator ---
        resource_reward, resource_feedback = self._apply_resource(
            patient, action.resource_action, action.assigned_ward
        )
        step_reward += resource_reward
        feedback_parts.append(resource_feedback)

        # --- 5. SLA breach check ---
        if check_sla_breach(patient.true_severity, patient.time_in_queue):
            step_reward -= 0.3
            feedback_parts.append(
                f"SLA breach: {patient.patient_id} waited {patient.time_in_queue} steps"
            )

        # --- 6. Log decision for LLM grader (RFC 005) ---
        self._state.decision_log.append({
            "step":               self._state.step_count,
            "patient_id":         patient.patient_id,
            "mews_score":         patient.mews_score,
            "true_severity":      patient.true_severity,
            "assigned_severity":  action.assigned_severity,
            "assigned_ward":      action.assigned_ward,
            "treatment_protocol": action.treatment_protocol,
            "resource_action":    action.resource_action,
            "step_reward":        round(step_reward, 3),
            "feedback":           " | ".join(feedback_parts),
        })

        # --- 7. Advance simulation ---
        self._advance_simulation()

        # --- 8. Check done ---
        done = self._check_done()
        if done:
            trajectory_reward = self._compute_trajectory_reward()
            self._state.trajectory_reward = trajectory_reward
            step_reward += trajectory_reward

        step_reward = self._normalize_reward(step_reward)

        feedback = " | ".join(feedback_parts)
        self._last_info = self._build_info(action.patient_id, feedback)

        obs = self._build_observation(
            last_patient_id=action.patient_id,
            valid=True,
            feedback=feedback,
            step_reward=step_reward,
        )
        obs.episode_done = done
        return obs

    @property
    def state(self) -> TriageState:
        """Return current episode state (RFC 001)."""
        return self._state

    @property
    def info(self) -> dict:
        return getattr(self, "_last_info", {})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_info(self, patient_id: str, feedback: str) -> dict:
        return {
            "patient_id": patient_id,
            "feedback": feedback,
            "mews_available": True,
            "episode_id": self._state.episode_id,
            "task": getattr(self, "_current_task", {}).get("name", "unknown"),
            "resources": {
                "icu_beds": self._state.icu_beds_available,
                "general_beds": self._state.general_beds_available,
                "staff": self._state.staff_units_free,
                "lab_slots": LAB_SLOTS_TOTAL - self._state.lab_queue_length,
            },
        }

    def _find_patient(self, patient_id: str) -> Optional[PatientState]:
        for p in self._queue:
            if p.patient_id == patient_id:
                return p
        return None

    def _apply_treatment(
        self, patient: PatientState, protocol: int
    ) -> tuple[float, str]:
        """Apply doctor treatment and return (reward, feedback)."""
        reward = 0.0
        feedback = ""

        if protocol == TreatmentProtocol.STABILIZE:
            if patient.true_severity <= 2:
                # Correct: critical patient needs stabilization
                patient.is_stabilized = True
                patient.is_discharged = True
                reward = 1.5
                feedback = f"STABILIZE applied to severity-{patient.true_severity} patient — correct"
                self._state.patients_stabilized += 1
                self._remove_patient(patient)
            else:
                # Over-treatment, minor waste but not harmful
                reward = -0.1
                feedback = "STABILIZE on non-critical patient — over-treatment"

        elif protocol == TreatmentProtocol.MEDICATE:
            if patient.true_severity in (2, 3, 4):
                patient.treatment_steps += 1
                reward = 0.8
                feedback = "MEDICATE applied — appropriate"
                if patient.treatment_steps >= 2:
                    patient.is_stabilized = True
                    patient.is_discharged = True
                    self._state.patients_stabilized += 1
                    self._remove_patient(patient)
            else:
                reward = 0.2
                feedback = "MEDICATE applied — acceptable"

        elif protocol == TreatmentProtocol.REFER:
            if patient.true_severity <= 3:
                reward = 0.5
                feedback = "REFER — appropriate for complex case"
                patient.is_discharged = True
                self._remove_patient(patient)
            else:
                reward = -0.2
                feedback = "REFER for non-urgent case — inefficient"

        elif protocol == TreatmentProtocol.DISCHARGE:
            if patient.true_severity >= 4:
                reward = 1.0
                feedback = "DISCHARGE — correct for non-urgent patient"
                patient.is_discharged = True
                self._remove_patient(patient)
            elif patient.true_severity == 3:
                reward = -0.3
                feedback = "DISCHARGE for URGENT patient — risky"
            else:
                reward = -2.0
                feedback = "DISCHARGE for CRITICAL/EMERGENCY — dangerous!"
                patient.is_deteriorated = True
                self._state.patients_deteriorated += 1

            if getattr(self, "_hard_mode", False) and patient.mews_score >= 7:
                if not patient.is_deteriorated:
                    patient.is_deteriorated = True
                    self._state.patients_deteriorated += 1
                reward -= 0.5
                feedback += " | Hard mode escalation: critical discharge penalized"

        return round(reward, 2), feedback

    def _apply_resource(
        self, patient: PatientState, resource_action: int, assigned_ward: int
    ) -> tuple[float, str]:
        """Apply resource allocator decision."""
        reward = 0.0
        feedback = ""

        if resource_action == ResourceAction.ASSIGN_ICU_BED:
            if assigned_ward == Ward.ICU:
                if self._state.icu_beds_available > 0:
                    self._state.icu_beds_available -= 1
                    reward = 0.5
                    feedback = f"ICU bed assigned ({self._state.icu_beds_available} remaining)"
                else:
                    reward = -0.3
                    feedback = "ICU bed requested but none available — overflow!"
                    self._state.queue_overflow_count += 1
            else:
                reward = -0.2
                feedback = "ICU bed assigned to non-ICU patient — resource waste"

        elif resource_action == ResourceAction.ASSIGN_GENERAL_BED:
            if assigned_ward in (Ward.GENERAL, Ward.EMERGENCY):
                if self._state.general_beds_available > 0:
                    self._state.general_beds_available -= 1
                    reward = 0.3
                    feedback = f"General bed assigned ({self._state.general_beds_available} remaining)"
                else:
                    reward = -0.2
                    feedback = "No general beds available"
            else:
                reward = -0.1
                feedback = "General bed mismatch with ward assignment"

        elif resource_action == ResourceAction.SCHEDULE_LAB:
            if self._state.lab_queue_length < LAB_SLOTS_TOTAL:
                self._state.lab_queue_length += 1
                reward = 0.2
                feedback = f"Lab scheduled (queue: {self._state.lab_queue_length})"
            else:
                reward = -0.1
                feedback = "Lab queue full"

        elif resource_action == ResourceAction.ALLOCATE_STAFF:
            if self._state.staff_units_free > 0:
                self._state.staff_units_free -= 1
                reward = 0.3
                feedback = f"Staff allocated ({self._state.staff_units_free} units free)"
            else:
                reward = -0.2
                feedback = "No staff units available"

        elif resource_action == ResourceAction.HOLD:
            reward = 0.0
            feedback = "No resource assigned (HOLD)"

        return round(reward, 2), feedback

    def _remove_patient(self, patient: PatientState) -> None:
        """Remove discharged/stabilized patient from queue."""
        self._queue = [p for p in self._queue if p.patient_id != patient.patient_id]

    def _advance_simulation(self) -> None:
        """
        Advance all patients by one step:
          - Increment time_in_queue for waiting patients
          - Deteriorate untreated critical patients
          - Possibly admit a new patient
          - Free lab slot occasionally
        """
        deterioration_steps = (
            self._current_task.get("deterioration_steps", 2)
            if getattr(self, "_hard_mode", False)
            else DETERIORATION_STEPS
        )
        new_patient_prob = (
            self._current_task.get("new_patient_prob", 0.6)
            if getattr(self, "_hard_mode", False)
            else NEW_PATIENT_PROB
        )

        for patient in self._queue:
            patient.time_in_queue += 1

            # Deterioration: untreated critical/emergency patients worsen
            if (
                patient.true_severity <= 2
                and patient.time_in_queue > 0
                and patient.time_in_queue % deterioration_steps == 0
            ):
                needs_critical_care = patient.assigned_ward is None
                if getattr(self, "_hard_mode", False):
                    needs_critical_care = (
                        needs_critical_care
                        or patient.assigned_ward != Ward.ICU
                    )

                if needs_critical_care:
                    patient.is_deteriorated = True
                    self._state.patients_deteriorated += 1
                    # Escalate MEWS slightly to reflect worsening
                    patient.mews_score = min(patient.mews_score + 2, 17)

        # Remove deceased/critically deteriorated patients
        before = len(self._queue)
        self._queue = [
            p for p in self._queue
            if not (p.is_deteriorated and p.time_in_queue > deterioration_steps * 2)
        ]
        deceased = before - len(self._queue)
        self._state.patients_deceased += deceased

        # New patient arrival
        if (
            len(self._queue) < MAX_QUEUE_SIZE
            and random.random() < new_patient_prob
        ):
            new_patient = generate_patient()
            new_patient.patient_id = f"PT-NEW-{self._state.step_count:03d}"
            self._queue.append(new_patient)

        # Free one lab slot every 3 steps
        if self._state.step_count % 3 == 0 and self._state.lab_queue_length > 0:
            self._state.lab_queue_length -= 1

    def _check_done(self) -> bool:
        """Episode ends when max steps reached or queue is empty."""
        task = getattr(self, '_current_task', {})
        max_steps = task.get('max_steps', MAX_STEPS)
        if self._state.step_count >= max_steps:
            return True
        if len(self._queue) == 0:
            return True
        return False

    def _normalize_reward(self, raw_reward: float) -> float:
        """
        Normalize raw step reward to 0.0-1.0 range.
        Raw step rewards range from about -3.0 to +3.5
        We shift and scale to 0.0-1.0
        """
        MIN_RAW = -3.0
        MAX_RAW = 3.5
        normalized = (raw_reward - MIN_RAW) / (MAX_RAW - MIN_RAW)
        return round(max(0.01, min(0.99, normalized)), 3)

    def _compute_trajectory_reward(self) -> float:
        """
        RFC 004 — Delayed trajectory-level reward.
        Computed once at episode end. This is the PRIMARY reward signal.

        Components:
          +5.0 per patient stabilized
          -3.0 per patient deteriorated
          -8.0 per patient deceased
          -0.5 per queue overflow event
          +2.0 bonus if zero deaths
          +1.0 bonus if ICU utilization > 50%
        """
        s = self._state
        reward = 0.0

        reward += s.patients_stabilized   * 5.0
        reward -= s.patients_deteriorated * 3.0
        reward -= s.patients_deceased     * 8.0
        reward -= s.queue_overflow_count  * 0.5

        if s.patients_deceased == 0:
            reward += 2.0

        icu_used = ICU_BEDS_TOTAL - s.icu_beds_available
        if icu_used >= 1:
            reward += 1.0

        MIN_TRAJ = -50.0
        MAX_TRAJ = 50.0
        normalized = (reward - MIN_TRAJ) / (MAX_TRAJ - MIN_TRAJ)
        return round(max(0.01, min(0.99, normalized)), 3)

    def grade_task(self) -> dict:
        s = self._state
        task = self._current_task
        difficulty = task["difficulty"]

        stabilized = s.patients_stabilized
        deteriorated = s.patients_deteriorated
        deceased = s.patients_deceased
        overflows = s.queue_overflow_count
        total_treated = stabilized + deteriorated + deceased

        if total_treated == 0:
            # No patients treated yet — base score on step rewards
            # from decision_log so grader never returns 0 mid-episode
            if s.decision_log:
                avg_step = sum(
                    e.get("step_reward", 0) for e in s.decision_log
                ) / len(s.decision_log)
                score = round(max(0.01, min(0.99, avg_step)), 3)
            else:
                score = 0.01
            return {
                "task_name": task["name"],
                "difficulty": difficulty,
                "score": score,
                "success": score >= task["success_threshold"],
                "patients_stabilized": 0,
                "patients_deteriorated": 0,
                "patients_deceased": 0,
                "stabilization_rate": 0.0,
                "steps_taken": s.step_count,
                "reason": "Graded on step rewards (no completed treatments yet)",
            }

        stabilization_rate = stabilized / max(total_treated, 1)
        death_penalty = deceased * 0.15
        overflow_penalty = overflows * 0.05
        raw_score = stabilization_rate - death_penalty - overflow_penalty

        # Step reward bonus — reward good decisions even if
        # not all patients are formally discharged
        if s.decision_log:
            avg_step_reward = sum(
                e.get("step_reward", 0) for e in s.decision_log
            ) / len(s.decision_log)
            # Blend: 60% outcome-based + 40% step-reward-based
            raw_score = 0.6 * raw_score + 0.4 * avg_step_reward

        score = round(max(0.01, min(0.99, raw_score)), 3)

        if difficulty == "medium" and deceased > 0:
            score = min(score, 0.5)

        if difficulty == "hard" and s.step_count < task.get("max_steps", MAX_STEPS):
            progress_factor = max(
                0.4,
                s.step_count / max(task.get("max_steps", MAX_STEPS), 1),
            )
            score = round(score * progress_factor, 3)

        if difficulty == "easy":
            success = score >= task["success_threshold"]
        elif difficulty == "medium":
            success = score >= task["success_threshold"] and deceased == 0
        else:  # hard
            success = score >= task["success_threshold"] and deceased < 2

        return {
            "task_name": task["name"],
            "difficulty": difficulty,
            "score": score,
            "success": success,
            "patients_stabilized": stabilized,
            "patients_deteriorated": deteriorated,
            "patients_deceased": deceased,
            "stabilization_rate": round(stabilization_rate, 3),
            "steps_taken": s.step_count,
            "reason": f"Stabilized {stabilized}/{total_treated} treated patients",
        }

    def _build_observation(
        self,
        last_patient_id: str,
        valid: bool,
        feedback: str,
        step_reward: float,
    ) -> TriageObservation:
        """Build the TriageObservation returned to the agent."""
        s = self._state
        return TriageObservation(
            patient_queue=[p.to_obs_dict() for p in self._queue],
            icu_beds_available=s.icu_beds_available,
            general_beds_available=s.general_beds_available,
            lab_queue_length=s.lab_queue_length,
            staff_units_free=s.staff_units_free,
            last_action_patient_id=last_patient_id,
            last_action_valid=valid,
            last_action_feedback=feedback,
            step_reward=step_reward,
            step_count=s.step_count,
            patients_stabilized=s.patients_stabilized,
            patients_deteriorated=s.patients_deteriorated,
            patients_deceased=s.patients_deceased,
            episode_done=False,
        )