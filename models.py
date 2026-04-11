"""
Clinical Triage Coordinator — Data Models
All Actions, Observations, and State dataclasses for the environment.
Pydantic v2 + OpenEnv compatible.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Severity(IntEnum):
    """MEWS-aligned triage severity levels (1 = critical, 5 = minor)."""
    CRITICAL   = 1
    EMERGENCY  = 2
    URGENT     = 3
    SEMI_URGENT = 4
    NON_URGENT = 5


class Ward(IntEnum):
    """Hospital ward destinations."""
    ICU        = 1
    EMERGENCY  = 2
    GENERAL    = 3
    DISCHARGE  = 4


class TreatmentProtocol(IntEnum):
    """Doctor treatment options."""
    STABILIZE  = 1   # immediate life-saving intervention
    MEDICATE   = 2   # administer medication
    REFER      = 3   # refer to specialist
    DISCHARGE  = 4   # send home


class ResourceAction(IntEnum):
    """Resource allocator options."""
    ASSIGN_ICU_BED    = 1
    ASSIGN_GENERAL_BED = 2
    SCHEDULE_LAB      = 3
    ALLOCATE_STAFF    = 4
    HOLD              = 5   # wait, no resource assigned yet


# ---------------------------------------------------------------------------
# Patient State (internal simulation object)
# ---------------------------------------------------------------------------

@dataclass
class PatientState:
    """Represents one patient in the hospital simulation."""

    patient_id: str

    # Demographics
    age: int                          # 1–90

    # Vitals
    heart_rate: int                   # beats/min,  20–180
    systolic_bp: int                  # mmHg,       50–200
    respiratory_rate: int             # breaths/min, 5–40
    spo2: float                       # %, 70.0–100.0
    temperature: float                # °C, 34.0–41.0
    avpu: int                         # 0=Alert,1=Voice,2=Pain,3=Unresponsive

    # Computed
    mews_score: int = 0               # filled by mews_scorer
    true_severity: int = 0           # ground truth label (1–5)

    # Queue management
    time_in_queue: int = 0            # steps waiting
    assigned_ward: Optional[int] = None
    current_treatment: Optional[int] = None
    treatment_steps: int = 0

    # Outcome tracking
    is_stabilized: bool = False
    is_discharged: bool = False
    is_deteriorated: bool = False
    is_deceased: bool = False

    def to_obs_dict(self) -> dict:
        """Serialise for inclusion in agent observations."""
        return {
            "patient_id":        self.patient_id,
            "age":               self.age,
            "heart_rate":        self.heart_rate,
            "systolic_bp":       self.systolic_bp,
            "respiratory_rate":  self.respiratory_rate,
            "spo2":              self.spo2,
            "temperature":       self.temperature,
            "avpu":              self.avpu,
            "mews_score":        self.mews_score,
            "time_in_queue":     self.time_in_queue,
            "assigned_ward":     self.assigned_ward,
            "current_treatment": self.current_treatment,
        }


# ---------------------------------------------------------------------------
# Actions  (what agents send to the environment)
# ---------------------------------------------------------------------------

class TriageAction(BaseModel):
    """
    Combined action from all three agents in one step.

    Triage Nurse   → assigns severity + routes patient
    Doctor         → selects treatment protocol
    Resource Alloc → assigns hospital resource

    Set patient_id to the ID of the patient being acted upon.
    Leave doctor_* and resource_* as None if only triaging.
    """

    # Which patient this action targets
    patient_id: str

    # --- Triage Nurse fields ---
    assigned_severity: int            # 1–5  (Severity enum)
    assigned_ward: int                # 1–4  (Ward enum)

    # --- Doctor fields (optional if patient not yet seen by doctor) ---
    treatment_protocol: Optional[int] = None   # TreatmentProtocol enum

    # --- Resource Allocator fields ---
    resource_action: int = ResourceAction.HOLD  # ResourceAction enum

    class Config:
        use_enum_values = True


# ---------------------------------------------------------------------------
# Observations  (what agents receive back)
# ---------------------------------------------------------------------------

class TriageObservation(BaseModel):
    """
    Full environment observation returned after every step.
    Contains queue state, resource state, and last action feedback.
    """

    # Queue snapshot (list of patient dicts)
    patient_queue: list[dict]

    # Hospital resources
    icu_beds_available: int           # 0–2
    general_beds_available: int       # 0–10
    lab_queue_length: int             # 0–5
    staff_units_free: int             # 0–5

    # Feedback on the last action
    last_action_patient_id: str
    last_action_valid: bool
    last_action_feedback: str

    # Step-level reward breakdown (for transparency)
    step_reward: float
    reward_breakdown: dict = {}

    # Episode progress
    step_count: int
    patients_stabilized: int
    patients_deteriorated: int
    patients_deceased: int
    episode_done: bool

    class Config:
        use_enum_values = True


# ---------------------------------------------------------------------------
# State  (episode metadata — returned by state() call)
# ---------------------------------------------------------------------------

@dataclass
class TriageState:
    """Episode-level metadata exposed via the state() endpoint."""

    episode_id: str = ""
    step_count: int = 0
    max_steps: int = 100

    # Running counters
    patients_stabilized: int = 0
    patients_deteriorated: int = 0
    patients_deceased: int = 0
    queue_overflow_count: int = 0

    # Resource state
    icu_beds_available: int = 2
    general_beds_available: int = 10
    lab_queue_length: int = 0
    staff_units_free: int = 5

    # Trajectory reward (RFC 004 — delayed reward)
    trajectory_reward: float = 0.0

    # Decision transcript for LLM grader (RFC 005)
    decision_log: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "episode_id":            self.episode_id,
            "step_count":            self.step_count,
            "max_steps":             self.max_steps,
            "patients_stabilized":   self.patients_stabilized,
            "patients_deteriorated": self.patients_deteriorated,
            "patients_deceased":     self.patients_deceased,
            "queue_overflow_count":  self.queue_overflow_count,
            "icu_beds_available":    self.icu_beds_available,
            "general_beds_available": self.general_beds_available,
            "lab_queue_length":      self.lab_queue_length,
            "staff_units_free":      self.staff_units_free,
            "trajectory_reward":     self.trajectory_reward,
        }