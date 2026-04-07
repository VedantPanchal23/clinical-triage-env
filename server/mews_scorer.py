"""
MEWS Scorer — Modified Early Warning Score
Clinically accurate implementation for Indian district hospital triage.

MEWS scoring rubric:
  Each vital gets 0–3 points. Total 0–17.
  Score ≥ 5  → Critical / ICU consideration
  Score 3–4  → Urgent monitoring
  Score 1–2  → Semi-urgent
  Score 0    → Non-urgent

References:
  Subbe CP et al. (2001) QJM: Modified Early Warning Score
  Adapted for Indian district hospital resource constraints.
"""

from __future__ import annotations
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# MEWS sub-score tables
# ---------------------------------------------------------------------------

def _score_heart_rate(hr: int) -> int:
    """Heart rate in beats/min."""
    if hr < 40:
        return 3
    elif hr < 51:
        return 2
    elif hr < 101:
        return 0
    elif hr < 111:
        return 1
    elif hr < 130:
        return 2
    else:
        return 3


def _score_systolic_bp(sbp: int) -> int:
    """Systolic blood pressure in mmHg."""
    if sbp < 70:
        return 3
    elif sbp < 81:
        return 2
    elif sbp < 101:
        return 1
    elif sbp < 200:
        return 0
    else:
        return 2


def _score_respiratory_rate(rr: int) -> int:
    """Respiratory rate in breaths/min."""
    if rr < 9:
        return 2
    elif rr < 15:
        return 0
    elif rr < 21:
        return 1
    elif rr < 30:
        return 2
    else:
        return 3


def _score_temperature(temp: float) -> int:
    """Temperature in Celsius."""
    if temp < 35.0:
        return 2
    elif temp < 38.5:
        return 0
    else:
        return 2


def _score_spo2(spo2: float) -> int:
    """Oxygen saturation %."""
    if spo2 < 85.0:
        return 3
    elif spo2 < 90.0:
        return 2
    elif spo2 < 95.0:
        return 1
    else:
        return 0


def _score_avpu(avpu: int) -> int:
    """
    AVPU consciousness scale.
    0 = Alert, 1 = Voice, 2 = Pain, 3 = Unresponsive
    """
    if avpu == 0:
        return 0
    elif avpu == 1:
        return 1
    elif avpu == 2:
        return 2
    else:
        return 3


# ---------------------------------------------------------------------------
# Breakdown dataclass for transparency
# ---------------------------------------------------------------------------

@dataclass
class MEWSBreakdown:
    """Detailed MEWS breakdown for grader transparency."""
    heart_rate_score:       int
    systolic_bp_score:      int
    respiratory_rate_score: int
    temperature_score:      int
    spo2_score:             int
    avpu_score:             int
    total:                  int
    severity_label:         str
    recommended_ward:       str

    def to_dict(self) -> dict:
        return {
            "heart_rate_score":       self.heart_rate_score,
            "systolic_bp_score":      self.systolic_bp_score,
            "respiratory_rate_score": self.respiratory_rate_score,
            "temperature_score":      self.temperature_score,
            "spo2_score":             self.spo2_score,
            "avpu_score":             self.avpu_score,
            "total":                  self.total,
            "severity_label":         self.severity_label,
            "recommended_ward":       self.recommended_ward,
        }


# ---------------------------------------------------------------------------
# Main scorer function
# ---------------------------------------------------------------------------

def compute_mews(
    heart_rate: int,
    systolic_bp: int,
    respiratory_rate: int,
    temperature: float,
    spo2: float,
    avpu: int,
) -> MEWSBreakdown:
    """
    Compute full MEWS score and return breakdown with routing recommendation.
    """
    hr_s   = _score_heart_rate(heart_rate)
    sbp_s  = _score_systolic_bp(systolic_bp)
    rr_s   = _score_respiratory_rate(respiratory_rate)
    temp_s = _score_temperature(temperature)
    spo2_s = _score_spo2(spo2)
    avpu_s = _score_avpu(avpu)

    total = hr_s + sbp_s + rr_s + temp_s + spo2_s + avpu_s

    severity_label, recommended_ward = _classify(total)

    return MEWSBreakdown(
        heart_rate_score=hr_s,
        systolic_bp_score=sbp_s,
        respiratory_rate_score=rr_s,
        temperature_score=temp_s,
        spo2_score=spo2_s,
        avpu_score=avpu_s,
        total=total,
        severity_label=severity_label,
        recommended_ward=recommended_ward,
    )


def _classify(total: int) -> tuple[str, str]:
    """Map MEWS total to severity label and ward recommendation."""
    if total >= 7:
        return "CRITICAL", "ICU"
    elif total >= 5:
        return "EMERGENCY", "EMERGENCY"
    elif total >= 3:
        return "URGENT", "GENERAL"
    elif total >= 1:
        return "SEMI_URGENT", "GENERAL"
    else:
        return "NON_URGENT", "DISCHARGE"


def mews_to_severity_int(total: int) -> int:
    """Convert MEWS total to Severity enum int (1=critical, 5=non-urgent)."""
    if total >= 7:
        return 1   # CRITICAL
    elif total >= 5:
        return 2   # EMERGENCY
    elif total >= 3:
        return 3   # URGENT
    elif total >= 1:
        return 4   # SEMI_URGENT
    else:
        return 5   # NON_URGENT


# ---------------------------------------------------------------------------
# Grader: score a triage decision against MEWS ground truth
# ---------------------------------------------------------------------------

def score_triage_decision(
    assigned_severity: int,
    assigned_ward: int,
    mews_total: int,
) -> tuple[float, str]:
    """
    Score a single triage decision against MEWS ground truth.

    Returns:
        (reward: float, feedback: str)

    Reward scale:
        +1.0  exact severity match
        +0.5  off by one level (close enough)
         0.0  off by two levels
        -0.5  off by three levels
        -1.0  completely wrong (dangerous under-triage)

    Ward routing adds/subtracts an extra 0.5:
        +0.5  correct ward for MEWS score
        -0.5  wrong ward (e.g. sent critical to DISCHARGE)
    """
    true_severity = mews_to_severity_int(mews_total)
    true_ward_str = _classify(mews_total)[1]

    # Ward int → string map (matches Ward enum)
    ward_map = {1: "ICU", 2: "EMERGENCY", 3: "GENERAL", 4: "DISCHARGE"}
    assigned_ward_str = ward_map.get(assigned_ward, "UNKNOWN")

    # Severity reward
    diff = abs(assigned_severity - true_severity)
    if diff == 0:
        severity_reward = 1.0
        sev_feedback = "Correct severity"
    elif diff == 1:
        severity_reward = 0.5
        sev_feedback = f"Severity off by 1 (assigned {assigned_severity}, true {true_severity})"
    elif diff == 2:
        severity_reward = 0.0
        sev_feedback = f"Severity off by 2 (assigned {assigned_severity}, true {true_severity})"
    elif diff == 3:
        severity_reward = -0.5
        sev_feedback = f"Severity off by 3 — poor triage"
    else:
        severity_reward = -1.0
        sev_feedback = f"Dangerous mis-triage (assigned {assigned_severity}, true {true_severity})"

    # Ward routing reward
    if assigned_ward_str == true_ward_str:
        ward_reward = 0.5
        ward_feedback = f"Correct ward ({assigned_ward_str})"
    elif _is_safe_ward_upgrade(assigned_ward_str, true_ward_str):
        ward_reward = 0.2
        ward_feedback = f"Conservative over-triage to {assigned_ward_str} (acceptable)"
    else:
        ward_reward = -0.5
        ward_feedback = f"Wrong ward (assigned {assigned_ward_str}, should be {true_ward_str})"

    total_reward = severity_reward + ward_reward
    feedback = f"{sev_feedback} | {ward_feedback} | MEWS={mews_total}"

    return round(total_reward, 2), feedback


def _is_safe_ward_upgrade(assigned: str, true: str) -> bool:
    """
    A patient sent to a higher-care ward than needed is safer than
    being sent to a lower-care ward. Reward conservative over-triage slightly.
    """
    hierarchy = {"DISCHARGE": 0, "GENERAL": 1, "EMERGENCY": 2, "ICU": 3}
    return hierarchy.get(assigned, 0) > hierarchy.get(true, 0)


# ---------------------------------------------------------------------------
# Queue SLA checker
# ---------------------------------------------------------------------------

MAX_WAIT_STEPS = {
    1: 1,   # CRITICAL  — must be seen in 1 step
    2: 2,   # EMERGENCY — 2 steps
    3: 4,   # URGENT    — 4 steps
    4: 6,   # SEMI_URGENT
    5: 10,  # NON_URGENT
}

def check_sla_breach(severity: int, time_in_queue: int) -> bool:
    """Return True if patient has waited longer than allowed for their severity."""
    max_wait = MAX_WAIT_STEPS.get(severity, 10)
    return time_in_queue > max_wait