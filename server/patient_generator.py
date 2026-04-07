"""
Patient Generator — Realistic Indian District Hospital Cases
Generates randomised but clinically plausible patient cases for simulation.

Designed around common presentations in Indian public hospitals:
  - Sepsis (most common ICU admission)
  - Trauma (road traffic accidents — leading cause of ER visits)
  - Cardiac events
  - Respiratory distress (TB, pneumonia common in India)
  - Obstetric emergencies
  - Diabetic emergencies
  - Mild/routine cases (majority of district hospital load)
"""

from __future__ import annotations
import random
import uuid
from dataclasses import dataclass
from typing import Optional

from server.mews_scorer import compute_mews, mews_to_severity_int


# ---------------------------------------------------------------------------
# Case templates — clinically grounded vital ranges per condition
# ---------------------------------------------------------------------------

CASE_TEMPLATES = [
    # (name, weight, age_range, hr, sbp, rr, spo2, temp, avpu)
    # weight = relative probability of this case appearing

    # --- Critical cases ---
    {
        "name": "Septic Shock",
        "weight": 8,
        "age_range": (30, 75),
        "heart_rate":       (120, 160),
        "systolic_bp":      (50, 85),
        "respiratory_rate": (28, 38),
        "spo2":             (80.0, 90.0),
        "temperature":      (38.5, 40.5),
        "avpu_choices":     [1, 2, 3],
        "avpu_weights":     [0.3, 0.4, 0.3],
    },
    {
        "name": "Acute MI",
        "weight": 6,
        "age_range": (45, 80),
        "heart_rate":       (40, 55),
        "systolic_bp":      (60, 90),
        "respiratory_rate": (20, 30),
        "spo2":             (85.0, 93.0),
        "temperature":      (36.0, 37.5),
        "avpu_choices":     [0, 1, 2],
        "avpu_weights":     [0.4, 0.4, 0.2],
    },
    {
        "name": "Road Traffic Accident",
        "weight": 10,
        "age_range": (18, 55),
        "heart_rate":       (100, 150),
        "systolic_bp":      (60, 100),
        "respiratory_rate": (22, 35),
        "spo2":             (82.0, 94.0),
        "temperature":      (35.0, 37.0),
        "avpu_choices":     [0, 1, 2, 3],
        "avpu_weights":     [0.2, 0.3, 0.3, 0.2],
    },
    {
        "name": "Respiratory Failure",
        "weight": 7,
        "age_range": (40, 80),
        "heart_rate":       (110, 140),
        "systolic_bp":      (90, 130),
        "respiratory_rate": (30, 40),
        "spo2":             (70.0, 85.0),
        "temperature":      (37.0, 39.5),
        "avpu_choices":     [0, 1, 2],
        "avpu_weights":     [0.3, 0.5, 0.2],
    },
    {
        "name": "Obstetric Emergency",
        "weight": 5,
        "age_range": (18, 40),
        "heart_rate":       (110, 150),
        "systolic_bp":      (55, 90),
        "respiratory_rate": (22, 32),
        "spo2":             (88.0, 95.0),
        "temperature":      (37.5, 40.0),
        "avpu_choices":     [0, 1, 2],
        "avpu_weights":     [0.4, 0.4, 0.2],
    },

    # --- Urgent cases ---
    {
        "name": "Diabetic Ketoacidosis",
        "weight": 8,
        "age_range": (20, 65),
        "heart_rate":       (100, 130),
        "systolic_bp":      (85, 115),
        "respiratory_rate": (22, 32),
        "spo2":             (92.0, 97.0),
        "temperature":      (36.5, 38.5),
        "avpu_choices":     [0, 1],
        "avpu_weights":     [0.6, 0.4],
    },
    {
        "name": "Stroke",
        "weight": 6,
        "age_range": (50, 85),
        "heart_rate":       (60, 100),
        "systolic_bp":      (160, 200),
        "respiratory_rate": (14, 22),
        "spo2":             (90.0, 96.0),
        "temperature":      (36.5, 38.0),
        "avpu_choices":     [1, 2],
        "avpu_weights":     [0.6, 0.4],
    },
    {
        "name": "Severe Pneumonia",
        "weight": 9,
        "age_range": (5, 80),
        "heart_rate":       (100, 130),
        "systolic_bp":      (90, 130),
        "respiratory_rate": (25, 35),
        "spo2":             (85.0, 93.0),
        "temperature":      (38.5, 40.5),
        "avpu_choices":     [0, 1],
        "avpu_weights":     [0.7, 0.3],
    },

    # --- Semi-urgent cases ---
    {
        "name": "Moderate Fever",
        "weight": 15,
        "age_range": (5, 70),
        "heart_rate":       (90, 110),
        "systolic_bp":      (100, 140),
        "respiratory_rate": (16, 22),
        "spo2":             (95.0, 99.0),
        "temperature":      (38.5, 40.0),
        "avpu_choices":     [0],
        "avpu_weights":     [1.0],
    },
    {
        "name": "Minor Trauma",
        "weight": 12,
        "age_range": (10, 60),
        "heart_rate":       (85, 105),
        "systolic_bp":      (110, 150),
        "respiratory_rate": (14, 20),
        "spo2":             (96.0, 99.0),
        "temperature":      (36.5, 37.5),
        "avpu_choices":     [0],
        "avpu_weights":     [1.0],
    },

    # --- Non-urgent cases ---
    {
        "name": "Routine Checkup",
        "weight": 10,
        "age_range": (20, 70),
        "heart_rate":       (65, 90),
        "systolic_bp":      (110, 140),
        "respiratory_rate": (12, 18),
        "spo2":             (97.0, 100.0),
        "temperature":      (36.5, 37.2),
        "avpu_choices":     [0],
        "avpu_weights":     [1.0],
    },
    {
        "name": "Mild Gastroenteritis",
        "weight": 10,
        "age_range": (5, 60),
        "heart_rate":       (80, 100),
        "systolic_bp":      (105, 130),
        "respiratory_rate": (14, 18),
        "spo2":             (97.0, 100.0),
        "temperature":      (37.5, 38.5),
        "avpu_choices":     [0],
        "avpu_weights":     [1.0],
    },
]

# Pre-compute weights list for random.choices
_TEMPLATE_WEIGHTS = [t["weight"] for t in CASE_TEMPLATES]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate_patient(patient_id: Optional[str] = None) -> "PatientState":
    """
    Generate a single randomised but clinically plausible patient.
    Imports PatientState here to avoid circular imports.
    """
    from models import PatientState

    if patient_id is None:
        patient_id = f"PT-{uuid.uuid4().hex[:6].upper()}"

    template = random.choices(CASE_TEMPLATES, weights=_TEMPLATE_WEIGHTS, k=1)[0]

    age = random.randint(*template["age_range"])
    hr  = random.randint(*template["heart_rate"])
    sbp = random.randint(*template["systolic_bp"])
    rr  = random.randint(*template["respiratory_rate"])
    spo2 = round(random.uniform(*template["spo2"]), 1)
    temp = round(random.uniform(*template["temperature"]), 1)
    avpu = random.choices(
        template["avpu_choices"],
        weights=template["avpu_weights"],
        k=1
    )[0]

    # Compute ground truth MEWS
    mews = compute_mews(
        heart_rate=hr,
        systolic_bp=sbp,
        respiratory_rate=rr,
        temperature=temp,
        spo2=spo2,
        avpu=avpu,
    )

    patient = PatientState(
        patient_id=patient_id,
        age=age,
        heart_rate=hr,
        systolic_bp=sbp,
        respiratory_rate=rr,
        spo2=spo2,
        temperature=temp,
        avpu=avpu,
        mews_score=mews.total,
        true_severity=mews_to_severity_int(mews.total),
    )

    return patient


def generate_initial_queue(size: int = 8) -> list["PatientState"]:
    """
    Generate the starting patient queue for a new episode.
    Guarantees at least 1 critical and 1 non-urgent patient
    so every episode has meaningful decision pressure.
    """
    patients = []

    # Guarantee distribution
    patients.append(_generate_from_template("Septic Shock"))
    patients.append(_generate_from_template("Road Traffic Accident"))
    patients.append(_generate_from_template("Routine Checkup"))

    # Fill rest randomly
    for i in range(size - 3):
        patients.append(generate_patient())

    # Shuffle so critical patient isn't always first
    random.shuffle(patients)

    # Assign clean IDs after shuffle
    for i, p in enumerate(patients):
        p.patient_id = f"PT-{i+1:03d}"

    return patients


def _generate_from_template(name: str) -> "PatientState":
    """Generate a patient from a specific named template."""
    template = next(t for t in CASE_TEMPLATES if t["name"] == name)
    from models import PatientState

    patient_id = f"PT-{uuid.uuid4().hex[:6].upper()}"
    age  = random.randint(*template["age_range"])
    hr   = random.randint(*template["heart_rate"])
    sbp  = random.randint(*template["systolic_bp"])
    rr   = random.randint(*template["respiratory_rate"])
    spo2 = round(random.uniform(*template["spo2"]), 1)
    temp = round(random.uniform(*template["temperature"]), 1)
    avpu = random.choices(
        template["avpu_choices"],
        weights=template["avpu_weights"],
        k=1
    )[0]

    mews = compute_mews(
        heart_rate=hr,
        systolic_bp=sbp,
        respiratory_rate=rr,
        temperature=temp,
        spo2=spo2,
        avpu=avpu,
    )

    return PatientState(
        patient_id=patient_id,
        age=age,
        heart_rate=hr,
        systolic_bp=sbp,
        respiratory_rate=rr,
        spo2=spo2,
        temperature=temp,
        avpu=avpu,
        mews_score=mews.total,
        true_severity=mews_to_severity_int(mews.total),
    )