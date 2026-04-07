"""Clinical Triage Coordinator - OpenEnv Environment Package."""

from models import (
	TriageAction,
	TriageObservation,
	TriageState,
	PatientState,
	Severity,
	Ward,
	TreatmentProtocol,
	ResourceAction,
)
from client import TriageEnv

__all__ = [
	"TriageEnv",
	"TriageAction",
	"TriageObservation",
	"TriageState",
	"PatientState",
	"Severity",
	"Ward",
	"TreatmentProtocol",
	"ResourceAction",
]

__version__ = "1.0.0"
__author__ = "Vedant"
__description__ = "Multi-Agent RL Environment for Indian Public Hospitals"
