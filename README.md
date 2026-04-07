---
title: Clinical Triage Coordinator
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
---

# Clinical Triage Coordinator

> Multi-Agent RL Environment for Indian Public Hospitals
> Built on [OpenEnv](https://github.com/meta-pytorch/OpenEnv)
> by Meta-PyTorch

## Overview

Clinical Triage Coordinator simulates emergency triage flow in an Indian district hospital where incoming patients compete for scarce care capacity. The environment coordinates three agents: a triage nurse, a specialist doctor, and a resource allocator who jointly decide severity, treatment, and resource assignment at each step. The core challenge is to minimize patient deterioration under real-world constraints including only 2 ICU beds, 5 staff units, and limited lab slots.

## Agents

| Agent | Role | Action Space |
|---|---|---|
| Triage Nurse | Assesses incoming patients | assigned_severity (1-5), assigned_ward (ICU/Emergency/General/Discharge) |
| Specialist Doctor | Selects treatment protocol | treatment_protocol (Stabilize/Medicate/Refer/Discharge) |
| Resource Allocator | Manages hospital resources | resource_action (Assign ICU Bed/General Bed/Schedule Lab/Allocate Staff/Hold) |

## Environment Details

| Parameter | Value |
|---|---|
| Max patients (queue) | 30 simultaneous cases |
| Initial queue size | 8 patients |
| Episode length | 100 steps |
| ICU beds | 2 |
| General beds | 10 |
| Staff units | 5 |
| Lab slots | 5 |
| New patient probability | 30% per step |

## Reward Design

### Step-level rewards (immediate signal)

| Event | Reward |
|---|---|
| Correct severity classification | +1.0 |
| Correct ward routing | +0.5 |
| Safe over-triage (conservative) | +0.2 |
| Dangerous mis-triage | -1.0 |
| Wrong ward routing | -0.5 |
| SLA breach (patient waited too long) | -0.3 |
| Queue overflow | -0.3 |

### Trajectory-level reward (RFC 004 - delayed)

| Outcome | Reward |
|---|---|
| Patient stabilized | +5.0 |
| Patient deteriorated | -3.0 |
| Patient deceased | -8.0 |
| Zero deaths bonus | +2.0 |
| ICU utilization bonus | +1.0 |
| Queue overflow event | -0.5 |

## MEWS Scoring

Modified Early Warning Score (MEWS) converts patient vital signs into a clinically grounded acuity score that estimates risk of deterioration. We use MEWS-derived severity as ground truth for evaluating triage correctness, ward routing safety, and downstream reward shaping.

| MEWS Score | Severity | Recommended Ward |
|---|---|---|
| >= 7 | Critical (1) | ICU |
| 5-6 | Emergency (2) | Emergency Ward |
| 3-4 | Urgent (3) | General Ward |
| 1-2 | Semi-Urgent (4) | General Ward |
| 0 | Non-Urgent (5) | Discharge |

## Dual Grader System

### Programmatic grader

The programmatic grader uses MEWS-aligned scoring to evaluate whether assigned severity and ward decisions match clinical expectations. It also applies SLA breach penalties and captures resource utilization effects so episode quality reflects both medical safety and operational efficiency.

### LLM grader

The LLM grader evaluates medical justifiability on a 0-10 scale using a strict rubric over the full decision transcript. It requires an `ANTHROPIC_API_KEY` environment variable and returns structured justification plus recommendations.

## OpenEnv RFC Compliance

| RFC | Title | Status |
|---|---|---|
| RFC 001 | Baseline API (reset, step, state) | ✅ Implemented |
| RFC 002 | Tool discoverability | ✅ Implemented |
| RFC 004 | Delayed trajectory rewards | ✅ Implemented |
| RFC 005 | Agentic harness integration | ✅ Implemented |

## Quick Start

### Installation

```bash
git clone <your-repo-url>
cd clinical_triage_env
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r server/requirements.txt
```

### Start the server

```bash
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Use the client (async)

```python
import asyncio
from client import TriageEnv
from models import TriageAction, ResourceAction

async def main():
	async with TriageEnv(base_url="http://localhost:8000") as client:
		result = await client.reset()
		obs = result.observation
		patient = obs.patient_queue[0]

		action = TriageAction(
			patient_id=patient["patient_id"],
			assigned_severity=1,
			assigned_ward=1,
			treatment_protocol=1,
			resource_action=ResourceAction.ASSIGN_ICU_BED,
		)
		result = await client.step(action)
		print(f"Reward: {result.reward}")
		print(f"Feedback: {result.observation.last_action_feedback}")

asyncio.run(main())
```

### Use the client (sync)

```python
from client import TriageEnv
from models import TriageAction, ResourceAction

with TriageEnv(base_url="http://localhost:8000").sync() as client:
	result = client.reset()
	obs = result.observation
	patient = obs.patient_queue[0]

	action = TriageAction(
		patient_id=patient["patient_id"],
		assigned_severity=1,
		assigned_ward=1,
		treatment_protocol=1,
		resource_action=ResourceAction.ASSIGN_ICU_BED,
	)
	result = client.step(action)
	print(f"Reward: {result.reward}")
```

### Enable LLM grader

```bash
set ANTHROPIC_API_KEY=your_key_here   # Windows
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Project Structure

```text
clinical_triage_env/
|-- init.py              # Package exports
|-- models.py                # TriageAction, TriageObservation, TriageState
|-- client.py                # TriageEnv(EnvClient) - use this in training
|-- openenv.yaml             # OpenEnv environment manifest
|-- pyproject.toml           # Package configuration
|-- README.md                # This file
`-- server/
|-- init.py
|-- app.py               # FastAPI server (uvicorn entry point)
|-- triage_environment.py # TriageEnvironment(Environment) - core logic
|-- patient_generator.py  # 12 Indian hospital case templates
|-- mews_scorer.py        # Clinically accurate MEWS implementation
|-- llm_grader.py         # LLM justifiability scorer
|-- requirements.txt      # Docker dependencies
`-- Dockerfile            # Container definition
```

## Why This Environment

- Zero existing medical/healthcare environments in the OpenEnv ecosystem
- India-specific framing (district hospital constraints, MEWS scoring) resonates with the hackathon's India focus
- Universally legible objective - judges immediately understand "minimize patient deterioration"
- Deepest RFC alignment of any healthcare environment evaluated (RFC 001, 002, 004, 005)

## License

MIT
