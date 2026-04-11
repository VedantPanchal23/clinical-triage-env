# Agent Guide - Clinical Triage Coordinator

## How to build an agent for this environment

### Minimal agent (rule-based)
```python
import asyncio
from client import TriageEnv
from models import TriageAction, ResourceAction


def triage_action(patient: dict) -> TriageAction:
    mews = patient["mews_score"]
    if mews >= 7:
        return TriageAction(
            patient_id=patient["patient_id"],
            assigned_severity=1,
            assigned_ward=1,
            treatment_protocol=1,
            resource_action=ResourceAction.ASSIGN_ICU_BED,
        )
    elif mews >= 5:
        return TriageAction(
            patient_id=patient["patient_id"],
            assigned_severity=2,
            assigned_ward=2,
            treatment_protocol=2,
            resource_action=ResourceAction.ASSIGN_GENERAL_BED,
        )
    elif mews >= 3:
        return TriageAction(
            patient_id=patient["patient_id"],
            assigned_severity=3,
            assigned_ward=3,
            treatment_protocol=2,
            resource_action=ResourceAction.ALLOCATE_STAFF,
        )
    else:
        return TriageAction(
            patient_id=patient["patient_id"],
            assigned_severity=5,
            assigned_ward=4,
            treatment_protocol=4,
            resource_action=ResourceAction.HOLD,
        )


async def run():
    async with TriageEnv(
        base_url="https://vedantpanchal23-clinical-triage-env.hf.space"
    ) as env:
        result = await env.reset()
        while not result.done:
            obs = result.observation
            if not obs.patient_queue:
                break
            patient = max(
                obs.patient_queue,
                key=lambda p: p["mews_score"]
            )
            action = triage_action(patient)
            result = await env.step(action)
            print(f"Reward: {result.reward:.3f}")


asyncio.run(run())
```

## GRPO training with TRL
```python
from trl import GRPOTrainer
# See README for full training example
```

## Key observations for agent design
- Always prioritize highest MEWS score patient first
- ICU beds are scarce (2 total, 1 in hard mode) - use wisely
- STABILIZE only for severity 1-2 patients (MEWS >= 5)
- DISCHARGE for MEWS = 0 gives full reward with no resource use
- SLA breach at: CRITICAL > 1 step, EMERGENCY > 2 steps
