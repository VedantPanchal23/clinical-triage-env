---
title: Clinical Triage Coordinator
emoji: 🏥
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
tags:
    - openenv
    - reinforcement-learning
    - healthcare
    - multi-agent
    - india
    - triage
short_description: Multi-agent RL environment for Indian hospital triage
---

---

# 🏥 Clinical Triage Coordinator

> **The first medical RL environment in the OpenEnv ecosystem.**
> Three AI agents coordinate emergency triage in an Indian district
> hospital, learning to minimize patient deaths under real-world
> resource constraints.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-0.2.3-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python](https://img.shields.io/badge/Python-3.11-green)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-ready-blue)](https://docker.com)
[![HF Space](https://img.shields.io/badge/HF-Space-yellow)](https://huggingface.co/spaces/VedantPanchal23/clinical-triage-env)

**🎮 Interactive Demo:**
https://vedantpanchal23-clinical-triage-env.hf.space/web

**📡 Live API:**
https://vedantpanchal23-clinical-triage-env.hf.space/docs

---

## What This Environment Does

Indian district hospitals face a triage crisis. With only 2 ICU beds per facility and staff shortages, nurses and doctors must make life-or-death decisions in seconds. A wrong severity classification means a critical patient waits while a minor case gets priority - and patients die. This environment simulates that exact pressure.

Clinical Triage Coordinator puts three RL agents inside a simulated Indian district hospital. A triage nurse assigns severity scores. A specialist doctor selects treatment protocols. A resource allocator manages beds, staff, and lab slots. All three agents act on every step, coordinating in real time as new patients arrive and existing cases deteriorate. Ground truth comes from the Modified Early Warning Score (MEWS) - a clinically validated 17-point scoring system used in real Indian hospitals.

This environment is designed to train and evaluate agents that reason under scarcity and time pressure. The reward signal is dense and clinically meaningful - not hand-crafted but derived from real medical outcomes. An agent that learns to triage correctly in this environment has acquired genuine medical reasoning ability. The hard task (mass casualty with 1 ICU bed) requires multi-step planning that challenges frontier models.

---

## Why This Is Different From Other OpenEnv Environments

| Property | Clinical Triage | Typical Game/Toy Env |
|----------|-----------------|----------------------|
| Domain | Real Indian hospital | Games/puzzles |
| Ground truth | MEWS clinical scoring | Arbitrary rules |
| Stakes | Patient lives | Points |
| Resource pressure | 2 ICU beds, 5 staff | Unlimited |
| Agent coordination | 3 agents per step | Single agent |
| Difficulty curve | Easy -> Mass casualty | Fixed |
| LLM grader | Medical justifiability | N/A |

---

## Three Tasks

| Task | Difficulty | Patients | ICU Beds | Success Criteria |
|------|------------|----------|----------|------------------|
| `easy` | Easy | 5 | 2 | score > 0.7 |
| `medium` | Medium | 10 | 2 | score > 0.6, zero deaths |
| `hard` | Hard - Mass Casualty | 20 | 1 | score > 0.5, < 2 deaths |

`easy` tests clean MEWS-to-severity routing under low ambiguity and rewards precise triage fundamentals.

`medium` introduces realistic scarcity where bed/staff allocation mistakes create cascading deterioration risk.

`hard` adds mass-casualty arrivals with only 1 ICU bed, demanding multi-step planning and tight agent coordination.

---

## MEWS Scoring - The Clinical Ground Truth

MEWS (Modified Early Warning Score) maps vital signs and AVPU responsiveness into an acuity score from 0 to 17. We use MEWS as objective clinical ground truth for severity labels, routing decisions, and reward grading.

| MEWS Score | Severity | Correct Action |
|-----------|----------|----------------|
| ≥ 7 | Critical | ICU + Stabilize immediately |
| 5-6 | Emergency | Emergency ward + Medicate |
| 3-4 | Urgent | General ward + Medicate |
| 1-2 | Semi-urgent | General ward + Monitor |
| 0 | Non-urgent | Discharge |

An agent that assigns severity=1 to a MEWS=15 patient and severity=5 to a MEWS=0 patient demonstrates genuine clinical reasoning. Our grader rewards exactly this.

---

## Reward Design

### Step-level reward (immediate signal, normalized 0.01-0.99)

| Event | Reward |
|------|--------|
| Correct severity + ward | +1.5 -> normalized ~0.99 |
| Off by 1 severity level | +0.5 -> normalized ~0.73 |
| Wrong ward (dangerous) | -1.5 -> normalized ~0.01 |
| Patient treated + stabilized | +1.5 (treatment bonus) |
| SLA breach (patient waited too long) | -0.3 penalty |
| ICU bed correctly assigned | +0.5 |

### Trajectory reward (delayed, RFC 004)

Computed at episode end. Primary learning signal.

| Outcome | Weight |
|---------|--------|
| Patient stabilized | +5.0 |
| Patient deteriorated | -3.0 |
| Patient deceased | -8.0 |
| Zero deaths bonus | +2.0 |
| ICU utilization | +1.0 |

The trajectory reward means an agent that stabilizes all patients scores high even if individual step decisions were imperfect. This mirrors real clinical evaluation: outcomes matter most.

---

## Action Space

| Field | Type | Values | Description | Clinical meaning |
|------|------|--------|-------------|------------------|
| patient_id | str | PT-XXX | Target patient | Identifies which patient receives triage and treatment this step |
| assigned_severity | int | 1-5 | 1=Critical, 5=Non-urgent | Encodes estimated acuity and urgency of intervention |
| assigned_ward | int | 1-4 | 1=ICU, 2=Emergency, 3=General, 4=Discharge | Chooses clinical destination and determines access to resources |
| treatment_protocol | int | 1-4 | 1=Stabilize, 2=Medicate, 3=Refer, 4=Discharge | Represents physician-level intervention strategy |
| resource_action | int | 1-5 | 1=ICU Bed, 2=General Bed, 3=Lab, 4=Staff, 5=Hold | Allocates scarce infrastructure needed to execute care safely |

---

## Observation Space

| Field | Type | Clinical significance |
|------|------|-----------------------|
| patient_queue | list[dict] | Live waiting-room state with vitals and MEWS per patient |
| icu_beds_available | int | Critical-care capacity remaining for life-threatening cases |
| general_beds_available | int | Downstream admission capacity for non-ICU care |
| lab_queue_length | int | Diagnostic bottleneck pressure affecting decision latency |
| staff_units_free | int | Available clinical workforce for intervention execution |
| last_action_feedback | str | Immediate safety/quality feedback on prior medical decision |
| step_reward | float | Dense clinical utility signal normalized to 0.01-0.99 |
| reward_breakdown | dict | Transparent decomposition of triage, treatment, resource, and SLA components |
| episode_done | bool | Indicates episode termination and readiness for grading |

---

## RFC Compliance

| RFC | What it means | How we implement it |
|-----|---------------|---------------------|
| RFC 001 | Standard reset/step/state API | ✅ Full implementation |
| RFC 002 | Tool discoverability | ✅ Available actions in observation feedback |
| RFC 004 | Delayed trajectory rewards | ✅ Episode-end outcome scoring |
| RFC 005 | Agentic harness | ✅ Full decision log for LLM grader |

---

## Dual Grader System

### Programmatic grader

Every step is graded against MEWS ground truth. The grader knows the correct severity for each patient (computed from their vitals) and penalizes mis-triage proportional to how dangerous the error is. Sending a MEWS=15 patient to Discharge is maximally penalized. Sending a MEWS=0 patient to ICU wastes a resource but does not kill.

### LLM grader (RFC 005)

Every decision is logged with its context (patient vitals, MEWS score, agent action, clinical feedback). At episode end, an LLM judge reads the transcript and scores medical justifiability 0-10. This catches errors the programmatic grader misses - like technically correct routing with medically wrong reasoning. Set ANTHROPIC_API_KEY to enable.

---

## Quick Start

### Option 1: Use live HF Space (no setup needed)

```bash
curl -X POST https://vedantpanchal23-clinical-triage-env.hf.space/reset
# Returns 8 patients with vitals and MEWS scores
```

### Option 2: Run locally

```bash
git clone https://github.com/VedantPanchal23/clinical-triage-env
cd clinical-triage-env
python -m venv venv && venv\Scripts\activate
pip install -r server/requirements.txt
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
# Open http://localhost:7860/web for visual dashboard
```

### Option 3: Docker

```bash
docker build -t clinical-triage-env .
docker run -p 7860:7860 clinical-triage-env
# Open http://localhost:7860/web
```

---

## Run the Baseline Inference

```bash
pip install openai
set OPENAI_API_KEY=your_groq_key
set API_BASE_URL=https://api.groq.com/openai/v1
set MODEL_NAME=llama-3.1-8b-instant
set ENV_URL=http://localhost:7860
python inference.py
```

Expected output:

```text
[START] task=easy env=clinical-triage-env model=llama-3.1-8b-instant
[STEP] step=1 action=PT-001 reward=0.99 done=false error=null
...
[END] success=true steps=6 score=0.99 rewards=[...]
[START] task=medium ...
[END] success=true steps=15 score=0.99 rewards=[...]
[START] task=hard ...
[END] success=false steps=15 score=0.306 rewards=[...]
[SUMMARY] tasks=['easy','medium','hard'] scores=[0.99,0.99,0.306] mean=0.762
```

---

## Baseline Scores

| Task | Score | Steps | Success | Model |
|------|-------|-------|---------|-------|
| easy | 0.99 | 6 | ✅ | llama-3.1-8b-instant |
| medium | 0.99 | 15 | ✅ | llama-3.1-8b-instant |
| hard | 0.306 | 15 | ❌ | llama-3.1-8b-instant |

Note: Hard task requires frontier model reasoning.
Baseline uses rule-based fallback (no API key).
Reproducible with seed=42.

---

## All API Endpoints

| Method | Endpoint | Description | Example |
|--------|----------|-------------|---------|
| GET | /health | Health check | {"status":"healthy"} |
| POST | /reset | Reset (medium task) | Returns 10 patients |
| POST | /reset/{task} | Reset specific task | easy/medium/hard |
| POST | /step | Execute action | Returns reward + observation |
| GET | /state | Episode metadata | step_count, episode_id |
| POST | /grade | Task score 0.0-1.0 | {"score":0.99} |
| GET | /tasks | List all tasks | {"easy":{...},"medium":{...}} |
| GET | /metadata | Full env spec | action/obs space details |
| GET | /web | Visual dashboard | Interactive HTML UI |
| GET | /docs | API documentation | Swagger UI |

---

## Project Structure

```text
clinical_triage_env/
├── Dockerfile              # Root Dockerfile (HF Spaces)
├── inference.py            # Baseline inference script
├── models.py               # Typed Action/Observation/State
├── client.py               # TriageEnv(EnvClient) — async+sync
├── openenv.yaml            # OpenEnv manifest
├── pyproject.toml          # Package config
├── AGENTS.md               # Guide for building agents
├── CITATION.cff            # Academic citation
├── README.md               # This file
├── examples/
│   └── grpo_training.py    # GRPO training example
├── tests/
│   └── test_end_to_end.py  # 10 end-to-end tests
└── server/
    ├── Dockerfile          # Server Dockerfile
    ├── app.py              # FastAPI + custom endpoints
    ├── web_interface.py    # Visual dashboard HTML
    ├── triage_environment.py  # Core environment logic
    ├── patient_generator.py   # 12 Indian hospital cases
    ├── mews_scorer.py         # Clinical MEWS implementation
    ├── llm_grader.py          # LLM justifiability scorer
    └── requirements.txt
```

---

## Why This Fills a Real Gap

- First healthcare/medical environment in OpenEnv - zero competition.
- India-specific: models district hospital constraints that affect 500M+ Indians who depend on public healthcare.
- Clinically grounded: MEWS is a validated real-world scoring system, not an arbitrary reward function.
- Dual grader: combines programmatic accuracy with LLM medical reasoning assessment - more robust than either alone.
