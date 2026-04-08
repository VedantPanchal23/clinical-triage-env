# Clinical Triage Coordinator
> Multi-Agent RL Environment for Indian Public Hospitals

🚀 **Live Demo:** https://vedantpanchal23-clinical-triage-env.hf.space/docs
📦 **HuggingFace Space:** https://huggingface.co/spaces/VedantPanchal23/clinical-triage-env
💻 **GitHub:** https://github.com/VedantPanchal23/clinical-triage-env

[![OpenEnv](https://img.shields.io/badge/OpenEnv-0.2.3-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python](https://img.shields.io/badge/Python-3.11-green)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-ready-blue)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Overview

This environment simulates triage workflow inside an Indian district hospital where patient inflow exceeds capacity during peak demand. Three coordinated agents act each step: a triage nurse, a specialist doctor, and a resource allocator. Clinical grounding comes from the Modified Early Warning Score (MEWS), which defines severity truth labels and recommended interventions. It fills a real gap in the OpenEnv ecosystem by introducing a healthcare-native, safety-critical benchmark with operational constraints.

## Why This Environment

- First medical/healthcare environment in OpenEnv ecosystem
- India-specific: district hospital constraints (2 ICU beds, limited staff)
- Clinically grounded: Modified Early Warning Score (MEWS) as ground truth
- RFC 001, 002, 004, 005 fully implemented

## Tasks

| Task | Difficulty | Patients | Max Steps | Success Threshold | Description |
|------|-----------|---------|-----------|------------------|-------------|
| easy | Easy | 5 | 10 | score >= 0.7 | Basic MEWS-guided triage |
| medium | Medium | 10 | 30 | score >= 0.6 + zero deaths | Resource-constrained triage |
| hard | Hard | 20 | 60 | score >= 0.5 + < 2 deaths | Mass casualty incident |

## Scoring / Grader

### Programmatic grader (per step)

| Event | Raw Reward | Normalized (0-1) |
|------|------------|------------------|
| Correct severity + ward | +1.5 | ~0.69 |
| Correct severity only | +1.0 | ~0.59 |
| Wrong severity (dangerous) | -1.5 | ~0.22 |
| SLA breach | -0.3 | ~0.39 |
| Patient stabilized | +5.0 | trajectory |

### Task grader (episode end, 0.0–1.0)

Formula: `score = stabilization_rate - (deaths × 0.15) - (overflows × 0.05)`

Clamped to `[0.0, 1.0]`

### LLM grader (optional)

Scores medical justifiability 0-10 via Anthropic API.
Set `ANTHROPIC_API_KEY` to enable.

## Action Space

| Field | Type | Values | Description |
|------|------|--------|-------------|
| patient_id | str | PT-XXX | Target patient |
| assigned_severity | int | 1-5 | 1=Critical, 5=Non-urgent |
| assigned_ward | int | 1-4 | 1=ICU, 2=Emergency, 3=General, 4=Discharge |
| treatment_protocol | int | 1-4 | 1=Stabilize, 2=Medicate, 3=Refer, 4=Discharge |
| resource_action | int | 1-5 | 1=ICU Bed, 2=General Bed, 3=Lab, 4=Staff, 5=Hold |

## Observation Space

| Field | Type | Description |
|------|------|-------------|
| patient_queue | list[dict] | Up to 30 patients with vitals + MEWS |
| icu_beds_available | int | 0-2 |
| general_beds_available | int | 0-10 |
| lab_queue_length | int | 0-5 |
| staff_units_free | int | 0-5 |
| last_action_feedback | str | Clinical feedback on last decision |
| step_reward | float | Normalized reward 0.0-1.0 |
| episode_done | bool | True when episode ends |

## Patient Vitals (per patient in queue)

| Field | Range | Description |
|------|-------|-------------|
| heart_rate | 20-180 | Beats per minute |
| systolic_bp | 50-200 | mmHg |
| respiratory_rate | 5-40 | Breaths per minute |
| spo2 | 70-100 | Oxygen saturation % |
| temperature | 34-41 | Celsius |
| avpu | 0-3 | 0=Alert 1=Voice 2=Pain 3=Unresponsive |
| mews_score | 0-17 | Computed ground truth severity |
| true_severity | 1-5 | MEWS-derived ground truth label |

## MEWS Reference

| MEWS Score | Severity Level | Recommended Action |
|-----------|----------------|--------------------|
| >= 7 | Critical (1) | ICU + Stabilize immediately |
| 5-6 | Emergency (2) | Emergency ward + Medicate |
| 3-4 | Urgent (3) | General ward + Medicate |
| 1-2 | Semi-urgent (4) | General ward + Monitor |
| 0 | Non-urgent (5) | Discharge |

## OpenEnv RFC Compliance

| RFC | Description | Status |
|-----|-------------|--------|
| RFC 001 | Baseline API reset/step/state | ✅ |
| RFC 002 | Tool discoverability | ✅ |
| RFC 004 | Delayed trajectory rewards | ✅ |
| RFC 005 | Agentic harness + decision log | ✅ |

## Quick Start

### Option 1: Run locally

```bash
git clone https://github.com/VedantPanchal23/clinical-triage-env
cd clinical-triage-env
python -m venv venv
venv\Scripts\activate
pip install -r server/requirements.txt
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Option 2: Docker

```bash
docker build -t clinical-triage-env .
docker run -p 8000:8000 clinical-triage-env
```

### Option 3: Use live HF Space

Base URL: https://vedantpanchal23-clinical-triage-env.hf.space

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Health check |
| POST | /reset | Reset episode (default: medium task) |
| POST | /reset/{task} | Reset with specific task (easy/medium/hard) |
| POST | /step | Execute triage action |
| GET | /state | Get episode state |
| POST | /grade | Get task score (0.0-1.0) |
| GET | /tasks | List all tasks with descriptions |
| GET | /docs | Interactive API documentation |

## Run Inference

```bash
pip install openai
set ENV_URL=http://localhost:8000
set MODEL_NAME=gpt-4o-mini
set API_BASE_URL=https://api.openai.com/v1
set HF_TOKEN=your_openai_key
python inference.py
```

Expected output format:

```text
{"type": "START", "task": "easy", "env": "clinical-triage-env", "model": "gpt-4o-mini"}
{"type": "STEP", "step": 1, "action": "PT-001", "reward": 0.857, "done": false, "error": null}
{"type": "END", "success": true, "steps": 5, "score": 0.8, "rewards": [...]}
{"type": "SUMMARY", "tasks": ["easy","medium","hard"], "scores": [...], "mean_score": 0.7}
```

## Baseline Scores

| Task | Difficulty | Baseline Score | Notes |
|------|------------|----------------|-------|
| easy | Easy | ~0.75 | LLM correctly identifies MEWS critical patients |
| medium | Medium | ~0.60 | Resource constraints reduce perfect score |
| hard | Hard | ~0.45 | Mass casualty overwhelms resources |

## Environment Design

### Multi-agent coordination

Each step() call coordinates all 3 agents simultaneously.
Agents share a single observation space.
Turn-based execution avoids concurrency issues.

### Patient deterioration model

Untreated critical patients (MEWS >= 7) worsen every 3 steps.
MEWS score increases by 2 per deterioration event.
Deceased patients are removed from queue after 2x deterioration threshold.

### Resource management

ICU beds: 2 total (most constrained resource)
General beds: 10 total
Staff units: 5 total
Lab slots: 5 total, 1 freed every 3 steps

## Project Structure

```text
clinical_triage_env/
├── Dockerfile              # Root Dockerfile (for validator)
├── inference.py            # Baseline inference script
├── __init__.py             # Package exports
├── models.py               # Typed Action/Observation/State models
├── client.py               # TriageEnv(EnvClient)
├── openenv.yaml            # OpenEnv manifest
├── pyproject.toml          # Package config
├── README.md               # This file
├── tests/
│   └── test_end_to_end.py  # 10 end-to-end tests
└── server/
    ├── Dockerfile          # Server Dockerfile
    ├── app.py              # FastAPI server
    ├── triage_environment.py
    ├── patient_generator.py
    ├── mews_scorer.py
    ├── llm_grader.py
    └── requirements.txt
```

## License

MIT — Built for Meta PyTorch OpenEnv Hackathon x Scaler SST 2026
