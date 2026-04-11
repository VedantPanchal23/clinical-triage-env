"""
Example: Train an LLM on Clinical Triage using GRPO + TRL.
This shows how to use this environment for actual RL training.
"""

import asyncio
from client import TriageEnv
from models import TriageAction

ENV_URL = "https://vedantpanchal23-clinical-triage-env.hf.space"


async def collect_rollout(task: str = "medium") -> list[dict]:
    """
    Collect one episode of experience for RL training.
    Returns list of (observation, action, reward) tuples.
    """
    rollout = []
    async with TriageEnv(base_url=ENV_URL) as env:
        result = await env.reset()
        step = 0
        while not result.done and step < 30:
            obs = result.observation
            if not obs.patient_queue:
                break
            # Agent picks action here
            patient = max(
                obs.patient_queue,
                key=lambda p: p["mews_score"]
            )
            mews = patient["mews_score"]
            action = TriageAction(
                patient_id=patient["patient_id"],
                assigned_severity=1 if mews >= 7 else 5,
                assigned_ward=1 if mews >= 7 else 4,
                treatment_protocol=1 if mews >= 7 else 4,
                resource_action=1 if mews >= 7 else 5,
            )
            result = await env.step(action)
            rollout.append({
                "obs": obs.model_dump(),
                "action": action.model_dump(),
                "reward": result.reward,
                "done": result.done,
            })
            step += 1
    return rollout


if __name__ == "__main__":
    rollout = asyncio.run(collect_rollout("medium"))
    total_reward = sum(r["reward"] for r in rollout)
    print(f"Episode: {len(rollout)} steps, total reward: {total_reward:.3f}")
    print(f"Mean reward: {total_reward/len(rollout):.3f}")
