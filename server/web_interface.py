"""Web interface for Clinical Triage Coordinator.
Serves an interactive HTML dashboard at /web endpoint.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"UTF-8\">
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
<title>Clinical Triage Coordinator</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
  * { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg-0: #050816;
    --bg-1: #0b1229;
    --bg-2: #101a36;
    --card: #10192f;
    --line: #24365c;
    --ink: #dbe7ff;
    --ink-muted: #8aa1cf;
    --accent: #22d3ee;
    --ok: #4ade80;
    --warn: #fbbf24;
    --bad: #f87171;
  }
  body {
    font-family: \"Space Grotesk\", \"Segoe UI\", sans-serif;
    background: radial-gradient(1200px 600px at 10% -10%, #12305b 0%, transparent 60%),
                radial-gradient(900px 500px at 100% 0%, #1f4e64 0%, transparent 65%),
                linear-gradient(135deg, var(--bg-0), var(--bg-1) 40%, var(--bg-2));
    color: var(--ink);
    min-height: 100vh;
    padding: 20px;
  }
  .wrap { max-width: 1200px; margin: 0 auto; }
  h1 { color: var(--accent); font-size: 1.65rem; letter-spacing: 0.02em; margin-bottom: 4px; }
  .subtitle { color: var(--ink-muted); font-size: 0.9rem; margin-bottom: 20px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
  .card {
    background: color-mix(in oklab, var(--card) 88%, black 12%);
    border-radius: 14px;
    padding: 16px;
    border: 1px solid var(--line);
    box-shadow: 0 10px 26px rgba(0, 0, 0, 0.3);
  }
  .card h2 {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--ink-muted);
    margin-bottom: 12px;
  }
  .resource-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .resource { background: #091127; border-radius: 9px; padding: 10px; text-align: center; border: 1px solid #17274a; }
  .resource-val { font-size: 1.5rem; font-weight: 700; color: var(--accent); }
  .resource-label { font-size: 0.72rem; color: var(--ink-muted); margin-top: 2px; }
  .patient {
    background: #091127;
    border-radius: 9px;
    padding: 10px;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 10px;
    border: 1px solid #17274a;
  }
  .severity-badge {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.875rem;
    flex-shrink: 0;
  }
  .s1 { background: #dc2626; color: #ffffff; }
  .s2 { background: #ea580c; color: #ffffff; }
  .s3 { background: #d97706; color: #ffffff; }
  .s4 { background: #65a30d; color: #ffffff; }
  .s5 { background: #16a34a; color: #ffffff; }
  .patient-info { flex: 1; }
  .patient-id { font-family: \"JetBrains Mono\", monospace; font-size: 0.8rem; font-weight: 600; color: var(--ink); }
  .patient-vitals { font-size: 0.7rem; color: var(--ink-muted); margin-top: 2px; }
  .mews-bar { height: 4px; background: #1e293b; border-radius: 2px; margin-top: 4px; overflow: hidden; }
  .mews-fill { height: 100%; border-radius: 2px; transition: width 0.25s ease; }
  .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
  .stat { background: #091127; border-radius: 9px; padding: 10px; text-align: center; border: 1px solid #17274a; }
  .stat-val { font-size: 1.25rem; font-weight: 700; }
  .stat-label { font-size: 0.7rem; color: var(--ink-muted); }
  .green { color: var(--ok); }
  .red { color: var(--bad); }
  .yellow { color: var(--warn); }
  .btn {
    background: #0284c7;
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 8px 16px;
    cursor: pointer;
    font-size: 0.875rem;
    margin-right: 8px;
    margin-top: 8px;
    transition: transform 0.15s ease, filter 0.15s ease;
  }
  .btn:hover { filter: brightness(1.06); transform: translateY(-1px); }
  .btn.danger { background: #dc2626; }
  .full-width { grid-column: 1 / -1; }
  #log {
    font-family: \"JetBrains Mono\", monospace;
    font-size: 0.75rem;
    color: var(--ok);
    background: #020617;
    border: 1px solid #17274a;
    border-radius: 8px;
    padding: 10px;
    height: 140px;
    overflow-y: auto;
    margin-top: 8px;
    white-space: pre-wrap;
  }
  @media (max-width: 900px) {
    .grid { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>
<div class=\"wrap\">
  <h1>Clinical Triage Coordinator</h1>
  <p class=\"subtitle\">Multi-Agent RL Environment for Indian District Hospitals</p>

  <div class=\"grid\">
    <div class=\"card\">
      <h2>Hospital Resources</h2>
      <div class=\"resource-grid\">
        <div class=\"resource\">
          <div class=\"resource-val\" id=\"icu\">2</div>
          <div class=\"resource-label\">ICU Beds</div>
        </div>
        <div class=\"resource\">
          <div class=\"resource-val\" id=\"beds\">10</div>
          <div class=\"resource-label\">General Beds</div>
        </div>
        <div class=\"resource\">
          <div class=\"resource-val\" id=\"staff\">5</div>
          <div class=\"resource-label\">Staff Units</div>
        </div>
        <div class=\"resource\">
          <div class=\"resource-val\" id=\"lab\">0</div>
          <div class=\"resource-label\">Lab Queue</div>
        </div>
      </div>
    </div>

    <div class=\"card\">
      <h2>Episode Stats</h2>
      <div class=\"stats\">
        <div class=\"stat\">
          <div class=\"stat-val green\" id=\"stabilized\">0</div>
          <div class=\"stat-label\">Stabilized</div>
        </div>
        <div class=\"stat\">
          <div class=\"stat-val yellow\" id=\"deteriorated\">0</div>
          <div class=\"stat-label\">Deteriorated</div>
        </div>
        <div class=\"stat\">
          <div class=\"stat-val red\" id=\"deceased\">0</div>
          <div class=\"stat-label\">Deceased</div>
        </div>
      </div>
      <div style=\"margin-top:12px; font-size:0.75rem; color:var(--ink-muted)\">
        Step: <span id=\"step\">0</span> /
        Task: <span id=\"task\" style=\"color:var(--accent)\">medium</span>
      </div>
    </div>

    <div class=\"card full-width\">
      <h2>Patient Queue (<span id=\"qlen\">0</span> patients)</h2>
      <div id=\"queue\"></div>
    </div>

    <div class=\"card full-width\">
      <h2>Controls</h2>
      <button class=\"btn\" onclick=\"resetEnv('easy')\">Reset Easy</button>
      <button class=\"btn\" onclick=\"resetEnv('medium')\">Reset Medium</button>
      <button class=\"btn danger\" onclick=\"resetEnv('hard')\">Reset Hard</button>
      <button class=\"btn\" onclick=\"autoStep()\">Auto Step</button>
      <div id=\"log\">Ready. Click Reset to start.</div>
    </div>
  </div>
</div>

<script>
let currentTask = \"medium\";
let obs = null;

function log(msg) {
  const el = document.getElementById(\"log\");
  el.textContent += "\\n" + msg;
  el.scrollTop = el.scrollHeight;
}

function severityColor(mews) {
  if (mews >= 7) return \"s1\";
  if (mews >= 5) return \"s2\";
  if (mews >= 3) return \"s3\";
  if (mews >= 1) return \"s4\";
  return \"s5\";
}

function mewsColor(mews) {
  if (mews >= 7) return \"#dc2626\";
  if (mews >= 5) return \"#ea580c\";
  if (mews >= 3) return \"#d97706\";
  if (mews >= 1) return \"#65a30d\";
  return \"#16a34a\";
}

function updateUI(data) {
  obs = data.observation;
  document.getElementById(\"icu\").textContent = obs.icu_beds_available;
  document.getElementById(\"beds\").textContent = obs.general_beds_available;
  document.getElementById(\"staff\").textContent = obs.staff_units_free;
  document.getElementById(\"lab\").textContent = obs.lab_queue_length;
  document.getElementById(\"stabilized\").textContent = obs.patients_stabilized;
  document.getElementById(\"deteriorated\").textContent = obs.patients_deteriorated;
  document.getElementById(\"deceased\").textContent = obs.patients_deceased;
  document.getElementById(\"step\").textContent = obs.step_count;
  document.getElementById(\"task\").textContent = currentTask;

  const queue = obs.patient_queue || [];
  document.getElementById(\"qlen\").textContent = queue.length;

  const sorted = [...queue].sort((a, b) => b.mews_score - a.mews_score);
  document.getElementById(\"queue\").innerHTML = sorted.slice(0, 8).map((p) => {
    const sc = severityColor(p.mews_score);
    const pct = Math.min(100, (p.mews_score / 17) * 100);
    const col = mewsColor(p.mews_score);
    return `<div class=\"patient\">
      <div class=\"severity-badge ${sc}\">${p.mews_score}</div>
      <div class=\"patient-info\">
        <div class=\"patient-id\">${p.patient_id}</div>
        <div class=\"patient-vitals\">
          HR:${p.heart_rate} BP:${p.systolic_bp} SpO2:${p.spo2}%
          Temp:${p.temperature}&deg;C Wait:${p.time_in_queue}
        </div>
        <div class=\"mews-bar\">
          <div class=\"mews-fill\" style=\"width:${pct}%;background:${col}\"></div>
        </div>
      </div>
    </div>`;
  }).join(\"\");
}

async function resetEnv(task) {
  currentTask = task;
  log(`> Resetting (${task})...`);
  try {
    const r = await fetch(`/reset/${task}`, { method: \"POST\" });
    const data = await r.json();
    updateUI(data);
    log(`> Reset OK: ${data.observation.patient_queue.length} patients`);
  } catch (e) {
    log(`> Error: ${e}`);
  }
}

async function autoStep() {
  if (!obs || !obs.patient_queue.length) {
    log(\"> No episode running. Click Reset first.\");
    return;
  }

  const p = obs.patient_queue.reduce((a, b) => (a.mews_score > b.mews_score ? a : b));
  const mews = p.mews_score;
  const action = {
    patient_id: p.patient_id,
    assigned_severity: mews >= 7 ? 1 : mews >= 5 ? 2 : mews >= 3 ? 3 : mews >= 1 ? 4 : 5,
    assigned_ward: mews >= 7 ? 1 : mews >= 5 ? 2 : mews >= 3 ? 3 : 4,
    treatment_protocol: mews >= 7 ? 1 : mews >= 5 ? 2 : 2,
    resource_action: mews >= 7 ? 1 : mews >= 5 ? 2 : 5,
  };

  try {
    const r = await fetch(\"/step\", {
      method: \"POST\",
      headers: { \"Content-Type\": \"application/json\" },
      body: JSON.stringify(action),
    });
    const data = await r.json();
    updateUI(data);
    log(`> Step ${data.observation.step_count}: ${p.patient_id} MEWS=${mews} reward=${(data.reward || 0).toFixed(3)}`);
    if (data.done) {
      log(\"> Episode done!\");
    }
  } catch (e) {
    log(`> Error: ${e}`);
  }
}

resetEnv(\"medium\");
</script>
</body>
</html>
"""


def get_dashboard_html() -> str:
    return DASHBOARD_HTML
