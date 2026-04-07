"""Generate HTML report for an interactive control run."""

import base64
import io
import json
from pathlib import Path
from datetime import datetime
from PIL import Image


RUN_DIR = Path("local/runs/interactive_20260406_110438")

# Step data from the interactive session
steps = [
    {"step": 0, "pos": -15.7, "scores": {"move+": 80, "move-": 10, "hold": 35}, "best": "move+",
     "reason": "Mouse is to the RIGHT of the claw. move+ centers it ahead of claw tips. move- scatters debris, mouse gone. hold keeps mouse to the right."},
    {"step": 1, "pos": -5.6, "scores": {"move+": 80, "move-": 10, "hold": 35}, "best": "move+",
     "reason": "Mouse still to the right, closer than before. move+ centers mouse ahead of claw. move- shows large blobs, wrong direction."},
    {"step": 2, "pos": 4.3, "scores": {"move+": 75, "move-": 10, "hold": 45}, "best": "move+",
     "reason": "Mouse slightly to the right, getting close. move+ shifts mouse near center. One more RIGHT should center it."},
    {"step": 3, "pos": 14.3, "scores": {"move+": 85, "move-": 10, "hold": 55}, "best": "move+",
     "reason": "Mouse nearly centered, slightly right. move+ shows excellent alignment directly ahead of claw tips."},
    {"step": 4, "pos": 24.0, "scores": {"move+": 25, "move-": 25, "hold": 85}, "best": "hold",
     "reason": "Mouse centered between claw tips. Claw is pointing at the mouse. Hold prediction confirms stable alignment. Arrived at target."},
    {"step": 5, "pos": 23.9, "scores": {"move+": 20, "move-": 20, "hold": 90}, "best": "hold",
     "reason": "Mouse still centered. Holding is correct. Confirmed stable at target position."},
]


def img_to_b64(path, max_w=200):
    pil = Image.open(path)
    if pil.width > max_w:
        r = max_w / pil.width
        pil = pil.resize((max_w, int(pil.height * r)), Image.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def generate():
    rows = ""
    for s in steps:
        step = s["step"]
        prefix = f"step{step:03d}"

        obs_base = img_to_b64(RUN_DIR / f"{prefix}_obs_base.png")
        obs_wrist = img_to_b64(RUN_DIR / f"{prefix}_obs_wrist.png")

        pred_cells = ""
        for action_name in ["move+", "move-", "hold"]:
            score = s["scores"][action_name]
            b64_base = img_to_b64(RUN_DIR / f"{prefix}_pred_{action_name}_base.png")
            b64_wrist = img_to_b64(RUN_DIR / f"{prefix}_pred_{action_name}_wrist.png")
            is_best = action_name == s["best"]
            border = "3px solid #00ff00" if is_best else "1px solid #666"
            bg = "#1a3a1a" if is_best else "#1a1a1a"
            pred_cells += f"""
                <td style="background:{bg};padding:4px;text-align:center;">
                    <img src="data:image/png;base64,{b64_base}" style="max-width:140px;border:{border};border-radius:4px;"><br>
                    <small>overhead</small><br>
                    <img src="data:image/png;base64,{b64_wrist}" style="max-width:140px;border:{border};border-radius:4px;"><br>
                    <small>wrist</small><br>
                    <b>{action_name}</b>: {score}
                </td>"""

        rows += f"""
        <tr>
            <td style="text-align:center;font-weight:bold;font-size:18px;">{step}</td>
            <td style="text-align:center;">
                <img src="data:image/png;base64,{obs_base}" style="max-width:140px;border:2px solid #333;border-radius:4px;"><br>
                <small>overhead</small><br>
                <img src="data:image/png;base64,{obs_wrist}" style="max-width:140px;border:2px solid #333;border-radius:4px;"><br>
                <small>wrist</small><br>
                <small>pos={s['pos']:.1f}</small>
            </td>
            {pred_cells}
            <td style="font-size:12px;max-width:220px;">
                <b>Choice:</b> {s['best']}<br><br>
                <b>Reasoning:</b> {s['reason']}
            </td>
        </tr>"""

    positions = [s["pos"] for s in steps]
    actions_taken = [s["best"] for s in steps]
    move_plus = sum(1 for a in actions_taken if a == "move+")
    move_minus = sum(1 for a in actions_taken if a == "move-")
    holds = sum(1 for a in actions_taken if a == "hold")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>Interactive Control Report — Claude as VLM</title>
<style>
body {{ font-family: 'Segoe UI', sans-serif; background: #0a0a0a; color: #ddd; margin: 20px; }}
h1 {{ color: #4fc3f7; }}
h2 {{ color: #81c784; border-bottom: 1px solid #333; padding-bottom: 4px; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
th {{ background: #1a237e; color: white; padding: 8px; text-align: center; }}
td {{ padding: 6px; border-bottom: 1px solid #333; vertical-align: top; }}
tr:hover {{ background: #1a1a2e; }}
.config {{ background: #1a1a1a; padding: 12px; border-radius: 8px; font-size: 13px; }}
.overview {{ background: #0d1b2a; padding: 12px; border-radius: 8px; border: 1px solid #1b3a5c; margin: 10px 0; }}
.summary {{ background: #1b5e20; padding: 12px; border-radius: 8px; margin: 10px 0; }}
img {{ border-radius: 4px; }}
</style>
</head><body>
<h1>Interactive Control Report &mdash; Claude as VLM Scorer</h1>
<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

<div class="overview">
<h2>Experiment Overview</h2>
<p>This report demonstrates <b>human-in-the-loop MPC control</b> using a trained canvas diffusion world model
with <b>Claude (Opus 4) as the VLM scorer</b>, replacing the local Gemma 4 E4B model which suffered from
position bias (always answering &ldquo;RIGHT&rdquo; regardless of input).</p>
<p><b>How it works:</b> Each step, the robot captures two camera views (overhead + wrist). The world model
predicts what each camera would see 2 steps ahead for each candidate action (RIGHT, LEFT, HOLD).
Claude examines the wrist camera predictions &mdash; specifically the spatial relationship between the
claw tips and the target object (mouse) &mdash; and scores each prediction 1-100.</p>
<p><b>Key insight:</b> The wrist camera is the most informative view for this task. In the move+ (RIGHT)
predictions, the mouse shifts toward the center of the claw view when that is the correct direction.
In move- (LEFT) predictions, the mouse disappears and debris/other objects appear, clearly indicating
the wrong direction. This consistent visual signal enables reliable scoring.</p>
</div>

<div class="config">
<h2>Configuration</h2>
<b>Task:</b> move the arm so the claw is pointed at the mouse<br>
<b>Scorer:</b> Claude (Opus 4) via interactive session (not API)<br>
<b>Prediction depth:</b> 2 (chained 2-step predictions)<br>
<b>Control joint:</b> shoulder_pan (step: 10.0 deg)<br>
<b>Joint limits:</b> [-60.0, 60.0]
</div>

<div class="summary">
<h2>Summary</h2>
<b>Steps run:</b> {len(steps)}<br>
<b>Start position:</b> {positions[0]:.1f} deg<br>
<b>End position:</b> {positions[-1]:.1f} deg<br>
<b>Total movement:</b> {positions[-1] - positions[0]:+.1f} deg<br>
<b>Actions taken:</b> {move_plus} move+, {move_minus} move-, {holds} hold<br>
<b>Trajectory:</b> {' &rarr; '.join(actions_taken)}<br>
<b>Result:</b> Clean convergence in 4 steps (RIGHT &times;4), then stable HOLD &times;2. No oscillation.
</div>

<h2>Step-by-Step</h2>
<table>
<tr>
    <th>Step</th>
    <th>Observation</th>
    <th>Predict: Move+ (RIGHT)</th>
    <th>Predict: Move- (LEFT)</th>
    <th>Predict: Hold</th>
    <th>Claude's Reasoning</th>
</tr>
{rows}
</table>

<div class="config" style="margin-top:20px;">
<h2>Comparison: Claude vs Gemma 4 E4B</h2>
<table>
<tr><th>Metric</th><th>Gemma 4 E4B (score mode)</th><th>Claude (this run)</th></tr>
<tr><td>Varied responses</td><td style="color:#ff9800;">Yes, but noisy</td><td style="color:#4caf50;">Yes, consistent</td></tr>
<tr><td>Correct direction</td><td style="color:#f44336;">Oscillating (R,L,L,R,L)</td><td style="color:#4caf50;">Consistent (R,R,R,R,H,H)</td></tr>
<tr><td>Convergence</td><td style="color:#f44336;">Never converged</td><td style="color:#4caf50;">4 steps to target</td></tr>
<tr><td>Image grounding</td><td style="color:#f44336;">Text reasoning, not visual</td><td style="color:#4caf50;">Spatial reasoning on wrist camera</td></tr>
<tr><td>Inference time</td><td style="color:#4caf50;">~500ms per step</td><td style="color:#ff9800;">Manual (seconds)</td></tr>
</table>
</div>

</body></html>"""

    out_path = RUN_DIR / "report.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Report written to: {out_path}")


if __name__ == "__main__":
    generate()
