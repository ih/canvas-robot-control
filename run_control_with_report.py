"""Canvas world model control loop with detailed HTML report.

Saves all predicted frames, observations, VLM responses, and scores
into a self-contained HTML report for debugging.

Usage:
    python run_control_with_report.py --task "center the container in the camera view" --scorer gemma4 --max-steps 20
"""

import argparse
import base64
import io
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from control.config import ControlConfig
from control.robot_interface import RobotInterface, DryRunRobotInterface, JOINTS
from control.canvas_utils import stack_cameras_vertically, FRAME_SIZE
from control.world_model import WorldModelPredictor
from run_control import get_scorer, parse_args

CANDIDATE_ACTIONS = [1, 2, 3]  # move+, move-, hold
ACTION_NAMES = {1: "move+", 2: "move-", 3: "hold"}


def img_to_b64(frame_rgb, max_w=300):
    pil = Image.fromarray(frame_rgb)
    if pil.width > max_w:
        r = max_w / pil.width
        pil = pil.resize((max_w, int(pil.height * r)), Image.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def pil_to_b64(pil_img, max_w=300):
    if pil_img.width > max_w:
        r = max_w / pil_img.width
        pil_img = pil_img.resize((max_w, int(pil_img.height * r)), Image.LANCZOS)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def generate_html_report(config, steps_data, output_path):
    """Generate self-contained HTML report with embedded images."""
    rows = ""
    for s in steps_data:
        obs_base_img = f'<img src="data:image/png;base64,{s["obs_base_b64"]}" style="max-width:160px;border:2px solid #333;">'
        obs_wrist_img = f'<img src="data:image/png;base64,{s["obs_wrist_b64"]}" style="max-width:160px;border:2px solid #333;">'

        pred_cells = ""
        for action_name in ["move+", "move-", "hold"]:
            score = s["scores"].get(action_name, 0)
            pred = s["pred_b64"].get(action_name, {})
            b64_base = pred.get("base", "")
            b64_wrist = pred.get("wrist", "")
            is_best = action_name == s["best_action"]
            border = "3px solid #00ff00" if is_best else "1px solid #666"
            bg = "#1a3a1a" if is_best else "#1a1a1a"
            pred_cells += f'''
                <td style="background:{bg};padding:4px;text-align:center;">
                    <img src="data:image/png;base64,{b64_base}" style="max-width:140px;border:{border};border-radius:4px;"><br>
                    <small>overhead</small><br>
                    <img src="data:image/png;base64,{b64_wrist}" style="max-width:140px;border:{border};border-radius:4px;"><br>
                    <small>wrist</small><br>
                    <b>{action_name}</b>: {score:.0f}
                </td>'''

        vlm_resp = s.get("vlm_response", "?")
        rows += f'''
        <tr>
            <td style="text-align:center;font-weight:bold;font-size:18px;">{s["step"]}</td>
            <td style="text-align:center;">
                {obs_base_img}<br><small>overhead</small><br>
                {obs_wrist_img}<br><small>wrist</small><br>
                <small>pos={s["joint_pos"]:.1f}</small>
            </td>
            {pred_cells}
            <td style="font-size:12px;">
                <b>Choice:</b> {s["best_action"]}<br>
                <b>VLM said:</b> "{vlm_resp}"<br>
                <b>Predict:</b> {s["predict_ms"]:.0f}ms<br>
                <b>Score:</b> {s["score_ms"]:.0f}ms
            </td>
        </tr>'''

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>Canvas Control Report</title>
<style>
body {{ font-family: 'Segoe UI', sans-serif; background: #0a0a0a; color: #ddd; margin: 20px; }}
h1 {{ color: #4fc3f7; }}
h2 {{ color: #81c784; border-bottom: 1px solid #333; padding-bottom: 4px; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
th {{ background: #1a237e; color: white; padding: 8px; text-align: center; }}
td {{ padding: 6px; border-bottom: 1px solid #333; vertical-align: top; }}
tr:hover {{ background: #1a1a2e; }}
.config {{ background: #1a1a1a; padding: 12px; border-radius: 8px; font-size: 13px; }}
.summary {{ background: #1b5e20; padding: 12px; border-radius: 8px; margin: 10px 0; }}
img {{ border-radius: 4px; }}
</style>
</head><body>
<h1>Canvas World Model Control Report</h1>
<p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

<div class="config" style="background:#0d1b2a;border:1px solid #1b3a5c;">
<h2>Experiment Overview</h2>
<p>This report shows an MPC-style robot control loop using a trained <b>canvas diffusion world model</b>
to predict future camera frames for candidate actions, scored by <b>{
"Claude (Opus) via Claude Code CLI &mdash; using subscription, not API" if config.scorer == "claude"
else "a local VLM (" + config.scorer + ")"
}</b>.</p>
<p><b>How it works:</b> Each step, the robot captures two camera views (overhead + wrist). The world model
predicts what each camera would see after moving RIGHT, LEFT, or holding still (chained {config.prediction_depth}
step(s) ahead). The VLM scores each predicted outcome against the task goal. The highest-scoring action is executed.</p>
<p><b>Prompt style "{config.prompt_style}":</b>
{"Each prediction is scored independently (1-100) in a separate VLM call, eliminating position bias from multi-choice prompts." if config.prompt_style == "score" else "All 3 predictions shown to the VLM in one call, asked to pick the best action."}</p>
{"<p><b>Why Claude?</b> Local VLMs (Gemma 4 E4B, Qwen3-VL-8B, Qwen2.5-VL 7B/32B, Pixtral-12B) all failed to differentiate the world model predictions &mdash; they output constant scores or suffer from position bias. Claude Opus is the first model to reliably perform spatial reasoning on these blurry 224&times;224 predicted frames.</p>" if config.scorer == "claude" else ""}
</div>

<div class="config" style="font-family:monospace;font-size:12px;">
<h2>Reproduce</h2>
<code>C:/Projects/pythonenv-lerobot/Scripts/python run_control_with_report.py --task "{config.task_prompt}" --scorer {config.scorer} --prompt-style {config.prompt_style} --prediction-depth {config.prediction_depth} --max-steps {config.max_steps} --success-threshold {config.success_threshold} --save-frames</code>
</div>

<div class="config">
<h2>Configuration</h2>
<b>Task:</b> {config.task_prompt}<br>
<b>Scorer:</b> {config.scorer} (prompt: {config.prompt_style}, depth: {config.prediction_depth})<br>
<b>Control joint:</b> {config.control_joint} (step: {config.step_size_degrees} deg)<br>
<b>Joint limits:</b> [{config.joint_min}, {config.joint_max}]<br>
<b>Max steps:</b> {config.max_steps}<br>
<b>Success threshold:</b> {config.success_threshold}
</div>

<div class="summary">
<h2>Summary</h2>
<b>Steps run:</b> {len(steps_data)}<br>
<b>Start position:</b> {steps_data[0]["joint_pos"]:.1f} deg<br>
<b>End position:</b> {steps_data[-1]["joint_pos"]:.1f} deg<br>
<b>Total movement:</b> {steps_data[-1]["joint_pos"] - steps_data[0]["joint_pos"]:+.1f} deg<br>
<b>Actions taken:</b> {sum(1 for s in steps_data if s["best_action"]=="move+")} move+,
    {sum(1 for s in steps_data if s["best_action"]=="move-")} move-,
    {sum(1 for s in steps_data if s["best_action"]=="hold")} hold<br>
<b>Avg step time:</b> {np.mean([s["predict_ms"]+s["score_ms"] for s in steps_data]):.0f}ms
    (predict: {np.mean([s["predict_ms"] for s in steps_data]):.0f}ms,
     score: {np.mean([s["score_ms"] for s in steps_data]):.0f}ms)
</div>

<h2>Step-by-Step</h2>
<table>
<tr>
    <th>Step</th>
    <th>Observation</th>
    <th>Predict: Move+</th>
    <th>Predict: Move-</th>
    <th>Predict: Hold</th>
    <th>Details</th>
</tr>
{rows}
</table>
</body></html>"""

    Path(output_path).write_text(html, encoding="utf-8")
    return output_path


def main():
    config = parse_args()

    # For comparative scorer, pass the raw task — not the old "1 to 100" template
    if config.scorer == "gemma4":
        # Extract raw task from the wrapped prompt
        if "The task is:" in config.task_prompt:
            raw_task = config.task_prompt.split("The task is:")[-1].split(".")[0].strip()
            config.task_prompt = raw_task

    print(f"Task: {config.task_prompt}")
    print(f"Scorer: {config.scorer}")
    print()

    # Load components
    print("Loading world model...")
    predictor = WorldModelPredictor(
        checkpoint_path=config.checkpoint_path,
        canvas_world_model_path=config.canvas_world_model_path,
        num_inference_steps=config.num_inference_steps,
    )
    predictor.load()

    print(f"Loading VLM scorer ({config.scorer}, style={config.prompt_style})...")
    scorer = get_scorer(config.scorer, config.prompt_style)
    scorer.load(predictor.device)
    print(f"  {scorer.name()} ready")

    # Robot
    if config.dry_run:
        robot = DryRunRobotInterface(
            control_joint=config.control_joint,
            step_size=config.step_size_degrees,
            joint_min=config.joint_min,
            joint_max=config.joint_max,
        )
    else:
        robot = RobotInterface(
            port=config.robot_port,
            robot_id=config.robot_id,
            control_joint=config.control_joint,
            step_size=config.step_size_degrees,
            joint_min=config.joint_min,
            joint_max=config.joint_max,
            base_camera_index=config.base_camera_index,
            wrist_camera_index=config.wrist_camera_index,
        )

    control_idx = JOINTS.index(config.control_joint)

    # Output dir
    run_dir = Path("local/runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    robot.connect()
    print("\nStarting control loop\n")

    steps_data = []

    try:
        for step in range(config.max_steps):
            t_start = time.time()

            # 1. Observe
            cameras, motor_state = robot.get_state()
            context_frame = stack_cameras_vertically(
                cameras["base"], cameras["wrist"], FRAME_SIZE
            )
            obs_base_b64 = img_to_b64(cameras["base"])  # fallback
            obs_wrist_b64 = img_to_b64(cameras["wrist"])  # fallback

            # 2. Predict
            t_predict = time.time()
            predicted_pairs = predictor.predict_batch(
                context_frame, motor_state, CANDIDATE_ACTIONS,
                step_size=config.step_size_degrees,
                control_joint_idx=control_idx,
                prediction_depth=config.prediction_depth,
            )
            predict_ms = (time.time() - t_predict) * 1000

            # 3. Score — pass current observation for context
            if hasattr(scorer, 'set_current_observation'):
                scorer.set_current_observation(cameras["base"], cameras.get("wrist"))
            t_score = time.time()
            if config.scorer in ("gemma4", "claude"):
                scores = scorer.score_frames(predicted_pairs, config.task_prompt)
            else:
                base_views = [base for base, _ in predicted_pairs]
                scores = scorer.score_frames(base_views, config.task_prompt)
            score_ms = (time.time() - t_score) * 1000

            vlm_response = getattr(scorer, 'last_response', '?')

            # 4. Select
            best_idx = int(np.argmax(scores))
            best_action = CANDIDATE_ACTIONS[best_idx]

            # Use exact images sent to VLM for report (if available)
            vlm_images = getattr(scorer, 'last_images', None)
            if vlm_images and len(vlm_images) == 8:
                # last_images order: obs_base, obs_wrist, right_base, right_wrist,
                #                    left_base, left_wrist, hold_base, hold_wrist
                obs_base_b64 = pil_to_b64(vlm_images[0])
                obs_wrist_b64 = pil_to_b64(vlm_images[1])
                pred_b64 = {
                    "move+": {"base": pil_to_b64(vlm_images[2], max_w=200),
                              "wrist": pil_to_b64(vlm_images[3], max_w=200)},
                    "move-": {"base": pil_to_b64(vlm_images[4], max_w=200),
                              "wrist": pil_to_b64(vlm_images[5], max_w=200)},
                    "hold":  {"base": pil_to_b64(vlm_images[6], max_w=200),
                              "wrist": pil_to_b64(vlm_images[7], max_w=200)},
                }
            else:
                pred_b64 = {}
                for action, (base, wrist) in zip(CANDIDATE_ACTIONS, predicted_pairs):
                    pred_b64[ACTION_NAMES[action]] = {
                        "base": img_to_b64(base, max_w=200),
                        "wrist": img_to_b64(wrist, max_w=200),
                    }

            joint_pos = float(motor_state[control_idx])
            total_ms = (time.time() - t_start) * 1000

            score_str = " | ".join(
                f"{ACTION_NAMES[a]}={s:.0f}" for a, s in zip(CANDIDATE_ACTIONS, scores)
            )
            print(
                f"Step {step:3d} | {score_str} | "
                f"best={ACTION_NAMES[best_action]} | vlm=\"{vlm_response}\" | "
                f"pos={joint_pos:.1f} | {total_ms:.0f}ms"
            )

            steps_data.append({
                "step": step,
                "scores": {ACTION_NAMES[a]: s for a, s in zip(CANDIDATE_ACTIONS, scores)},
                "best_action": ACTION_NAMES[best_action],
                "joint_pos": joint_pos,
                "vlm_response": vlm_response,
                "predict_ms": predict_ms,
                "score_ms": score_ms,
                "obs_base_b64": obs_base_b64,
                "obs_wrist_b64": obs_wrist_b64,
                "pred_b64": pred_b64,
            })

            # Save raw images too
            cv2.imwrite(str(run_dir / f"step{step:03d}_obs_base.png"),
                        cv2.cvtColor(cameras["base"], cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(run_dir / f"step{step:03d}_obs_wrist.png"),
                        cv2.cvtColor(cameras["wrist"], cv2.COLOR_RGB2BGR))
            for action, (base, wrist) in zip(CANDIDATE_ACTIONS, predicted_pairs):
                cv2.imwrite(str(run_dir / f"step{step:03d}_{ACTION_NAMES[action]}_base.png"),
                            cv2.cvtColor(base, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(run_dir / f"step{step:03d}_{ACTION_NAMES[action]}_wrist.png"),
                            cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR))

            # 5. Execute
            robot.execute_action(best_action)

            if scores[best_idx] >= config.success_threshold:
                print(f"\nSuccess! Score {scores[best_idx]:.1f} >= {config.success_threshold}")
                break

            time.sleep(config.settle_time)

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        robot.disconnect()

    # Generate report
    report_path = run_dir / "report.html"
    generate_html_report(config, steps_data, str(report_path))
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
