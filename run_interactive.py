"""Interactive control loop — human-in-the-loop VLM scoring.

Saves predicted images to local/interactive/ and polls for scores.json.
An external evaluator (e.g., Claude in a chat session) reads the images,
writes scores, and the loop continues.

Usage:
    python run_interactive.py --task "point at the mouse" --max-steps 10
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from control.config import ControlConfig
from control.canvas_utils import stack_cameras_vertically, FRAME_SIZE
from control.world_model import WorldModelPredictor
from control.robot_interface import RobotInterface, DryRunRobotInterface, JOINTS

CANDIDATE_ACTIONS = [1, 2, 3]
ACTION_NAMES = {1: "move+", 2: "move-", 3: "hold"}
INTERACTIVE_DIR = Path("local/interactive")


def save_step_images(cameras, predicted_pairs, step, out_dir):
    """Save observation + prediction images for evaluation."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clear previous step images
    for f in out_dir.glob("*.png"):
        f.unlink()

    # Save observations
    cv2.imwrite(str(out_dir / "obs_base.png"),
                cv2.cvtColor(cameras["base"], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / "obs_wrist.png"),
                cv2.cvtColor(cameras["wrist"], cv2.COLOR_RGB2BGR))

    # Save predictions
    for action, (base, wrist) in zip(CANDIDATE_ACTIONS, predicted_pairs):
        name = ACTION_NAMES[action]
        cv2.imwrite(str(out_dir / f"pred_{name}_base.png"),
                    cv2.cvtColor(base, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / f"pred_{name}_wrist.png"),
                    cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR))


def wait_for_scores(out_dir, timeout=300):
    """Poll for scores.json until it appears or timeout."""
    scores_path = out_dir / "scores.json"
    start = time.time()
    while time.time() - start < timeout:
        if scores_path.exists():
            try:
                with open(scores_path) as f:
                    data = json.load(f)
                scores_path.unlink()  # consume it
                return data
            except (json.JSONDecodeError, KeyError):
                time.sleep(0.5)
        time.sleep(0.5)
    raise TimeoutError("Timed out waiting for scores.json")


def main():
    p = argparse.ArgumentParser(description="Interactive control loop")
    p.add_argument("--task", type=str, required=True)
    p.add_argument("--checkpoint", type=str,
                   default="../canvas-world-model/local/checkpoints/hold_exp/iter1/diff_finetune/best.pth")
    p.add_argument("--cwm-path", type=str, default="../canvas-world-model")
    p.add_argument("--inference-steps", type=int, default=50)
    p.add_argument("--prediction-depth", type=int, default=2)
    p.add_argument("--port", type=str, default="COM3")
    p.add_argument("--robot-id", type=str, default="my_so101_follower")
    p.add_argument("--step-size", type=float, default=10.0)
    p.add_argument("--joint-min", type=float, default=-60.0)
    p.add_argument("--joint-max", type=float, default=60.0)
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--settle-time", type=float, default=0.5)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    print(f"Task: {args.task}")
    print(f"Prediction depth: {args.prediction_depth}")
    print()

    # Load world model
    print("Loading world model...")
    predictor = WorldModelPredictor(
        checkpoint_path=args.checkpoint,
        canvas_world_model_path=args.cwm_path,
        num_inference_steps=args.inference_steps,
    )
    predictor.load()

    # Robot
    if args.dry_run:
        robot = DryRunRobotInterface(
            control_joint="shoulder_pan",
            step_size=args.step_size,
            joint_min=args.joint_min,
            joint_max=args.joint_max,
        )
    else:
        robot = RobotInterface(
            port=args.port,
            robot_id=args.robot_id,
            control_joint="shoulder_pan",
            step_size=args.step_size,
            joint_min=args.joint_min,
            joint_max=args.joint_max,
        )

    control_idx = JOINTS.index("shoulder_pan")

    # Prepare interactive dir
    INTERACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    # Clean up any leftover files
    for f in INTERACTIVE_DIR.glob("*"):
        f.unlink()

    robot.connect()
    print("\n=== Interactive control loop ===")
    print(f"Images will be saved to: {INTERACTIVE_DIR}/")
    print("Write scores to local/interactive/scores.json to continue each step.")
    print('Format: {"move+": 75, "move-": 20, "hold": 40}')
    print()

    log = []

    try:
        for step in range(args.max_steps):
            # 1. Observe
            cameras, motor_state = robot.get_state()
            context_frame = stack_cameras_vertically(
                cameras["base"], cameras["wrist"], FRAME_SIZE
            )
            joint_pos = float(motor_state[control_idx])

            # 2. Predict
            t_predict = time.time()
            predicted_pairs = predictor.predict_batch(
                context_frame, motor_state, CANDIDATE_ACTIONS,
                step_size=args.step_size,
                control_joint_idx=control_idx,
                prediction_depth=args.prediction_depth,
            )
            predict_ms = (time.time() - t_predict) * 1000

            # 3. Save images for evaluation
            save_step_images(cameras, predicted_pairs, step, INTERACTIVE_DIR)

            # Write ready signal
            ready_info = {
                "step": step,
                "joint_pos": joint_pos,
                "predict_ms": predict_ms,
                "task": args.task,
                "status": "waiting_for_scores",
            }
            with open(INTERACTIVE_DIR / "ready.json", "w") as f:
                json.dump(ready_info, f, indent=2)

            print(f"Step {step:3d} | pos={joint_pos:.1f} | predict={predict_ms:.0f}ms | WAITING for scores...")

            # 4. Wait for scores
            scores_data = wait_for_scores(INTERACTIVE_DIR)
            scores = [
                float(scores_data.get("move+", 50)),
                float(scores_data.get("move-", 50)),
                float(scores_data.get("hold", 50)),
            ]

            # Remove ready signal
            ready_path = INTERACTIVE_DIR / "ready.json"
            if ready_path.exists():
                ready_path.unlink()

            # 5. Select best action
            best_idx = int(np.argmax(scores))
            best_action = CANDIDATE_ACTIONS[best_idx]

            score_str = " | ".join(
                f"{ACTION_NAMES[a]}={s:.0f}" for a, s in zip(CANDIDATE_ACTIONS, scores)
            )
            print(f"         | {score_str} | best={ACTION_NAMES[best_action]}")

            log.append({
                "step": step,
                "joint_pos": joint_pos,
                "scores": scores_data,
                "best_action": ACTION_NAMES[best_action],
            })

            # 6. Execute
            robot.execute_action(best_action)
            time.sleep(args.settle_time)

    except KeyboardInterrupt:
        print("\nInterrupted")
    except TimeoutError as e:
        print(f"\n{e}")
    finally:
        robot.disconnect()
        # Clean up
        for f in INTERACTIVE_DIR.glob("*"):
            f.unlink()

    # Summary
    if log:
        print(f"\n=== Done: {len(log)} steps ===")
        for entry in log:
            print(f"  Step {entry['step']}: pos={entry['joint_pos']:.1f} -> {entry['best_action']} (scores: {entry['scores']})")


if __name__ == "__main__":
    main()
