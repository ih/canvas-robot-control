"""MPC control loop: world model prediction + VLM scoring.

Usage:
    # Dry run (no hardware):
    python run_control.py --dry-run --task "center the red block"

    # Real hardware:
    python run_control.py --task "center the red block" --scorer moondream

    # With Florence-2 detection scorer:
    python run_control.py --task "center the red block" --scorer florence
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from control.config import ControlConfig
from control.canvas_utils import stack_cameras_vertically, FRAME_SIZE
from control.world_model import WorldModelPredictor
from control.robot_interface import RobotInterface, DryRunRobotInterface, JOINTS


def get_scorer(name: str, prompt_style: str = "comparative"):
    """Get a VLM scorer by name."""
    if name == "moondream":
        from scorers.moondream import MoondreamScorer
        return MoondreamScorer()
    elif name == "florence":
        from scorers.florence import FlorenceScorer
        return FlorenceScorer()
    elif name == "qwen":
        from scorers.qwen_vl import QwenVLScorer
        return QwenVLScorer()
    elif name == "paligemma":
        from scorers.paligemma import PaliGemmaScorer
        return PaliGemmaScorer()
    elif name == "smolvlm":
        from scorers.smolvlm import SmolVLMScorer
        return SmolVLMScorer()
    elif name == "gemma4":
        from scorers.gemma4_comparative import Gemma4ComparativeScorer
        return Gemma4ComparativeScorer(prompt_style=prompt_style)
    elif name == "claude":
        from scorers.claude_code_scorer import ClaudeCodeScorer
        return ClaudeCodeScorer(prompt_style=prompt_style)
    else:
        raise ValueError(f"Unknown scorer: {name}. Choose from: qwen, paligemma, smolvlm, moondream, florence, gemma4, claude")


def parse_args() -> ControlConfig:
    p = argparse.ArgumentParser(description="World model MPC control loop")

    # Task
    p.add_argument("--task", type=str, required=True, help="Natural language task description")
    p.add_argument("--scorer", type=str, default="qwen", help="VLM scorer: qwen, paligemma, smolvlm, moondream, florence")

    # World model
    p.add_argument("--checkpoint", type=str,
                   default="../canvas-world-model/local/checkpoints/hold_exp/iter1/diff_finetune/best.pth")
    p.add_argument("--cwm-path", type=str, default="../canvas-world-model")
    p.add_argument("--inference-steps", type=int, default=50)

    # Robot
    p.add_argument("--port", type=str, default="COM3")
    p.add_argument("--robot-id", type=str, default="my_so101_follower")
    p.add_argument("--step-size", type=float, default=10.0)
    p.add_argument("--joint-min", type=float, default=-60.0)
    p.add_argument("--joint-max", type=float, default=60.0)

    # Cameras
    p.add_argument("--base-camera", type=int, default=1)
    p.add_argument("--wrist-camera", type=int, default=0)

    # Scoring
    p.add_argument("--prompt-style", type=str, default="comparative",
                   choices=["comparative", "score"],
                   help="VLM prompt style: comparative (pick best) or score (rate each)")
    p.add_argument("--prediction-depth", type=int, default=1,
                   help="Number of chained prediction steps (2 = predict 2 moves ahead)")

    # Control loop
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--settle-time", type=float, default=0.5)
    p.add_argument("--success-threshold", type=float, default=80.0)
    p.add_argument("--dry-run", action="store_true", help="Run without hardware")
    p.add_argument("--save-frames", action="store_true", help="Save frames each step")
    p.add_argument("--output-dir", type=str, default="local/runs")

    args = p.parse_args()

    config = ControlConfig(
        checkpoint_path=args.checkpoint,
        canvas_world_model_path=args.cwm_path,
        num_inference_steps=args.inference_steps,
        robot_port=args.port,
        robot_id=args.robot_id,
        step_size_degrees=args.step_size,
        joint_min=args.joint_min,
        joint_max=args.joint_max,
        control_joint="shoulder_pan",
        base_camera_index=args.base_camera,
        wrist_camera_index=args.wrist_camera,
        scorer=args.scorer,
        prompt_style=args.prompt_style,
        prediction_depth=args.prediction_depth,
        task_prompt=(
            f"This image shows a robot's view. The task is: {args.task}. "
            "On a scale of 1 to 100, how well does this image show the task being achieved? "
            "Reply with only a number."
        ),
        max_steps=args.max_steps,
        settle_time=args.settle_time,
        success_threshold=args.success_threshold,
        dry_run=args.dry_run,
        save_frames=args.save_frames,
        output_dir=args.output_dir,
    )
    return config


ACTION_NAMES = {1: "move+", 2: "move-", 3: "hold"}
CANDIDATE_ACTIONS = [1, 2, 3]


def control_loop(config: ControlConfig) -> None:
    """Run the MPC control loop."""

    # --- Initialize components ---
    # For comparative scorer, pass the raw task — not the old "1 to 100" template
    if config.scorer in ("gemma4", "claude"):
        if "The task is:" in config.task_prompt:
            raw_task = config.task_prompt.split("The task is:")[-1].split(".")[0].strip()
            config.task_prompt = raw_task

    print(f"Task: {config.task_prompt}")
    print(f"Scorer: {config.scorer}")
    print(f"Dry run: {config.dry_run}")
    print()

    # World model
    print("Loading world model...")
    predictor = WorldModelPredictor(
        checkpoint_path=config.checkpoint_path,
        canvas_world_model_path=config.canvas_world_model_path,
        num_inference_steps=config.num_inference_steps,
    )
    predictor.load()
    print(f"  Model loaded on {predictor.device}")

    # VLM scorer
    print(f"Loading VLM scorer ({config.scorer})...")
    scorer = get_scorer(config.scorer, config.prompt_style)
    scorer.load(predictor.device)
    print(f"  {scorer.name()} ready (prompt_style={config.prompt_style})")

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

    # Output directory
    if config.save_frames:
        run_dir = Path(config.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = None

    # Control joint index in the motor array
    control_idx = JOINTS.index(config.control_joint)

    # --- Run loop ---
    print("\nConnecting to robot...")
    robot.connect()
    print("Starting control loop\n")

    log = []

    try:
        for step in range(config.max_steps):
            t_start = time.time()

            # 1. Observe
            cameras, motor_state = robot.get_state()
            context_frame = stack_cameras_vertically(
                cameras["base"], cameras["wrist"], FRAME_SIZE
            )

            # 2. Predict outcomes for each action
            t_predict = time.time()
            predicted_pairs = predictor.predict_batch(
                context_frame,
                motor_state,
                CANDIDATE_ACTIONS,
                step_size=config.step_size_degrees,
                control_joint_idx=control_idx,
                prediction_depth=config.prediction_depth,
            )
            predict_ms = (time.time() - t_predict) * 1000

            # 3. Score predictions
            if hasattr(scorer, 'set_current_observation'):
                scorer.set_current_observation(cameras["base"], cameras.get("wrist"))
            t_score = time.time()
            if config.scorer in ("gemma4", "claude"):
                scores = scorer.score_frames(predicted_pairs, config.task_prompt)
            else:
                base_views = [base for base, _ in predicted_pairs]
                scores = scorer.score_frames(base_views, config.task_prompt)
            score_ms = (time.time() - t_score) * 1000

            # 4. Select best action
            best_idx = int(np.argmax(scores))
            best_action = CANDIDATE_ACTIONS[best_idx]
            best_score = scores[best_idx]

            # Log
            score_str = " | ".join(
                f"{ACTION_NAMES[a]}={s:.1f}" for a, s in zip(CANDIDATE_ACTIONS, scores)
            )
            joint_pos = motor_state[control_idx]
            total_ms = (time.time() - t_start) * 1000
            print(
                f"Step {step:3d} | {score_str} | "
                f"best={ACTION_NAMES[best_action]} ({best_score:.1f}) | "
                f"pos={joint_pos:.1f} | "
                f"predict={predict_ms:.0f}ms score={score_ms:.0f}ms total={total_ms:.0f}ms"
            )

            log.append({
                "step": step,
                "scores": {ACTION_NAMES[a]: s for a, s in zip(CANDIDATE_ACTIONS, scores)},
                "best_action": ACTION_NAMES[best_action],
                "best_score": best_score,
                "joint_position": float(joint_pos),
                "predict_ms": predict_ms,
                "score_ms": score_ms,
            })

            # Save frames if requested
            if run_dir is not None:
                import cv2
                for action, (base, wrist) in zip(CANDIDATE_ACTIONS, predicted_pairs):
                    cv2.imwrite(
                        str(run_dir / f"step{step:03d}_{ACTION_NAMES[action]}_base.png"),
                        cv2.cvtColor(base, cv2.COLOR_RGB2BGR),
                    )
                    cv2.imwrite(
                        str(run_dir / f"step{step:03d}_{ACTION_NAMES[action]}_wrist.png"),
                        cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR),
                    )
                # Also save the actual observations
                cv2.imwrite(
                    str(run_dir / f"step{step:03d}_obs_base.png"),
                    cv2.cvtColor(cameras["base"], cv2.COLOR_RGB2BGR),
                )
                cv2.imwrite(
                    str(run_dir / f"step{step:03d}_obs_wrist.png"),
                    cv2.cvtColor(cameras["wrist"], cv2.COLOR_RGB2BGR),
                )

            # 5. Execute
            robot.execute_action(best_action)

            # 6. Check success
            if best_score >= config.success_threshold:
                print(f"\nSuccess! Score {best_score:.1f} >= threshold {config.success_threshold}")
                break

            # 7. Settle
            time.sleep(config.settle_time)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        print("Disconnecting robot...")
        robot.disconnect()

    # Save log
    if run_dir is not None:
        import json
        with open(run_dir / "log.json", "w") as f:
            json.dump({"config": vars(config), "log": log}, f, indent=2)
        print(f"\nRun saved to: {run_dir}")

    # Summary
    if log:
        scores_over_time = [entry["best_score"] for entry in log]
        print(f"\nSteps: {len(log)}")
        print(f"Score: {scores_over_time[0]:.1f} -> {scores_over_time[-1]:.1f}")
        print(f"Avg predict: {np.mean([e['predict_ms'] for e in log]):.0f}ms")
        print(f"Avg score: {np.mean([e['score_ms'] for e in log]):.0f}ms")


if __name__ == "__main__":
    config = parse_args()
    control_loop(config)
