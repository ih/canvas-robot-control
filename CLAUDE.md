# Project Instructions

## Python Environment

Always use the `pythonenv-lerobot` virtual environment when running Python commands.
Use the full path to the interpreter: `C:/Projects/pythonenv-lerobot/Scripts/python`

Examples:
- `C:/Projects/pythonenv-lerobot/Scripts/python script.py`
- `C:/Projects/pythonenv-lerobot/Scripts/pip install package`

## Project Overview

PoC robot control using trained canvas world models + local VLM scoring.
Consumes trained models from `../canvas-world-model/` (sibling repo).

### What is canvas-world-model?

A vision-only world model framework that trains generative models (GPT autoregressive ViT and Diffusion ViT) to predict the next video frame of a robot observation. Everything is encoded as images in a "canvas" format: camera frames concatenated horizontally with colored action separators.

- **Best model:** Diffusion Fine-tune (SSIM 0.891, PSNR 27.24, motor direction accuracy 87.3%, inference 459ms)
- **Dataset:** 1,500 canvases of shoulder_pan single-action + hold, from SO-101 arm
- **Canvas format:** 464H x 480W — two 224x224 camera frames (base + wrist stacked vertically) separated by 32px colored action strip, plus 16px motor position strip at bottom
- **Discrete actions:** 1=move+ (green), 2=move- (blue), 3=hold (red), 0=buffer (yellow)
- **Checkpoints:** `../canvas-world-model/local/checkpoints/hold_exp/iter1/diff_finetune/best.pth`

## Architecture

MPC-style control loop with 1-step planning horizon:
1. Capture live cameras + motor state from SO-101
2. Build 3 candidate canvases (one per discrete action: move+, move-, hold)
3. Batched diffusion inference -> 3 predicted next-frames
4. Local VLM scores each prediction based on natural language task prompt
5. Execute highest-scoring action
6. Wait for settle, repeat

Estimated ~2s/step (500ms diffusion + 900ms VLM + 500ms settle).

## Key Files

- `run_control.py` — Main entry point, CLI, MPC loop
- `control/config.py` — ControlConfig dataclass
- `control/world_model.py` — Loads diffusion model from canvas-world-model, batched prediction
- `control/canvas_utils.py` — Live canvas construction matching training distribution exactly
- `control/robot_interface.py` — SO-101 via FeetechMotorsBus + OpenCVCamera with Windows DSHOW patches. Also has DryRunRobotInterface mock.
- `scorers/base.py` — Abstract VLMScorer interface
- `scorers/moondream.py` — moondream2 (~1.9B) free-form VQA scorer
- `scorers/florence.py` — Florence-2 object detection + geometric centering score
- `scorers/qwen_vl.py` — Qwen2-VL (~2B) free-form VQA scorer

## Usage

```bash
# Dry run (no hardware):
C:/Projects/pythonenv-lerobot/Scripts/python run_control.py --dry-run --task "center the red block" --scorer moondream

# Real hardware:
C:/Projects/pythonenv-lerobot/Scripts/python run_control.py --task "center the red block" --scorer moondream

# Switch scorer:
C:/Projects/pythonenv-lerobot/Scripts/python run_control.py --task "center the red block" --scorer florence
```

## Robot Hardware

- **Arm:** SO-101 with STS3215 Feetech servos
- **Port:** COM3 (follower), COM9 (leader for teleop)
- **Joints:** shoulder_pan(1), shoulder_lift(2), elbow_flex(3), wrist_flex(4), wrist_roll(5), gripper(6)
- **Cameras:** base_0_rgb (USB index 1), left_wrist_0_rgb (USB index 0), 640x480, ROTATE_180, DSHOW backend
- **Calibration:** `~/.cache/huggingface/lerobot/calibration/robots/so101_follower/my_so101_follower.json`
- **Joint limits:** -60 to +60 degrees for shoulder_pan
- **Step size:** 10 degrees per discrete action

## Related Repos

- `../canvas-world-model/` — World model training, evaluation, dataset creation
- `../robotic-foundation-model-tests/` — SO-101 test scripts, recording, teleop (reference for hardware patterns)

## Current Task

**See `PLAN.md` for the active plan to implement next.**

Summary: Improve VLM scoring by sending both camera views (base + wrist) to Gemma 4 E4B with clear labels. No randomization. Generate HTML reports for debugging.

## Status

- Canvas world model working with **iterative x0 refinement** inference (fixed from broken DDIM)
- Gemma 4 E4B comparative scorer added (`scorers/gemma4_comparative.py`) — 175ms inference
- `run_control_with_report.py` generates HTML reports with images + VLM responses
- Camera white balance issue: red objects appear blue (hardware WB, not fixable in software easily)
- World model checkpoint: `../canvas-world-model/local/checkpoints/hold_exp/iter1/diff_finetune/best.pth` (sample prediction, 512d12)
- `transformers>=5.5.0` required for Gemma 4
- Camera rotation fix applied: `rotation=180` not `cv2.ROTATE_180`
- WB fix attempted via `videocapture.set(CAP_PROP_WB_TEMPERATURE, 6500)` — doesn't stick reliably

## Key Recent Files

- `run_control_with_report.py` — Control loop with HTML report generation
- `scorers/gemma4_comparative.py` — Gemma 4 E4B comparative scorer (current focus of improvement)
- `PLAN.md` — Active implementation plan
