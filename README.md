# Canvas Robot Control

MPC-style robot control using a trained [canvas world model](https://github.com/irvinh/canvas-world-model) and VLM scoring. The world model predicts future camera frames for candidate actions, a VLM scores each prediction against a natural language task, and the highest-scoring action is executed.

## How It Works

```
 Capture cameras    Build 3 canvases     Diffusion inference     VLM scores        Execute
 + motor state  -->  (right/left/hold) --> 3 predicted frames --> each prediction --> best action
       ^                                                                                  |
       |________________________________________________________________________________|
                                          repeat
```

1. **Observe** - Capture overhead + wrist camera views and motor positions from an SO-101 arm
2. **Predict** - Build candidate canvases for each discrete action (move right, move left, hold) and run batched diffusion inference to predict next frames. Supports multi-step chaining (`--prediction-depth 2` predicts 2 moves ahead)
3. **Score** - A VLM evaluates each predicted outcome against the task goal (e.g. "point the claw at the mouse")
4. **Execute** - The highest-scoring action is sent to the robot
5. **Repeat** - Wait for settle, then loop

## Scorers

| Scorer | Flag | Description |
|--------|------|-------------|
| **Claude (Opus)** | `--scorer claude` | Uses Claude Code CLI on your subscription. Best accuracy (only model that reliably does spatial reasoning on predicted frames). ~25s/step. |
| Gemma 4 E4B | `--scorer gemma4` | Local 4B VLM. Fast (175ms) but suffers from position bias. |
| Qwen2-VL | `--scorer qwen` | Local 2B VLM. Fast but outputs constant scores. |
| Interactive | `run_interactive.py` | Human-in-the-loop scoring via file-based communication. |

### VLM Evaluation Results

We tested 7 local VLMs on the same set of predicted frames. None could reliably distinguish the predictions:

| Model | Size | Result |
|-------|------|--------|
| Gemma 4 E4B | 4B active | Always outputs "RIGHT" (position bias) |
| Qwen3-VL-8B | 8B | Constant output "3" |
| Qwen2.5-VL-7B | 7B | Constant output "6" |
| Pixtral-12B | 12B | Empty output |
| Qwen2.5-VL-32B (4-bit) | 32B | Some reasoning but inconsistent (2/5) |
| Gemma 4 31B (4-bit) | 31B | "not_visible" for everything |
| **Claude Opus** | **frontier** | **5/5 correct, clean convergence** |

See [docs/](docs/) for HTML reports with embedded images from each run.

## Quick Start

```bash
# Set up environment
# Requires: pythonenv-lerobot venv, canvas-world-model sibling repo, SO-101 hardware

# Automated control with Claude scorer (uses your Claude Code subscription):
C:/Projects/pythonenv-lerobot/Scripts/python run_control_with_report.py \
  --task "move the arm so the claw is pointed at the mouse" \
  --scorer claude --prompt-style score --prediction-depth 2 \
  --max-steps 10 --success-threshold 99 --save-frames

# Interactive mode (human scores each step):
C:/Projects/pythonenv-lerobot/Scripts/python run_interactive.py \
  --task "move the arm so the claw is pointed at the mouse" \
  --prediction-depth 2 --max-steps 10

# Dry run (no hardware):
C:/Projects/pythonenv-lerobot/Scripts/python run_control.py \
  --dry-run --task "center the red block" --scorer gemma4
```

## Key Files

| File | Purpose |
|------|---------|
| `run_control.py` | Main control loop with CLI args |
| `run_control_with_report.py` | Control loop + HTML report generation |
| `run_interactive.py` | Human-in-the-loop scoring via file exchange |
| `control/world_model.py` | Diffusion model inference, multi-step prediction chaining |
| `control/canvas_utils.py` | Canvas construction, camera view extraction |
| `control/robot_interface.py` | SO-101 hardware abstraction (motors + cameras) |
| `control/config.py` | Configuration dataclass |
| `scorers/claude_code_scorer.py` | Claude Code CLI scorer (subscription-based) |
| `scorers/gemma4_comparative.py` | Gemma 4 scorer (comparative + score modes) |
| `generate_interactive_report.py` | Generate HTML report from interactive session data |

## Hardware

- **Arm:** SO-101 with STS3215 Feetech servos on COM3
- **Cameras:** 2x USB cameras (overhead index 1, wrist index 0), 320x240, DSHOW backend, 180 deg rotation
- **World model:** Diffusion ViT fine-tune checkpoint from canvas-world-model

### Windows Camera Notes

- DSHOW multi-camera requires `grab()/retrieve()` pattern (not `read()`) to avoid cross-contamination
- Camera resolution set to 320x240 for wrist camera reliability (640x480 intermittently fails)
- Cameras may need USB replug if they stop producing frames

## Reports

Self-contained HTML reports in [docs/](docs/) with embedded images:

- `interactive_claude_scorer_report.html` - Claude (in-session) scoring with reasoning, 5/5 convergence
- `claude_scorer_run1_oscillating.html` - Automated Claude CLI scorer, first run
- `claude_scorer_run2_oscillating.html` - Automated Claude CLI scorer, second run

## Related Repos

- [canvas-world-model](https://github.com/irvinh/canvas-world-model) - World model training, evaluation, dataset creation
- [robotic-foundation-model-tests](https://github.com/irvinh/robotic-foundation-model-tests) - SO-101 test scripts, recording, teleop
