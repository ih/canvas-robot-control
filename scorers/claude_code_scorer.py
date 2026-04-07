"""VLM scorer using Claude Code CLI (subscription-based, no API key needed).

Spawns `claude -p` as a subprocess to score prediction images.
Uses the Read tool to view images, so it runs on the user's
Claude Code subscription — no API billing.

Usage:
    python run_control_with_report.py --task "point at the mouse" --scorer claude --prompt-style score
"""

import json
import os
import re
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np


def _find_claude_cli():
    """Find the claude CLI executable."""
    # Try common locations
    for candidate in [
        "claude",
        os.path.expandvars(r"%APPDATA%\npm\claude.cmd"),
        os.path.expanduser("~/AppData/Roaming/npm/claude.cmd"),
    ]:
        resolved = os.path.expandvars(candidate)
        if os.path.isfile(resolved):
            return resolved
    # Fall back to just "claude" and hope PATH works
    return "claude"


class ClaudeCodeScorer:
    """Scores predicted frames using Claude Code CLI as VLM."""

    def __init__(self, prompt_style="score"):
        self.prompt_style = prompt_style
        self._current_base = None
        self._current_wrist = None
        self._temp_dir = Path("local/claude_scorer_tmp")
        self._claude_cmd = _find_claude_cli()

    def name(self) -> str:
        return "claude-code"

    def load(self, device="cuda"):
        """No model to load — uses claude CLI."""
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        # Verify claude CLI is available
        try:
            result = subprocess.run(
                [self._claude_cmd, "--version"],
                capture_output=True, text=True, timeout=10,
                shell=True,
            )
            version = result.stdout.strip().split("\n")[0]
            print(f"  Claude Code scorer ready (CLI: {version})")
        except FileNotFoundError:
            raise RuntimeError(
                f"claude CLI not found at '{self._claude_cmd}'. Install Claude Code first."
            )

    def set_current_observation(self, base_rgb, wrist_rgb=None):
        """Save current camera observation for context."""
        self._current_base = base_rgb.copy() if base_rgb is not None else None
        self._current_wrist = wrist_rgb.copy() if wrist_rgb is not None else None

    def score_frames(self, predicted_view_pairs, task_prompt):
        """Score 3 predicted frame pairs using Claude Code CLI.

        Saves all images to temp files, spawns one `claude -p` call
        that reads all images and returns scores as JSON.
        """
        if len(predicted_view_pairs) != 3:
            return [50.0] * len(predicted_view_pairs)

        # Extract task
        task = task_prompt
        if "The task is:" in task:
            task = task.split("The task is:")[-1].split(".")[0].strip()

        # Save prediction wrist images to temp files
        paths = {}
        for i, (base, wrist) in enumerate(predicted_view_pairs):
            action = ["move+", "move-", "hold"][i]
            wrist_path = self._temp_dir / f"pred_{action}_wrist.png"
            cv2.imwrite(str(wrist_path), cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR))
            paths[action] = str(wrist_path.resolve()).replace("\\", "/")

        # Save observation wrist for context
        if self._current_wrist is not None:
            obs_path = self._temp_dir / "obs_wrist.png"
            cv2.imwrite(str(obs_path), cv2.cvtColor(self._current_wrist, cv2.COLOR_RGB2BGR))
            obs_wrist_path = str(obs_path.resolve()).replace("\\", "/")
        else:
            obs_wrist_path = None

        # Build prompt — ask Claude to read all images and score them
        prompt = self._build_prompt(task, paths, obs_wrist_path)

        # Call claude CLI
        t = time.time()
        scores, response = self._call_claude(prompt)
        ms = (time.time() - t) * 1000

        self.last_response = response
        self.last_prompt = prompt
        # Store images for report (load them back as PIL)
        self._store_last_images(obs_wrist_path, paths, predicted_view_pairs)

        print(f"    Claude scored in {ms:.0f}ms: {scores}")
        return scores

    def _build_prompt(self, task, pred_paths, obs_path):
        """Build the scoring prompt for Claude CLI."""
        image_reads = ""
        if obs_path:
            image_reads += f"1. Current wrist camera: {obs_path}\n"
        image_reads += f"2. Predicted wrist after move RIGHT: {pred_paths['move+']}\n"
        image_reads += f"3. Predicted wrist after move LEFT: {pred_paths['move-']}\n"
        image_reads += f"4. Predicted wrist after HOLD: {pred_paths['hold']}\n"

        return (
            f"Read these image files:\n{image_reads}\n"
            f"These show a robot claw from above. A white computer mouse is on the table.\n"
            f"Task: {task}\n\n"
            f"For each predicted image (RIGHT, LEFT, HOLD), rate how well centered "
            f"the mouse is between the claw tips on a scale of 1-100.\n\n"
            f"Reply with ONLY valid JSON in this exact format, nothing else:\n"
            f'{{"move+": <score>, "move-": <score>, "hold": <score>}}'
        )

    def _call_claude(self, prompt):
        """Call claude CLI and parse response."""
        try:
            # Pass prompt via stdin to avoid shell quoting issues
            result = subprocess.run(
                [
                    self._claude_cmd,
                    "-p", "-",
                    "--model", "opus",
                    "--effort", "max",
                    "--allowedTools", "Read",
                    "--output-format", "text",
                ],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=60,
                shell=True,
            )
            response = result.stdout.strip()
        except subprocess.TimeoutExpired:
            print("    Claude CLI timed out")
            return [50.0, 50.0, 50.0], "timeout"
        except Exception as e:
            print(f"    Claude CLI error: {e}")
            return [50.0, 50.0, 50.0], str(e)

        self.last_response = response

        # Parse JSON scores from response
        scores = [50.0, 50.0, 50.0]
        try:
            # Find JSON in response
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                data = json.loads(json_match.group())
                scores = [
                    float(data.get("move+", 50)),
                    float(data.get("move-", 50)),
                    float(data.get("hold", 50)),
                ]
        except (json.JSONDecodeError, ValueError):
            print(f"    Failed to parse scores from: {response[:100]}")

        return scores, response

    def _store_last_images(self, obs_path, pred_paths, predicted_view_pairs):
        """Store images for report compatibility."""
        from PIL import Image

        images = []
        if self._current_base is not None:
            images.append(Image.fromarray(self._current_base))
        else:
            images.append(Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)))
        if self._current_wrist is not None:
            images.append(Image.fromarray(self._current_wrist))
        else:
            images.append(Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)))

        for base, wrist in predicted_view_pairs:
            images.append(Image.fromarray(base))
            images.append(Image.fromarray(wrist))

        self.last_images = images

    def unload(self):
        """Clean up temp files."""
        if self._temp_dir.exists():
            for f in self._temp_dir.glob("*.png"):
                f.unlink()
