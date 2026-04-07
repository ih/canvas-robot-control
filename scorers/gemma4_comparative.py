"""Comparative scorer using Gemma 4 E4B.

Instead of scoring each frame independently (1-100), shows all 3 candidate
frames to the VLM and asks which one best achieves the task. Relative
comparisons are much more reliable than absolute numeric scores.
"""

import sys
import io
import re
import torch
import numpy as np
from PIL import Image


ACTION_LABELS = {0: "move RIGHT", 1: "move LEFT", 2: "HOLD STILL"}


class Gemma4ComparativeScorer:
    """Scores predicted frames by comparing them, not rating them individually."""

    def __init__(self, model_id="google/gemma-4-E4B-it", prompt_style="comparative"):
        self.model_id = model_id
        self.model = None
        self.processor = None
        self.prompt_style = prompt_style  # "comparative" or "score"

    def name(self) -> str:
        return "gemma4-comparative"

    def load(self, device="cuda"):
        """Load Gemma 4 model."""
        # Fix Windows encoding
        if sys.platform == "win32":
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

        from transformers import AutoModelForMultimodalLM, AutoProcessor

        print(f"Loading Gemma 4 comparative scorer ({self.model_id})...")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            self.model_id, dtype=torch.float16, device_map=device
        )
        self.model.eval()
        print(f"  Gemma 4 ready ({sum(p.numel() for p in self.model.parameters())/1e9:.1f}B params)")

    def set_current_observation(self, base_rgb, wrist_rgb=None):
        """Set the current real camera observation for context."""
        self._current_base = base_rgb.copy() if base_rgb is not None else None
        self._current_wrist = wrist_rgb.copy() if wrist_rgb is not None else None

    def score_frames(self, predicted_view_pairs, task_prompt):
        """Score 3 predicted frame pairs.

        Delegates to comparative or score-based method depending on prompt_style.
        """
        if self.prompt_style == "score":
            return self._score_individual(predicted_view_pairs, task_prompt)
        return self._score_comparative(predicted_view_pairs, task_prompt)

    def _get_current_images(self):
        """Get current observation as PIL images."""
        if hasattr(self, '_current_base') and self._current_base is not None:
            base = Image.fromarray(self._current_base)
        else:
            base = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

        if hasattr(self, '_current_wrist') and self._current_wrist is not None:
            wrist = Image.fromarray(self._current_wrist)
        else:
            wrist = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

        return base, wrist

    def _extract_task(self, task_prompt):
        """Extract raw task from wrapped prompt."""
        task = task_prompt
        if "The task is:" in task:
            task = task.split("The task is:")[-1].split(".")[0].strip()
        return task

    def _generate(self, messages, max_new_tokens=5):
        """Run VLM inference on messages."""
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        ).to(self.model.device)

        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        generated = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    def _score_comparative(self, predicted_view_pairs, task_prompt):
        """8-image comparative prompt — ask VLM to pick best action.

        Option order reversed (HOLD, LEFT, RIGHT) to mitigate position bias.
        """
        if len(predicted_view_pairs) != 3:
            return [50.0] * len(predicted_view_pairs)

        cur_base, cur_wrist = self._get_current_images()

        # Build 8 images in fixed order
        images = [cur_base, cur_wrist]
        for base, wrist in predicted_view_pairs:
            images.append(Image.fromarray(base))
            images.append(Image.fromarray(wrist))

        # Store exact images sent to VLM for debugging/report
        self.last_images = list(images)

        task = self._extract_task(task_prompt)

        # Options listed HOLD, LEFT, RIGHT to mitigate position bias
        prompt = (
            "These are images from a robot with two cameras:\n"
            "- Overhead camera: shows the arm and table from above\n"
            "- Wrist camera: mounted on the wrist, shows the claw and what's in front of it\n\n"
            "Current view:\n"
            "- Image 1: overhead camera NOW\n"
            "- Image 2: wrist camera NOW\n\n"
            "If the arm moves RIGHT:\n"
            "- Image 3: predicted overhead camera\n"
            "- Image 4: predicted wrist camera\n\n"
            "If the arm moves LEFT:\n"
            "- Image 5: predicted overhead camera\n"
            "- Image 6: predicted wrist camera\n\n"
            "If the arm HOLDS STILL:\n"
            "- Image 7: predicted overhead camera\n"
            "- Image 8: predicted wrist camera\n\n"
            f"Task: {task}\n\n"
            "Which action should the arm take? Answer with just: HOLD, LEFT, or RIGHT."
        )

        self.last_prompt = prompt
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": prompt})

        response = self._generate([{"role": "user", "content": content}], max_new_tokens=5)
        self.last_response = response

        # Parse RIGHT/LEFT/HOLD
        scores = [33.0, 33.0, 34.0]
        response_upper = response.upper()

        if "RIGHT" in response_upper:
            scores = [80.0, 10.0, 10.0]  # move+ wins
        elif "LEFT" in response_upper:
            scores = [10.0, 80.0, 10.0]  # move- wins
        elif "HOLD" in response_upper:
            scores = [10.0, 10.0, 80.0]  # hold wins

        return scores

    def _score_individual(self, predicted_view_pairs, task_prompt):
        """Score each prediction independently — ask VLM for a 1-100 score per action.

        Eliminates position bias by evaluating each prediction in isolation.
        """
        if len(predicted_view_pairs) != 3:
            return [50.0] * len(predicted_view_pairs)

        cur_base, cur_wrist = self._get_current_images()
        task = self._extract_task(task_prompt)

        scores = []
        responses = []
        all_images = [cur_base, cur_wrist]  # Start with observation for report

        for i, (pred_base, pred_wrist) in enumerate(predicted_view_pairs):
            action_label = ACTION_LABELS[i]
            pred_base_pil = Image.fromarray(pred_base)
            pred_wrist_pil = Image.fromarray(pred_wrist)

            all_images.extend([pred_base_pil, pred_wrist_pil])

            images = [cur_base, cur_wrist, pred_base_pil, pred_wrist_pil]

            prompt = (
                "These are images from a robot with two cameras.\n\n"
                "Current view:\n"
                "- Image 1: overhead camera NOW\n"
                "- Image 2: wrist camera NOW\n\n"
                f"Predicted view after the arm {action_label}:\n"
                "- Image 3: predicted overhead camera\n"
                "- Image 4: predicted wrist camera\n\n"
                f"Task: {task}\n\n"
                "How well does the predicted view (images 3-4) achieve the task "
                "compared to the current view (images 1-2)?\n"
                "Briefly explain what you see in one sentence, then give a score "
                "from 1 to 100 where 100 means the task is perfectly achieved.\n"
                "Format: REASON: <your reason> SCORE: <number>"
            )

            content = [{"type": "image", "image": img} for img in images]
            content.append({"type": "text", "text": prompt})

            response = self._generate([{"role": "user", "content": content}], max_new_tokens=80)
            responses.append(response)

            # Parse numeric score — prefer SCORE: N, fall back to last number
            score = 50.0  # default
            score_match = re.search(r'SCORE:\s*(\d+)', response, re.IGNORECASE)
            if score_match:
                score = max(1.0, min(100.0, float(score_match.group(1))))
            else:
                nums = re.findall(r'\d+', response)
                if nums:
                    score = max(1.0, min(100.0, float(nums[-1])))
            scores.append(score)

        self.last_images = all_images
        self.last_response = " | ".join(
            f"{ACTION_LABELS[i]}={r}" for i, r in enumerate(responses)
        )
        self.last_prompt = f"[score mode] 3 independent calls, task: {task}"

        return scores

    def unload(self):
        """Free GPU memory."""
        import gc
        if self.model is not None:
            del self.model
            self.model = None
            self.processor = None
            gc.collect()
            torch.cuda.empty_cache()
