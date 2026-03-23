"""PaliGemma2 scorer — 3B VLM with 224px native input (mix fine-tuned)."""

import re

import numpy as np
from PIL import Image

from .base import VLMScorer


class PaliGemmaScorer(VLMScorer):
    """Score frames using PaliGemma2 (3B params, 224px input).

    Native 224px input matches workspace frame size exactly.
    Mix fine-tuned variant with better instruction following.
    """

    def __init__(self, model_id: str = "google/paligemma2-3b-mix-224"):
        self.model_id = model_id
        self.model = None
        self.processor = None

    def load(self, device: str) -> None:
        from transformers import PaliGemmaForConditionalGeneration, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map={"": device},
        )
        self.model.eval()
        self._device = device

    def score_frames(
        self, frames: list[np.ndarray], task_prompt: str
    ) -> list[float]:
        scores = []
        for frame in frames:
            score = self._score_single(frame, task_prompt)
            scores.append(score)
        return scores

    def name(self) -> str:
        return "paligemma2"

    def _score_single(self, frame: np.ndarray, task_prompt: str) -> float:
        import torch

        pil_img = Image.fromarray(frame)

        # PaliGemma works best with short prefix-style prompts
        # Extract the core task from the full prompt for better results
        short_prompt = self._shorten_prompt(task_prompt)
        inputs = self.processor(
            images=pil_img, text=short_prompt, return_tensors="pt"
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=32)

        input_len = inputs["input_ids"].shape[1]
        output_ids = generated_ids[:, input_len:]
        response = self.processor.batch_decode(
            output_ids, skip_special_tokens=True
        )[0]

        return self._parse_score(response)

    def describe_frame(self, frame: np.ndarray) -> str:
        import torch

        pil_img = Image.fromarray(frame)
        inputs = self.processor(
            images=pil_img, text="describe", return_tensors="pt"
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=100)
        input_len = inputs["input_ids"].shape[1]
        output_ids = generated_ids[:, input_len:]
        return self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    @staticmethod
    def _shorten_prompt(task_prompt: str) -> str:
        """Convert verbose VQA prompt to PaliGemma-friendly short prefix."""
        # Extract task from "The task is: X." pattern
        import re as _re
        match = _re.search(r"task is:\s*(.+?)\.", task_prompt)
        task = match.group(1).strip() if match else "rate the image"
        return f"on a scale of 1-100, rate how well this shows: {task}"

    @staticmethod
    def _parse_score(response: str) -> float:
        """Extract a numeric score from the VLM response."""
        match = re.search(r"(\d+(?:\.\d+)?)", response)
        if match:
            score = float(match.group(1))
            return min(score, 100.0)
        return 0.0
