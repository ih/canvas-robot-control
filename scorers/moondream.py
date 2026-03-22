"""Moondream2 VLM scorer — free-form VQA for numeric scoring."""

import re

import numpy as np
from PIL import Image

from .base import VLMScorer


class MoondreamScorer(VLMScorer):
    """Score frames using moondream2 (~1.9B params).

    Uses free-form VQA to ask the model for a numeric score.
    """

    def __init__(self, model_id: str = "vikhyatk/moondream2"):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None

    def load(self, device: str) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map={"": device},
        )
        self.model.eval()

    def score_frames(
        self, frames: list[np.ndarray], task_prompt: str
    ) -> list[float]:
        scores = []
        for frame in frames:
            pil_img = Image.fromarray(frame)
            enc_image = self.model.encode_image(pil_img)
            response = self.model.answer_question(enc_image, task_prompt, self.tokenizer)
            score = self._parse_score(response)
            scores.append(score)
        return scores

    def name(self) -> str:
        return "moondream2"

    @staticmethod
    def _parse_score(response: str) -> float:
        """Extract a numeric score from the VLM response."""
        match = re.search(r"(\d+(?:\.\d+)?)", response)
        if match:
            score = float(match.group(1))
            return min(score, 10.0)
        return 0.0
