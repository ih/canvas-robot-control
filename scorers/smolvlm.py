"""SmolVLM2 scorer — ultra-lightweight VLM (256M params)."""

import re

import numpy as np
from PIL import Image

from .base import VLMScorer


class SmolVLMScorer(VLMScorer):
    """Score frames using SmolVLM2 (~256M params).

    Extremely fast inference, instruction-tuned for following prompts.
    """

    def __init__(self, model_id: str = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"):
        self.model_id = model_id
        self.model = None
        self.processor = None

    def load(self, device: str) -> None:
        from transformers import SmolVLMForConditionalGeneration, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = SmolVLMForConditionalGeneration.from_pretrained(
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
        return "smolvlm2"

    def _score_single(self, frame: np.ndarray, task_prompt: str) -> float:
        import torch

        pil_img = Image.fromarray(frame)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": task_prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[pil_img], return_tensors="pt", padding=True
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
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": "Describe this image in one sentence."},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[pil_img], return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=100)
        input_len = inputs["input_ids"].shape[1]
        output_ids = generated_ids[:, input_len:]
        return self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    @staticmethod
    def _parse_score(response: str) -> float:
        """Extract a numeric score from the VLM response."""
        match = re.search(r"(\d+(?:\.\d+)?)", response)
        if match:
            score = float(match.group(1))
            return min(score, 100.0)
        return 0.0
