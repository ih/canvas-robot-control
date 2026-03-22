"""Florence-2 scorer — object detection + geometric centering score."""

import re

import numpy as np
from PIL import Image

from .base import VLMScorer


class FlorenceScorer(VLMScorer):
    """Score frames using Florence-2 object detection.

    Detects objects, finds the best matching bounding box, and computes
    a centering score based on how close the box center is to the frame center.
    """

    def __init__(
        self,
        model_id: str = "microsoft/Florence-2-base",
        object_label: str = "",
    ):
        self.model_id = model_id
        self.object_label = object_label
        self.model = None
        self.processor = None

    def load(self, device: str) -> None:
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map={"": device},
        )
        self.model.eval()
        self._device = device

    def score_frames(
        self, frames: list[np.ndarray], task_prompt: str
    ) -> list[float]:
        # Extract object label from task prompt if not set
        label = self.object_label or self._extract_object(task_prompt)

        scores = []
        for frame in frames:
            pil_img = Image.fromarray(frame)
            score = self._score_centering(pil_img, label)
            scores.append(score)
        return scores

    def name(self) -> str:
        return "florence2"

    def _score_centering(self, image: Image.Image, label: str) -> float:
        """Detect objects and score how centered the best match is."""
        import torch

        prompt = "<OD>"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=1024, num_beams=3
            )

        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        result = self.processor.post_process_generation(
            generated_text, task="<OD>", image_size=image.size
        )

        bboxes = result.get("<OD>", {}).get("bboxes", [])
        labels = result.get("<OD>", {}).get("labels", [])

        if not bboxes:
            return 0.0

        img_w, img_h = image.size
        cx, cy = img_w / 2, img_h / 2
        max_dist = ((img_w / 2) ** 2 + (img_h / 2) ** 2) ** 0.5

        best_score = 0.0
        for bbox, det_label in zip(bboxes, labels):
            # Filter by label if specified
            if label and label.lower() not in det_label.lower():
                continue

            x1, y1, x2, y2 = bbox
            box_cx = (x1 + x2) / 2
            box_cy = (y1 + y2) / 2
            dist = ((box_cx - cx) ** 2 + (box_cy - cy) ** 2) ** 0.5
            score = max(0.0, 10.0 * (1.0 - dist / max_dist))
            best_score = max(best_score, score)

        # If no label matched but we have detections, use closest one
        if best_score == 0.0 and bboxes and not label:
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                box_cx = (x1 + x2) / 2
                box_cy = (y1 + y2) / 2
                dist = ((box_cx - cx) ** 2 + (box_cy - cy) ** 2) ** 0.5
                score = max(0.0, 10.0 * (1.0 - dist / max_dist))
                best_score = max(best_score, score)

        return best_score

    @staticmethod
    def _extract_object(prompt: str) -> str:
        """Try to extract an object name from the task prompt."""
        match = re.search(
            r"(?:center|find|locate|detect)\s+(?:the\s+)?(.+?)(?:\s+in|\?|$)",
            prompt,
            re.IGNORECASE,
        )
        return match.group(1).strip() if match else ""
