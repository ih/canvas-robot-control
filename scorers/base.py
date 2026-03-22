"""Abstract base class for VLM scorers."""

from abc import ABC, abstractmethod

import numpy as np


class VLMScorer(ABC):
    """Base interface for scoring predicted frames using a VLM."""

    @abstractmethod
    def load(self, device: str) -> None:
        """Load model weights onto device."""

    @abstractmethod
    def score_frames(self, frames: list[np.ndarray], task_prompt: str) -> list[float]:
        """Score each frame on a 1-100 scale based on the task prompt.

        Args:
            frames: List of RGB uint8 images (H, W, 3) to score.
            task_prompt: Natural language description of the desired outcome.

        Returns:
            List of float scores, one per frame. Higher = better.
        """

    @abstractmethod
    def name(self) -> str:
        """Return model identifier for logging."""
