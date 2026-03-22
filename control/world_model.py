"""World model inference wrapper.

Loads a trained diffusion model from canvas-world-model and runs batched
prediction for candidate actions.
"""

import sys
import json
from pathlib import Path

import numpy as np
import torch

from .canvas_utils import (
    build_live_canvas,
    canvas_to_tensor,
    extract_predicted_frame,
    extract_workspace_view,
    SEPARATOR_WIDTH,
    MOTOR_STRIP_HEIGHT,
)


class WorldModelPredictor:
    """Wraps a trained canvas world model for live inference.

    Loads the diffusion checkpoint, builds candidate canvases for each
    discrete action, and returns predicted next-frames.
    """

    def __init__(
        self,
        checkpoint_path: str,
        canvas_world_model_path: str,
        num_inference_steps: int = 50,
        device: str | None = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.cwm_path = Path(canvas_world_model_path)
        self.num_inference_steps = num_inference_steps
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.noise_scheduler = None
        self.saved_args = None
        self.meta = None
        self.patch_mask = None

    def load(self) -> None:
        """Load model weights and metadata."""
        # Add canvas-world-model to path for imports
        cwm_str = str(self.cwm_path.resolve())
        if cwm_str not in sys.path:
            sys.path.insert(0, cwm_str)

        from models.diffusion import ConditionalDiffusionViT, NoiseScheduler
        from models.common import compute_last_frame_patch_mask

        ckpt = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False
        )
        self.saved_args = ckpt["args"]

        # Load dataset metadata for canvas dimensions and motor normalization
        dataset_path = self.saved_args["dataset"]
        meta_path = Path(dataset_path) / "dataset_meta.json"
        if not meta_path.exists():
            # Try relative to canvas-world-model
            meta_path = self.cwm_path / dataset_path / "dataset_meta.json"
        with open(meta_path) as f:
            self.meta = json.load(f)

        canvas_h, canvas_w = self.meta["canvas_size"]
        patch_size = self.saved_args["patch_size"]

        # Build model
        self.model = ConditionalDiffusionViT(
            img_height=canvas_h,
            img_width=canvas_w,
            patch_size=patch_size,
            embed_dim=self.saved_args["embed_dim"],
            depth=self.saved_args["depth"],
            num_heads=self.saved_args["num_heads"],
            prediction_type=self.saved_args["prediction_type"],
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        # Noise scheduler
        self.noise_scheduler = NoiseScheduler(
            num_train_timesteps=self.saved_args["num_train_timesteps"],
            beta_schedule=self.saved_args["beta_schedule"],
            prediction_type=self.saved_args["prediction_type"],
        )

        # Patch mask for last frame region
        num_frames = self.meta["canvas_history_size"]
        sep_width = self.meta["separator_width"]
        self.patch_mask = compute_last_frame_patch_mask(
            canvas_h, canvas_w, patch_size, num_frames, sep_width, device=self.device,
        )

        self._canvas_h = canvas_h
        self._canvas_w = canvas_w
        self._patch_size = patch_size
        self._grid_h = canvas_h // patch_size
        self._grid_w = canvas_w // patch_size

        # Frame dimensions (visual only, no motor strip)
        strip_h = self.meta.get("motor_strip_height", MOTOR_STRIP_HEIGHT)
        self._frame_h = canvas_h - strip_h
        self._frame_w = (canvas_w - (num_frames - 1) * sep_width) // num_frames
        self._strip_h = strip_h

    @torch.no_grad()
    def predict_batch(
        self,
        context_frame: np.ndarray,
        motor_state: np.ndarray,
        actions: list[int],
        step_size: float = 10.0,
        control_joint_idx: int = 0,
    ) -> list[np.ndarray]:
        """Predict next frame for each candidate action.

        Args:
            context_frame: Stacked camera frame (448, 224, 3), uint8.
            motor_state: Current joint positions, shape (num_joints,).
            actions: List of discrete actions to evaluate (e.g., [1, 2, 3]).
            step_size: Degrees per discrete step for the control joint.
            control_joint_idx: Index of the joint being controlled.

        Returns:
            List of predicted workspace camera views, one per action.
            Each is (224, 224, 3) uint8.
        """
        from models.common import patchify, unpatchify

        # Build canvases for each action
        canvases = []
        for action in actions:
            # Compute expected next motor state
            motor_next = motor_state.copy()
            if action == 1:  # move positive
                motor_next[control_joint_idx] += step_size
            elif action == 2:  # move negative
                motor_next[control_joint_idx] -= step_size
            # action 3 = hold, no change

            canvas = build_live_canvas(
                context_frame, action, motor_state, motor_next, self.meta
            )
            canvases.append(canvas)

        # Stack into batch tensor
        tensors = [canvas_to_tensor(c, "neg_one_one") for c in canvases]
        batch = torch.cat(tensors, dim=0).to(self.device)  # (B, 3, H, W)

        B = batch.shape[0]
        batch_mask = self.patch_mask.expand(B, -1)
        ps = self._patch_size
        gh, gw = self._grid_h, self._grid_w

        # Start from noise in last frame region
        target_patches = patchify(batch, ps)
        noisy_patches = target_patches.clone()
        noisy_patches[batch_mask] = torch.randn_like(noisy_patches[batch_mask])
        current = unpatchify(noisy_patches, ps, gh, gw)

        # DDIM denoising
        step_sz = self.noise_scheduler.num_train_timesteps // self.num_inference_steps
        timesteps = list(
            range(self.noise_scheduler.num_train_timesteps - 1, -1, -step_sz)
        )

        for t in timesteps:
            t_batch = torch.full((B,), t, device=self.device, dtype=torch.long)
            pred = self.model(current, t_batch)

            current_patches = patchify(current, ps)
            denoised_patches = current_patches.clone()
            pred_last = pred[batch_mask]
            current_last = current_patches[batch_mask]
            stepped = self.noise_scheduler.step(pred_last, t, current_last)
            denoised_patches[batch_mask] = stepped
            current = unpatchify(denoised_patches, ps, gh, gw)

        # Convert [-1,1] -> [0,1]
        output = current.clamp(-1, 1) * 0.5 + 0.5

        # Extract predicted workspace views
        results = []
        for i in range(B):
            frame = extract_predicted_frame(
                output[i:i+1], self._frame_h, self._frame_w, self._strip_h,
            )
            workspace = extract_workspace_view(frame)
            results.append(workspace)

        return results
