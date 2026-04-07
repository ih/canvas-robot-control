"""Live canvas construction matching canvas-world-model training distribution.

Builds canvases from live camera frames + motor state for world model inference.
"""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# Action-to-color mapping (must match canvas-world-model/data/canvas_builder.py)
ACTION_COLORS = {
    0: (255, 255, 0),   # Yellow: buffer
    1: (0, 255, 0),     # Green: move positive
    2: (0, 0, 255),     # Blue: move negative
    3: (255, 0, 0),     # Red: stay/hold
}

# Canvas constants (must match canvas-world-model/config.py)
SEPARATOR_WIDTH = 32
MOTOR_STRIP_HEIGHT = 16
FRAME_SIZE = (224, 224)  # per-camera (H, W)
PATCH_SIZE = 16


def load_dataset_meta(meta_path: str) -> dict:
    """Load dataset metadata for motor normalization bounds."""
    with open(meta_path) as f:
        return json.load(f)


def stack_cameras_vertically(
    base_frame: np.ndarray,
    wrist_frame: np.ndarray,
    target_size: tuple[int, int] = FRAME_SIZE,
) -> np.ndarray:
    """Stack two camera frames vertically, resized to target size each.

    Args:
        base_frame: RGB image from base camera (any size).
        wrist_frame: RGB image from wrist camera (any size).
        target_size: (H, W) per camera.

    Returns:
        Stacked frame of shape (2*H, W, 3), dtype uint8.
    """
    h, w = target_size
    base_resized = cv2.resize(base_frame, (w, h), interpolation=cv2.INTER_LANCZOS4)
    wrist_resized = cv2.resize(wrist_frame, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return np.vstack([base_resized, wrist_resized])


def render_motor_strip(
    motor_state: np.ndarray,
    norm_min: np.ndarray,
    norm_max: np.ndarray,
    motor_velocity: np.ndarray | None = None,
    vel_norm_max: np.ndarray | None = None,
    strip_height: int = MOTOR_STRIP_HEIGHT,
    frame_width: int = FRAME_SIZE[1],
    patch_size: int = PATCH_SIZE,
) -> np.ndarray:
    """Render motor positions/velocities as grayscale patches.

    Mirrors canvas-world-model/data/canvas_builder.py::_render_motor_strip exactly.
    """
    num_joints = len(motor_state)
    strip = np.zeros((strip_height, frame_width, 3), dtype=np.uint8)

    # Normalize positions to [0, 1]
    pos_range = norm_max - norm_min
    pos_range = np.where(pos_range < 1e-8, 1.0, pos_range)
    norm_pos = np.clip((motor_state - norm_min) / pos_range, 0.0, 1.0)

    # Position patches
    for j in range(num_joints):
        x_start = j * patch_size
        x_end = x_start + patch_size
        if x_end > frame_width:
            break
        gray_val = int(norm_pos[j] * 255)
        strip[:, x_start:x_end] = gray_val

    # Velocity patches
    for j in range(num_joints):
        x_start = (num_joints + j) * patch_size
        x_end = x_start + patch_size
        if x_end > frame_width:
            break
        if motor_velocity is None:
            gray_val = 128  # mid-gray = zero velocity
        else:
            if vel_norm_max is not None and vel_norm_max[j] > 1e-8:
                norm_vel = np.clip(motor_velocity[j] / vel_norm_max[j], -1.0, 1.0)
            else:
                norm_vel = 0.0
            gray_val = int((norm_vel * 0.5 + 0.5) * 255)
        strip[:, x_start:x_end] = gray_val

    return strip


def build_live_canvas(
    context_frame: np.ndarray,
    action_int: int,
    motor_state_current: np.ndarray,
    motor_state_next: np.ndarray,
    meta: dict,
) -> np.ndarray:
    """Build a canvas from a live context frame and a candidate action.

    The prediction target (right side) is left as zeros — the world model
    will generate it.

    Args:
        context_frame: Stacked camera frame (2*224, 224, 3), uint8.
        action_int: Discrete action (1=move+, 2=move-, 3=hold).
        motor_state_current: Current joint positions, shape (num_joints,).
        motor_state_next: Expected next joint positions, shape (num_joints,).
        meta: Dataset metadata dict with normalization bounds.

    Returns:
        Canvas as uint8 array, shape (canvas_h, canvas_w, 3).
    """
    frame_h, frame_w = context_frame.shape[:2]
    norm_min = np.array(meta["motor_norm_min"])
    norm_max = np.array(meta["motor_norm_max"])
    vel_norm_max = np.array(meta["motor_vel_norm_max"]) if "motor_vel_norm_max" in meta else None
    strip_h = meta.get("motor_strip_height", MOTOR_STRIP_HEIGHT)

    total_h = frame_h + strip_h
    total_w = frame_w * 2 + SEPARATOR_WIDTH
    canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)

    # Place context frame (left side)
    canvas[:frame_h, :frame_w] = context_frame

    # Motor strip for context frame (no velocity for simplicity — first frame)
    motor_strip_ctx = render_motor_strip(
        motor_state_current, norm_min, norm_max,
        motor_velocity=None, vel_norm_max=vel_norm_max,
        strip_height=strip_h, frame_width=frame_w,
    )
    canvas[frame_h:, :frame_w] = motor_strip_ctx

    # Action separator
    sep_start = frame_w
    sep_end = frame_w + SEPARATOR_WIDTH
    sep_color = ACTION_COLORS.get(action_int, (128, 128, 128))
    canvas[:, sep_start:sep_end] = sep_color

    # Prediction target (right side) — zeros, world model fills this
    # But we do need the motor strip for the expected next state
    motor_velocity = motor_state_next - motor_state_current
    motor_strip_next = render_motor_strip(
        motor_state_next, norm_min, norm_max,
        motor_velocity=motor_velocity, vel_norm_max=vel_norm_max,
        strip_height=strip_h, frame_width=frame_w,
    )
    pred_start = sep_end
    canvas[frame_h:, pred_start:pred_start + frame_w] = motor_strip_next

    return canvas


def canvas_to_tensor(canvas: np.ndarray, normalize_mode: str = "neg_one_one"):
    """Convert uint8 canvas to model input tensor.

    Args:
        canvas: (H, W, 3) uint8 array.
        normalize_mode: "neg_one_one" for diffusion, "zero_one" for GPT/MAE.

    Returns:
        torch.Tensor of shape (1, 3, H, W).
    """
    import torch

    img = canvas.astype(np.float32) / 255.0
    if normalize_mode == "neg_one_one":
        img = img * 2.0 - 1.0
    # HWC -> CHW, add batch dim
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return tensor


def extract_predicted_frame(
    canvas_tensor, frame_h: int, frame_w: int, strip_h: int = MOTOR_STRIP_HEIGHT,
) -> np.ndarray:
    """Extract the predicted frame (right side, visual only) from a canvas tensor.

    Args:
        canvas_tensor: (B, 3, H, W) tensor in [0, 1] range.
        frame_h: Height of visual frame (e.g., 448).
        frame_w: Width of each frame (e.g., 224).
        strip_h: Motor strip height.

    Returns:
        (frame_h, frame_w, 3) uint8 array.
    """
    pred_x_start = frame_w + SEPARATOR_WIDTH
    # Extract visual region only (no motor strip)
    frame = canvas_tensor[0, :, :frame_h, pred_x_start:pred_x_start + frame_w]
    # CHW -> HWC, to uint8
    frame = frame.permute(1, 2, 0).cpu().numpy()
    frame = (frame * 255).clip(0, 255).astype(np.uint8)
    return frame


def extract_workspace_view(predicted_frame: np.ndarray) -> np.ndarray:
    """Extract just the workspace (base) camera view from a stacked frame.

    The frame has base camera on top and wrist camera on bottom,
    each 224px tall. Returns the top half for VLM scoring.

    Args:
        predicted_frame: (448, 224, 3) uint8 array.

    Returns:
        (224, 224, 3) uint8 array — workspace camera view only.
    """
    h = predicted_frame.shape[0] // 2
    return predicted_frame[:h]


def extract_both_views(predicted_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract both camera views from a stacked predicted frame.

    The frame has base camera on top and wrist camera on bottom,
    each 224px tall.

    Args:
        predicted_frame: (448, 224, 3) uint8 array.

    Returns:
        Tuple of (base_view, wrist_view), each (224, 224, 3) uint8.
    """
    h = predicted_frame.shape[0] // 2
    return predicted_frame[:h].copy(), predicted_frame[h:].copy()
