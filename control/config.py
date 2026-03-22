"""Control loop configuration."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ControlConfig:
    # World model
    checkpoint_path: str = "../canvas-world-model/local/checkpoints/hold_exp/iter1/diff_finetune/best.pth"
    canvas_world_model_path: str = "../canvas-world-model"
    dataset_meta_path: str = ""  # auto-derived from checkpoint if empty
    num_inference_steps: int = 50

    # Robot
    robot_port: str = "COM3"
    robot_id: str = "my_so101_follower"
    step_size_degrees: float = 10.0
    joint_min: float = -60.0
    joint_max: float = 60.0
    control_joint: str = "shoulder_pan"

    # Cameras
    base_camera_index: int = 1
    wrist_camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 10

    # VLM scorer
    scorer: str = "qwen"
    task_prompt: str = "On a scale of 0 to 10, how centered is the object in this image? Reply with only a number."

    # Control loop
    max_steps: int = 50
    settle_time: float = 0.5
    success_threshold: float = 8.0
    dry_run: bool = False

    # Logging
    save_frames: bool = False
    output_dir: str = "local/runs"
