"""SO-101 robot hardware abstraction.

Uses FeetechMotorsBus directly (proven pattern from robotic-foundation-model-tests)
with Windows camera patches for DSHOW backend.
"""

import json
import time
import platform
from pathlib import Path

import cv2
import numpy as np


# Joint configuration matching SO-101
JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def _apply_windows_camera_patches():
    """Disable background read thread on Windows DSHOW — threading issues.

    Also replaces read() with grab()/retrieve() to avoid cross-contamination
    when multiple DSHOW cameras are open simultaneously.
    """
    if platform.system() != "Windows":
        return

    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera

    OpenCVCamera._start_read_thread = lambda self: None

    def _sync_capture(self):
        raw_frame = self._read_from_hardware()
        processed = self._postprocess_image(raw_frame)
        with self.frame_lock:
            self.latest_frame = processed
            self.latest_timestamp = time.perf_counter()
        return processed

    def _patched_async_read(self, timeout_ms: float = 200):
        return self._sync_capture()

    def _patched_read(self, color_mode=None):
        return self._sync_capture()

    def _patched_read_latest(self, max_age_ms: int = 500):
        return self._sync_capture()

    OpenCVCamera._sync_capture = _sync_capture
    OpenCVCamera.async_read = _patched_async_read
    OpenCVCamera.read = _patched_read
    OpenCVCamera.read_latest = _patched_read_latest


class RobotInterface:
    """Hardware abstraction for SO-101 arm + cameras.

    Uses FeetechMotorsBus for motor control and OpenCVCamera for frame capture.
    """

    def __init__(
        self,
        port: str = "COM3",
        robot_id: str = "my_so101_follower",
        control_joint: str = "shoulder_pan",
        step_size: float = 10.0,
        joint_min: float = -60.0,
        joint_max: float = 60.0,
        base_camera_index: int = 1,
        wrist_camera_index: int = 0,
        camera_width: int = 320,
        camera_height: int = 240,
        camera_fps: int = 10,
    ):
        self.port = port
        self.robot_id = robot_id
        self.control_joint = control_joint
        self.step_size = step_size
        self.joint_min = joint_min
        self.joint_max = joint_max
        self.base_camera_index = base_camera_index
        self.wrist_camera_index = wrist_camera_index
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_fps = camera_fps

        self.bus = None
        self.base_camera = None
        self.wrist_camera = None

    def connect(self) -> None:
        """Connect to motors and cameras."""
        self._connect_motors()
        self._connect_cameras()

    def disconnect(self) -> None:
        """Disconnect motors and cameras safely."""
        if self.base_camera is not None:
            self.base_camera.disconnect()
            self.base_camera = None
        if self.wrist_camera is not None:
            self.wrist_camera.disconnect()
            self.wrist_camera = None
        if self.bus is not None:
            self.bus.disconnect()
            self.bus = None

    def get_state(self) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Read current cameras and motor positions.

        Returns:
            Tuple of (cameras_dict, motor_positions):
            - cameras_dict: {"base": (H,W,3), "wrist": (H,W,3)} RGB uint8
            - motor_positions: (6,) float array of joint positions
        """
        # DSHOW multi-camera fix: grab() both first, then retrieve().
        # Using read() (grab+retrieve) sequentially causes the second
        # camera to return the first camera's frame.
        # Flush stale buffers with extra grabs before the real capture.
        for _ in range(3):
            self.base_camera.videocapture.grab()
            self.wrist_camera.videocapture.grab()
        _, base_raw = self.base_camera.videocapture.retrieve()
        _, wrist_raw = self.wrist_camera.videocapture.retrieve()

        # _postprocess_image handles BGR->RGB conversion and rotation
        base_rgb = self.base_camera._postprocess_image(base_raw)
        wrist_rgb = self.wrist_camera._postprocess_image(wrist_raw)

        cameras = {"base": base_rgb, "wrist": wrist_rgb}

        # Read motor positions
        positions = self.bus.sync_read("Present_Position")
        motor_array = np.array(
            [positions[j] for j in JOINTS], dtype=np.float32
        )

        return cameras, motor_array

    def execute_action(self, action_int: int) -> None:
        """Execute a discrete action on the control joint.

        Args:
            action_int: 1=move positive, 2=move negative, 3=hold.
        """
        self.execute_action_on(action_int, self.control_joint)

    def execute_action_on(self, action_int: int, joint: str) -> None:
        """Same as execute_action but targets an arbitrary joint by name.

        Uses the same [joint_min, joint_max] clamp as the configured control
        joint. Intended for live-inference QA where the operator wants to
        probe any joint, not just the one the world model was trained on.
        """
        if joint not in JOINTS:
            raise ValueError(f"unknown joint {joint!r}; expected one of {JOINTS}")
        positions = self.bus.sync_read("Present_Position")
        current = positions[joint]

        if action_int == 1:
            target = current + self.step_size
        elif action_int == 2:
            target = current - self.step_size
        else:
            target = current

        target = max(self.joint_min, min(self.joint_max, target))

        goal = {j: positions[j] for j in JOINTS}
        goal[joint] = target
        self.bus.sync_write("Goal_Position", goal)

    def relax(self) -> None:
        """Disable torque on all joints so the arm can be moved by hand."""
        self.bus.disable_torque()

    def lock(self) -> None:
        """Re-enable torque holding the CURRENT position so the arm stays
        where the operator left it. Read pos -> set as Goal_Position ->
        enable torque (which also sets the Lock register)."""
        positions = self.bus.sync_read("Present_Position", normalize=False)
        self.bus.enable_torque()
        self.bus.sync_write("Goal_Position", positions, normalize=False)

    def _connect_motors(self) -> None:
        """Connect to FeetechMotorsBus with calibration."""
        from lerobot.motors.feetech import FeetechMotorsBus
        from lerobot.motors import Motor, MotorNormMode, MotorCalibration

        # Load calibration
        cal_dir = (
            Path.home()
            / ".cache"
            / "huggingface"
            / "lerobot"
            / "calibration"
            / "robots"
            / "so101_follower"
        )
        cal_path = cal_dir / f"{self.robot_id}.json"
        calibration = None
        if cal_path.is_file():
            with open(cal_path) as f:
                cal_dict = json.load(f)
            calibration = {
                motor: MotorCalibration(**cal_data)
                for motor, cal_data in cal_dict.items()
            }

        self.bus = FeetechMotorsBus(
            port=self.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
                "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
                "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
                "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
                "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=calibration,
        )
        self.bus.connect()

    def _connect_cameras(self) -> None:
        """Connect to OpenCV cameras with Windows patches."""
        _apply_windows_camera_patches()

        from lerobot.cameras.opencv.camera_opencv import OpenCVCamera, OpenCVCameraConfig
        from lerobot.cameras.configs import Cv2Backends

        # Match dataset recording config: explicit DSHOW backend and 2s warmup
        base_cfg = OpenCVCameraConfig(
            index_or_path=self.base_camera_index,
            width=self.camera_width,
            height=self.camera_height,
            fps=self.camera_fps,
            rotation=180,
            backend=Cv2Backends.DSHOW,
            warmup_s=2,
        )
        self.base_camera = OpenCVCamera(base_cfg)
        self.base_camera.connect()
        # Fix white balance — auto WB makes red appear blue
        if hasattr(self.base_camera, 'videocapture'):
            self.base_camera.videocapture.set(cv2.CAP_PROP_AUTO_WB, 0)
            self.base_camera.videocapture.set(cv2.CAP_PROP_WB_TEMPERATURE, 6500)

        wrist_cfg = OpenCVCameraConfig(
            index_or_path=self.wrist_camera_index,
            width=self.camera_width,
            height=self.camera_height,
            fps=self.camera_fps,
            rotation=180,
            backend=Cv2Backends.DSHOW,
            warmup_s=2,
        )
        self.wrist_camera = OpenCVCamera(wrist_cfg)
        self.wrist_camera.connect()
        if hasattr(self.wrist_camera, 'videocapture'):
            self.wrist_camera.videocapture.set(cv2.CAP_PROP_AUTO_WB, 0)
            self.wrist_camera.videocapture.set(cv2.CAP_PROP_WB_TEMPERATURE, 6500)


class DryRunRobotInterface:
    """Mock robot interface for testing without hardware.

    Returns synthetic camera frames and motor positions.
    """

    def __init__(self, control_joint: str = "shoulder_pan", step_size: float = 10.0,
                 joint_min: float = -60.0, joint_max: float = 60.0):
        self.control_joint = control_joint
        self.step_size = step_size
        self.joint_min = joint_min
        self.joint_max = joint_max
        self._positions = {j: 0.0 for j in JOINTS}

    def connect(self) -> None:
        print("[DRY RUN] Robot connected (mock)")

    def disconnect(self) -> None:
        print("[DRY RUN] Robot disconnected (mock)")

    def get_state(self) -> tuple[dict[str, np.ndarray], np.ndarray]:
        # Generate synthetic frames (gray with position indicator)
        base = np.full((480, 640, 3), 128, dtype=np.uint8)
        wrist = np.full((480, 640, 3), 100, dtype=np.uint8)

        # Draw a circle representing an "object" offset by shoulder_pan position
        cx = 320 + int(self._positions["shoulder_pan"] * 3)
        cv2.circle(base, (cx, 240), 30, (0, 0, 255), -1)

        cameras = {"base": base, "wrist": wrist}
        motor_array = np.array(
            [self._positions[j] for j in JOINTS], dtype=np.float32
        )
        return cameras, motor_array

    def execute_action(self, action_int: int) -> None:
        self.execute_action_on(action_int, self.control_joint)

    def execute_action_on(self, action_int: int, joint: str) -> None:
        if joint not in JOINTS:
            raise ValueError(f"unknown joint {joint!r}; expected one of {JOINTS}")
        current = self._positions[joint]
        if action_int == 1:
            target = current + self.step_size
        elif action_int == 2:
            target = current - self.step_size
        else:
            target = current
        target = max(self.joint_min, min(self.joint_max, target))
        self._positions[joint] = target
        print(f"[DRY RUN] {joint}: {current:.1f} -> {target:.1f} (action={action_int})")

    def relax(self) -> None:
        print("[DRY RUN] relax (torque disabled)")

    def lock(self) -> None:
        print("[DRY RUN] lock (torque enabled at current position)")
