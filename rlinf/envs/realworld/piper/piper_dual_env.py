# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import queue
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import gymnasium as gym
import numpy as np

from rlinf.envs.realworld.common.camera import Camera, CameraInfo
from rlinf.envs.realworld.common.video_player import VideoPlayer
from rlinf.scheduler import WorkerInfo
from rlinf.utils.logging import get_logger

from .piper_env import PIPER_ACTION_DIM, PIPER_ARM_JOINT_DIM
from .piper_robot_state import PiperRobotState

PIPER_DUAL_ARM_NUM = 2
PIPER_DUAL_ACTION_DIM = PIPER_DUAL_ARM_NUM * PIPER_ACTION_DIM
PIPER_DUAL_JOINT_DIM = PIPER_DUAL_ARM_NUM * PIPER_ARM_JOINT_DIM


@dataclass
class PiperDualRobotConfig:
    can_names: Optional[list[str]] = None
    camera_serials: Optional[list[str]] = None
    enable_camera_player: bool = True

    is_dummy: bool = False
    use_dense_reward: bool = False
    step_frequency: float = 10.0

    target_joint_pose: np.ndarray = field(
        default_factory=lambda: np.zeros((PIPER_DUAL_ARM_NUM, PIPER_ARM_JOINT_DIM))
    )
    reset_joint_pose: np.ndarray = field(
        default_factory=lambda: np.zeros((PIPER_DUAL_ARM_NUM, PIPER_ARM_JOINT_DIM))
    )
    max_num_steps: int = 100
    reward_threshold: np.ndarray = field(
        default_factory=lambda: np.zeros((PIPER_DUAL_ARM_NUM, PIPER_ARM_JOINT_DIM))
    )
    enable_random_reset: bool = False

    joint_limit_min: np.ndarray = field(
        default_factory=lambda: np.full((PIPER_DUAL_ARM_NUM, PIPER_ARM_JOINT_DIM), -3.14)
    )
    joint_limit_max: np.ndarray = field(
        default_factory=lambda: np.full((PIPER_DUAL_ARM_NUM, PIPER_ARM_JOINT_DIM), 3.14)
    )
    success_hold_steps: int = 1


class PiperDualEnv(gym.Env):
    """Dual-arm Piper env with 14D action: [left(7), right(7)]."""

    def __init__(
        self,
        config: PiperDualRobotConfig,
        worker_info: Optional[WorkerInfo],
        hardware_info: Optional[Any],
        env_idx: int,
    ):
        self._logger = get_logger()
        self.config = config
        self.hardware_info = hardware_info
        self.env_idx = env_idx
        self.node_rank = 0
        self.env_worker_rank = 0
        if worker_info is not None:
            self.node_rank = worker_info.cluster_node_rank
            self.env_worker_rank = worker_info.rank

        self._states = [PiperRobotState(), PiperRobotState()]
        self._num_steps = 0
        self._success_hold_counter = 0

        self._reset_pose = self._as_2x6(self.config.reset_joint_pose, "reset_joint_pose")
        self._target_pose = self._as_2x6(self.config.target_joint_pose, "target_joint_pose")
        self._reward_thr = self._as_2x6(self.config.reward_threshold, "reward_threshold")
        self._joint_lo = self._as_2x6(self.config.joint_limit_min, "joint_limit_min")
        self._joint_hi = self._as_2x6(self.config.joint_limit_max, "joint_limit_max")

        if not self.config.is_dummy:
            self._setup_hardware()

        assert (
            self.config.camera_serials is not None and len(self.config.camera_serials) > 0
        ), "At least one camera serial must be provided for PiperDualEnv."
        self._init_action_obs_spaces()

        if self.config.is_dummy:
            return

        start_time = time.time()
        while not (self._left_controller.is_robot_up() and self._right_controller.is_robot_up()):
            time.sleep(0.5)
            if time.time() - start_time > 10:
                self._logger.warning("Waiting for dual Piper robots to be ready.")

        self.go_to_rest()
        self._sync_states()
        self._open_cameras()
        self.camera_player = VideoPlayer(self.config.enable_camera_player)

    def _as_2x6(self, value: np.ndarray, name: str) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64)
        if arr.shape == (PIPER_DUAL_ARM_NUM, PIPER_ARM_JOINT_DIM):
            return arr.copy()
        if arr.shape == (PIPER_DUAL_JOINT_DIM,):
            return arr.reshape(PIPER_DUAL_ARM_NUM, PIPER_ARM_JOINT_DIM).copy()
        raise ValueError(f"{name} must be shape (2, 6) or (12,), got {arr.shape}")

    def _setup_hardware(self):
        from .piper_controller import PiperController

        can_names = self.config.can_names
        if can_names is None and self.hardware_info is not None:
            hw_cfg = getattr(self.hardware_info, "config", None)
            if hw_cfg is not None:
                can_names = getattr(hw_cfg, "can_names", None)
                if can_names is None:
                    left = getattr(hw_cfg, "can_name_left", None)
                    right = getattr(hw_cfg, "can_name_right", None)
                    if left is not None and right is not None:
                        can_names = [left, right]
        if can_names is None:
            raise ValueError("Dual Piper requires can_names: ['can_left', 'can_right']")
        if len(can_names) != PIPER_DUAL_ARM_NUM:
            raise ValueError(f"Expected 2 can_names, got {can_names}")
        self.config.can_names = [str(x) for x in can_names]

        if self.config.camera_serials is None and self.hardware_info is not None:
            self.config.camera_serials = getattr(self.hardware_info.config, "camera_serials", None)
        if self.config.camera_serials is not None:
            self.config.camera_serials = [str(serial) for serial in self.config.camera_serials]

        self._left_controller = PiperController.launch_controller(
            can_name=self.config.can_names[0],
            env_idx=self.env_idx * 2,
            node_rank=self.node_rank,
            worker_rank=self.env_worker_rank,
        )
        self._right_controller = PiperController.launch_controller(
            can_name=self.config.can_names[1],
            env_idx=self.env_idx * 2 + 1,
            node_rank=self.node_rank,
            worker_rank=self.env_worker_rank,
        )

    def _normalized_action_to_dual_command(self, action: np.ndarray) -> np.ndarray:
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        assert a.shape == (PIPER_DUAL_ACTION_DIM,), (
            f"Dual Piper action must be shape ({PIPER_DUAL_ACTION_DIM},), got {a.shape}"
        )
        a = np.clip(a, self.action_space.low, self.action_space.high)
        action_2x7 = a.reshape(PIPER_DUAL_ARM_NUM, PIPER_ACTION_DIM)
        command = np.zeros((PIPER_DUAL_ARM_NUM, PIPER_ACTION_DIM), dtype=np.float64)
        t = (np.clip(action_2x7[:, :PIPER_ARM_JOINT_DIM], -1.0, 1.0) + 1.0) * 0.5
        command[:, :PIPER_ARM_JOINT_DIM] = self._joint_lo + t * (self._joint_hi - self._joint_lo)
        command[:, PIPER_ARM_JOINT_DIM] = np.clip(action_2x7[:, PIPER_ARM_JOINT_DIM], -1.0, 1.0)
        return command

    def _sync_states(self):
        if self.config.is_dummy:
            return
        self._states[0] = self._left_controller.get_state().wait()[0]
        self._states[1] = self._right_controller.get_state().wait()[0]

    def step(self, action: np.ndarray):
        start_time = time.time()
        command = self._normalized_action_to_dual_command(action)
        if not self.config.is_dummy:
            self._left_controller.move_arm(command[0]).wait()
            self._right_controller.move_arm(command[1]).wait()

        self._num_steps += 1
        step_time = time.time() - start_time
        time.sleep(max(0, (1.0 / self.config.step_frequency) - step_time))
        self._sync_states()

        obs = self._get_observation()
        reward = self._calc_step_reward()
        terminated = (reward == 1.0) and (self._success_hold_counter >= self.config.success_hold_steps)
        truncated = self._num_steps >= self.config.max_num_steps
        return obs, reward, terminated, truncated, {}

    def _calc_step_reward(self) -> float:
        if self.config.is_dummy:
            return 0.0
        curr = np.stack([s.arm_joint_position for s in self._states], axis=0)
        target_delta = np.abs(curr - self._target_pose)
        in_zone = np.all(target_delta <= self._reward_thr)
        if in_zone:
            self._success_hold_counter += 1
            return 1.0
        self._success_hold_counter = 0
        if self.config.use_dense_reward:
            return float(np.exp(-5.0 * np.sum(np.square(target_delta))))
        return 0.0

    def reset(self, joint_reset=False, seed=None, options=None):
        if self.config.is_dummy:
            return self._get_observation(), {}
        self._success_hold_counter = 0
        self.go_to_rest(joint_reset=joint_reset)
        self._num_steps = 0
        self._sync_states()
        return self._get_observation(), {}

    def go_to_rest(self, joint_reset=False):
        if self.config.enable_random_reset:
            reset_pose = self._reset_pose.copy()
            reset_pose += np.random.uniform(-0.05, 0.05, size=(PIPER_DUAL_ARM_NUM, PIPER_ARM_JOINT_DIM))
        else:
            reset_pose = self._reset_pose.copy()
        if not self.config.is_dummy:
            self._left_controller.reset_joint(reset_pose[0])
            self._right_controller.reset_joint(reset_pose[1])

    def _init_action_obs_spaces(self):
        self.action_space = gym.spaces.Box(
            low=np.ones(PIPER_DUAL_ACTION_DIM, dtype=np.float32) * -1,
            high=np.ones(PIPER_DUAL_ACTION_DIM, dtype=np.float32),
        )
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "joint_position": gym.spaces.Box(-np.inf, np.inf, shape=(PIPER_DUAL_JOINT_DIM,)),
                        "joint_velocity": gym.spaces.Box(-np.inf, np.inf, shape=(PIPER_DUAL_JOINT_DIM,)),
                        "gripper_position": gym.spaces.Box(-np.inf, np.inf, shape=(PIPER_DUAL_ARM_NUM,)),
                    }
                ),
                "frames": gym.spaces.Dict(
                    {
                        f"wrist_{k + 1}": gym.spaces.Box(
                            0, 255, shape=(128, 128, 3), dtype=np.uint8
                        )
                        for k in range(len(self.config.camera_serials))
                    }
                ),
            }
        )
        self._base_observation_space = copy.deepcopy(self.observation_space)

    def _open_cameras(self):
        self._cameras: list[Camera] = []
        if self.config.camera_serials is None:
            return
        for i, serial in enumerate(self.config.camera_serials):
            info = CameraInfo(name=f"wrist_{i + 1}", serial_number=str(serial))
            camera = Camera(info)
            if not self.config.is_dummy:
                camera.open()
            self._cameras.append(camera)

    def _close_cameras(self):
        for camera in self._cameras:
            camera.close()
        self._cameras = []

    def _get_camera_frames(self) -> dict:
        frames = {}
        display_frames = {}
        for camera in self._cameras:
            try:
                frame = camera.get_frame()
                h, w, _ = frame.shape
                crop_size = min(h, w)
                start_x = (w - crop_size) // 2
                start_y = (h - crop_size) // 2
                cropped = frame[start_y : start_y + crop_size, start_x : start_x + crop_size]
                reshape_size = self.observation_space["frames"][camera._camera_info.name].shape[:2][::-1]
                resized = cv2.resize(cropped, reshape_size)
                frames[camera._camera_info.name] = resized[..., ::-1]
                display_frames[camera._camera_info.name] = resized
            except queue.Empty:
                self._logger.warning(f"Camera {camera._camera_info.name} failure. Reconnecting...")
                time.sleep(2)
                self._close_cameras()
                self._open_cameras()
                return self._get_camera_frames()
        self.camera_player.put_frame(display_frames)
        return frames

    def _get_observation(self) -> dict:
        if self.config.is_dummy:
            return self._base_observation_space.sample()
        frames = self._get_camera_frames()
        joint_position = np.concatenate([s.arm_joint_position for s in self._states], axis=0)
        joint_velocity = np.concatenate([s.arm_joint_velocity for s in self._states], axis=0)
        gripper_position = np.array([s.gripper_position for s in self._states], dtype=np.float64)
        return {
            "state": {
                "joint_position": joint_position,
                "joint_velocity": joint_velocity,
                "gripper_position": gripper_position,
            },
            "frames": frames,
        }
