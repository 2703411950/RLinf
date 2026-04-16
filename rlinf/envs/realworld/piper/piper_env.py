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
from itertools import cycle
from typing import Any, Optional

import cv2
import gymnasium as gym
import numpy as np

from rlinf.envs.realworld.common.camera import Camera, CameraInfo
from rlinf.envs.realworld.common.video_player import VideoPlayer
from rlinf.scheduler import WorkerInfo
from rlinf.utils.logging import get_logger

from .piper_robot_state import PiperRobotState

# Piper 双臂动作统一为 14 维：
# - 左臂 7 维 + 右臂 7 维（每臂: 6 关节 + 1 夹爪）。
# - gym action_space 仍为 Box([-1,1]^14)（兼容旧策略与 checker）；step() 内 **不做数值映射**，
#   原样交给 ``PiperController``（缩放/限幅等在 controller 或上层策略侧完成）。
# NOTE: Currently PiperEnv is configured for joint-space control since SDK naturally provides joint methods.
# Integrating Cartesian space logic will require IK/FK module in future.

PIPER_NUM_ARMS = 2
PIPER_SINGLE_ARM_ACTION_DIM = 7
PIPER_SINGLE_ARM_JOINT_DIM = 6
PIPER_ACTION_DIM = PIPER_NUM_ARMS * PIPER_SINGLE_ARM_ACTION_DIM
PIPER_ARM_JOINT_DIM = PIPER_NUM_ARMS * PIPER_SINGLE_ARM_JOINT_DIM


@dataclass
class PiperRobotConfig:
    can_name: Optional[str] = None
    camera_serials: Optional[list[str]] = None
    enable_camera_player: bool = True

    is_dummy: bool = False
    use_dense_reward: bool = False
    step_frequency: float = 10.0
    io_unit_mode: str = "radian_norm"
    dry_run_commands: bool = True
    image_preprocess_mode: str = "legacy"
    speed_ratio: int = 5
    high_follow: bool = True
    mode_refresh_interval_s: float = 1.0
    reset_duration_s: float = 3.0
    reset_interpolation_steps: int = 90

    target_joint_pose: np.ndarray = field(
        default_factory=lambda: np.zeros(PIPER_ARM_JOINT_DIM, dtype=np.float64)
    )
    reset_joint_pose: np.ndarray = field(
        default_factory=lambda: np.zeros(PIPER_ARM_JOINT_DIM)
    )
    max_num_steps: int = 100
    reward_threshold: np.ndarray = field(default_factory=lambda: np.zeros(PIPER_ARM_JOINT_DIM))
    enable_random_reset: bool = False

    joint_limit_min: np.ndarray = field(
        default_factory=lambda: -np.ones(PIPER_ARM_JOINT_DIM) * 3.14
    )
    joint_limit_max: np.ndarray = field(
        default_factory=lambda: np.ones(PIPER_ARM_JOINT_DIM) * 3.14
    )
    
    binary_gripper_threshold: float = 0.5
    enable_gripper_penalty: bool = True
    gripper_penalty: float = 0.1
    success_hold_steps: int = 1


class PiperEnv(gym.Env):

    def __init__(
        self,
        config: PiperRobotConfig,
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

        self._piper_state = PiperRobotState()
        self._reset_pose = np.asarray(
            self.config.reset_joint_pose, dtype=np.float64
        ).reshape(-1)[:PIPER_ARM_JOINT_DIM].copy()
        
        self._num_steps = 0
        self._success_hold_counter = 0

        if not self.config.is_dummy:
            self._setup_hardware()

        # Init action and observation spaces
        assert (
            self.config.camera_serials is not None
            and len(self.config.camera_serials) > 0
        ), "At least one camera serial must be provided for PiperEnv."
        self._init_action_obs_spaces()

        if self.config.is_dummy:
            return

        # Wait for the robot to be ready
        start_time = time.time()
        while not self._controller.is_robot_up():
            time.sleep(0.5)
            if time.time() - start_time > 10:
                self._logger.warning("Waiting for Piper robot to be ready.")

        # Go to reset configuration
        self._controller.reset_joint(self._reset_pose)
        self._piper_state = self._controller.get_state().wait()[0]

        # Init cameras
        self._open_cameras()
        self.camera_player = VideoPlayer(self.config.enable_camera_player)

    def _setup_hardware(self):
        from .piper_controller import PiperController

        assert self.env_idx >= 0, "env_idx must be set for PiperEnv."

        if self.config.can_name is None:
            self.config.can_name = self.hardware_info.config.can_name
        if self.config.camera_serials is None:
            self.config.camera_serials = self.hardware_info.config.camera_serials
        if self.config.camera_serials is not None:
            self.config.camera_serials = [str(serial) for serial in self.config.camera_serials]

        self._controller = PiperController.launch_controller(
            can_name=self.config.can_name,
            env_idx=self.env_idx,
            node_rank=self.node_rank,
            worker_rank=self.env_worker_rank,
            io_unit_mode=self.config.io_unit_mode,
            dry_run_commands=self.config.dry_run_commands,
            speed_ratio=self.config.speed_ratio,
            high_follow=self.config.high_follow,
            mode_refresh_interval_s=self.config.mode_refresh_interval_s,
            reset_duration_s=self.config.reset_duration_s,
            reset_interpolation_steps=self.config.reset_interpolation_steps,
        )

    def _normalized_action_to_command(self, action: np.ndarray) -> np.ndarray:
        """Pass-through: no clip or rescale; ``move_arm`` owns all numerical handling."""
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        assert a.shape == (PIPER_ACTION_DIM,), (
            f"Piper action must be shape ({PIPER_ACTION_DIM},), got {a.shape}"
        )
        return a.copy()

    def step(self, action: np.ndarray):
        """执行一步。`action` 为 14 维，原样传入 controller（不做环境侧映射）。"""
        start_time = time.time()

        action = np.asarray(action, dtype=np.float64).reshape(-1)
        command = self._normalized_action_to_command(action)

        if not self.config.is_dummy:
            self._move_action(command)
        
        # Determine hold time
        self._num_steps += 1
        step_time = time.time() - start_time
        time.sleep(max(0, (1.0 / self.config.step_frequency) - step_time))

        if not self.config.is_dummy:
            self._piper_state = self._controller.get_state().wait()[0]
            
        observation = self._get_observation()
        reward = self._calc_step_reward(observation)

        terminated = (reward == 1.0) and (
            self._success_hold_counter >= self.config.success_hold_steps
        )
        truncated = self._num_steps >= self.config.max_num_steps
        
        return observation, reward, terminated, truncated, {}

    @property
    def num_steps(self):
        return self._num_steps

    def _target_joint_pose_vec(self) -> np.ndarray:
        joint_dim = self._get_joint_dim()
        t = np.asarray(self.config.target_joint_pose, dtype=np.float64).reshape(-1)
        if len(t) >= joint_dim:
            return t[:joint_dim].copy()
        if len(t) == 0:
            return np.zeros(joint_dim, dtype=np.float64)
        raise ValueError(
            f"target_joint_pose must have at least {joint_dim} elements, got {len(t)}"
        )

    def _reward_threshold_vec(self) -> np.ndarray:
        joint_dim = self._get_joint_dim()
        r = np.asarray(self.config.reward_threshold, dtype=np.float64).reshape(-1)
        if r.size == 0:
            return np.zeros(joint_dim, dtype=np.float64)
        if r.size >= joint_dim:
            return r[:joint_dim].copy()
        raise ValueError(
            f"reward_threshold must have at least {joint_dim} elements, got {r.size}"
        )

    def _calc_step_reward(self, observation: dict) -> float:
        """Compute task completion reward. Currently based on joint error for simplicity in template."""
        if not self.config.is_dummy:
            target = self._target_joint_pose_vec()
            thr = self._reward_threshold_vec()
            target_delta = np.abs(self._piper_state.arm_joint_position - target)
            
            # success threshold checks
            is_in_target_zone = np.all(target_delta <= thr)

            if is_in_target_zone:
                self._success_hold_counter += 1
                return 1.0
            else:
                self._success_hold_counter = 0
                if self.config.use_dense_reward:
                    return np.exp(-5.0 * np.sum(np.square(target_delta)))
                else:
                    return 0.0
        return 0.0

    def reset(self, joint_reset=False, seed=None, options=None):
        if self.config.is_dummy:
            return self._get_observation(), {}

        self._success_hold_counter = 0 
        self.go_to_rest(joint_reset)
        
        self._num_steps = 0
        self._piper_state = self._controller.get_state().wait()[0]
        return self._get_observation(), {}

    def go_to_rest(self, joint_reset=False):
        # Move back to initial pose
        if self.config.enable_random_reset:
            reset_pose = self._reset_pose.copy()
            reset_pose += np.random.uniform(-0.05, 0.05, reset_pose.shape)
        else:
            reset_pose = self._reset_pose.copy()


        self._controller.reset_joint(reset_pose).wait()
        # if hasattr(self._controller, "reset_joint_dual"):
        #     self._controller.reset_joint_dual(reset_pose).wait()
        # elif reset_pose.shape[0] == PIPER_SINGLE_ARM_JOINT_DIM:
        #     self._controller.reset_joint(reset_pose).wait()
        # else:
        #     # Backward compatibility: old controller only supports single-arm reset.
        #     self._logger.warning(
        #         "PiperController has no dual-arm reset API; fallback to first arm reset."
        #     )
        #     self._controller.reset_joint(reset_pose[:PIPER_SINGLE_ARM_JOINT_DIM]).wait()


    def _init_action_obs_spaces(self):
        frame_shape = (128, 128, 3)
        if self.config.image_preprocess_mode == "evorl":
            frame_shape = (480, 640, 3)
        # 14D：策略 [-1,1]^14（左臂 7 + 右臂 7）
        self.action_space = gym.spaces.Box(
            np.ones(PIPER_ACTION_DIM, dtype=np.float32) * -1,
            np.ones(PIPER_ACTION_DIM, dtype=np.float32),
        )

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "joint_position": gym.spaces.Box(
                            -np.inf, np.inf, shape=(PIPER_ARM_JOINT_DIM,)
                        ),
                        "joint_velocity": gym.spaces.Box(
                            -np.inf, np.inf, shape=(PIPER_ARM_JOINT_DIM,)
                        ),
                        "gripper_position": gym.spaces.Box(
                            -np.inf, np.inf, shape=(PIPER_NUM_ARMS,)
                        ),
                    }
                ),
                "frames": gym.spaces.Dict(
                    {
                        f"wrist_{k + 1}": gym.spaces.Box(
                            0, 255, shape=frame_shape, dtype=np.uint8
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
        camera_infos = [
            CameraInfo(name=f"wrist_{i + 1}", serial_number=str(n))
            for i, n in enumerate(self.config.camera_serials)
        ]
        for info in camera_infos:
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
                if self.config.image_preprocess_mode == "evorl":
                    rgb = np.asarray(frame)[..., ::-1].copy()  # camera returns BGR8
                    frames[camera._camera_info.name] = rgb
                    display_frames[camera._camera_info.name] = frame
                else:
                    h, w, _ = frame.shape
                    crop_size = min(h, w)
                    start_x = (w - crop_size) // 2
                    start_y = (h - crop_size) // 2
                    cropped = frame[start_y : start_y + crop_size, start_x : start_x + crop_size]

                    reshape_size = self.observation_space["frames"][
                        camera._camera_info.name
                    ].shape[:2][::-1]
                    resized = cv2.resize(cropped, reshape_size)

                    frames[camera._camera_info.name] = resized[..., ::-1].copy()  # BGR -> RGB
                    display_frames[camera._camera_info.name] = resized
            except queue.Empty:
                self._logger.warning(f"Camera {camera._camera_info.name} failure. Reconnecting...")
                time.sleep(2)
                self._close_cameras()
                self._open_cameras()
                return self._get_camera_frames()

        self.camera_player.put_frame(display_frames)
        return frames

    def _move_action(self, action: np.ndarray):
        if not self.config.is_dummy:
            if hasattr(self._controller, "move_arm_dual"):
                self._controller.move_arm_dual(action).wait()
            elif action.shape[0] == PIPER_SINGLE_ARM_ACTION_DIM:
                self._controller.move_arm(action).wait()
            else:
                # Backward compatibility: old controller only supports one arm.
                self._logger.warning(
                    "PiperController has no dual-arm API; fallback to first arm action."
                )
                self._controller.move_arm(action[:PIPER_SINGLE_ARM_ACTION_DIM]).wait()
        else:
            self._logger.debug(f"Dummy move: {action}")

    def _expand_or_truncate(self, vec: np.ndarray, target_dim: int) -> np.ndarray:
        """Adapt observed state vector to target dim by truncating or zero-padding."""
        v = np.asarray(vec, dtype=np.float64).reshape(-1)
        if v.size >= target_dim:
            return v[:target_dim].copy()
        out = np.zeros(target_dim, dtype=np.float64)
        out[: v.size] = v
        return out

    def _get_joint_dim(self) -> int:
        v = np.asarray(self._piper_state.arm_joint_position).reshape(-1)
        if v.size > 0:
            return v.size
        return PIPER_ARM_JOINT_DIM

    def _get_observation(self) -> dict:
        if not self.config.is_dummy:
            frames = self._get_camera_frames()
            joint_position = self._expand_or_truncate(
                self._piper_state.arm_joint_position, PIPER_ARM_JOINT_DIM
            )
            joint_velocity = self._expand_or_truncate(
                self._piper_state.arm_joint_velocity, PIPER_ARM_JOINT_DIM
            )
            gripper_position = self._expand_or_truncate(
                self._piper_state.gripper_position, PIPER_NUM_ARMS
            )
            state = {
                "joint_position": joint_position,
                "joint_velocity": joint_velocity,
                "gripper_position": gripper_position,
            }
            return {
                "state": state,
                "frames": frames,
            }
        else:
            return self._base_observation_space.sample()
