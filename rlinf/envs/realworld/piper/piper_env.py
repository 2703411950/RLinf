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
from typing import Optional

import cv2
import gymnasium as gym
import numpy as np

from rlinf.envs.realworld.common.camera import Camera, CameraInfo
from rlinf.envs.realworld.common.video_player import VideoPlayer
from rlinf.scheduler import WorkerInfo
from rlinf.utils.logging import get_logger

from .piper_robot_state import PiperRobotState

# NOTE: Currently PiperEnv is configured for joint-space control since SDK naturally provides joint methods.
# Integrating Cartesian space logic will require IK/FK module in future.
@dataclass
class PiperRobotConfig:
    can_name: Optional[str] = None
    camera_serials: Optional[list[str]] = None
    enable_camera_player: bool = True

    is_dummy: bool = False
    use_dense_reward: bool = False
    step_frequency: float = 10.0

    target_joint_pose: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    reset_joint_pose: np.ndarray = field(
        default_factory=lambda: np.zeros(6)
    )
    max_num_steps: int = 100
    reward_threshold: np.ndarray = field(default_factory=lambda: np.zeros(6))
    
    # action scale for joint deltas
    action_scale: np.ndarray = field(
        default_factory=lambda: np.ones(7) * 0.05
    )
    enable_random_reset: bool = False

    joint_limit_min: np.ndarray = field(default_factory=lambda: -np.ones(6) * 3.14)
    joint_limit_max: np.ndarray = field(default_factory=lambda: np.ones(6) * 3.14)
    
    binary_gripper_threshold: float = 0.5
    enable_gripper_penalty: bool = True
    gripper_penalty: float = 0.1
    success_hold_steps: int = 1


class PiperEnv(gym.Env):
    """Piper robot arm environment."""

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
        self._reset_pose = self.config.reset_joint_pose.copy()
        
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

        self._controller = PiperController.launch_controller(
            can_name=self.config.can_name,
            env_idx=self.env_idx,
            node_rank=self.node_rank,
            worker_rank=self.env_worker_rank,
        )

    def step(self, action: np.ndarray):
        """Take a step in the environment.
        Input action should be 7D continuous [-1, 1], representing delta joint commands + gripper.
        """
        start_time = time.time()

        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        if not self.config.is_dummy:
            # 6 joints + 1 gripper
            scaled_action = action * self.config.action_scale
            next_joints = self._piper_state.arm_joint_position + scaled_action[:6]
            next_joints = np.clip(next_joints, self.config.joint_limit_min, self.config.joint_limit_max)
            
            # Combine to command array
            command = np.zeros(7)
            command[:6] = next_joints
            command[6] = scaled_action[6] # gripper
            
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

    def _calc_step_reward(self, observation: dict) -> float:
        """Compute task completion reward. Currently based on joint error for simplicity in template."""
        if not self.config.is_dummy:
            # calculate distance in joint space
            target_delta = np.abs(self._piper_state.arm_joint_position - self.config.target_joint_pose)
            
            # success threshold checks
            is_in_target_zone = np.all(target_delta <= self.config.reward_threshold)

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
            reset_pose += np.random.uniform(-0.05, 0.05, (6,))
        else:
            reset_pose = self._reset_pose.copy()

        self._controller.reset_joint(reset_pose)

    def _init_action_obs_spaces(self):
        # 6 joints + 1 gripper
        self.action_space = gym.spaces.Box(
            np.ones(7, dtype=np.float32) * -1,
            np.ones(7, dtype=np.float32),
        )

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "joint_position": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                        "joint_velocity": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                        "gripper_position": gym.spaces.Box(-np.inf, np.inf, shape=(1,)),
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
        camera_infos = [
            CameraInfo(name=f"wrist_{i + 1}", serial_number=n)
            for i, n in enumerate(self.config.camera_serials)
        ]
        from rlinf.envs.realworld.piper.piper_realsense_camera import PiperRealsenseCamera
        for info in camera_infos:
            camera = PiperRealsenseCamera(info)
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
                # Custom camera implements aligned RGB+Depth output format via dict
                cam_data = camera.get_frame()
                frame = cam_data['color_img']  # we only use color in the standardized state dict
                
                # Assume standard resizing to observation space definitions
                h, w, _ = frame.shape
                crop_size = min(h, w)
                start_x = (w - crop_size) // 2
                start_y = (h - crop_size) // 2
                cropped = frame[start_y:start_y+crop_size, start_x:start_x+crop_size]
                
                reshape_size = self.observation_space["frames"][camera._camera_info.name].shape[:2][::-1]
                resized = cv2.resize(cropped, reshape_size)
                
                frames[camera._camera_info.name] = resized[..., ::-1]  # RGB to BGR
                display_frames[camera._camera_info.name] = resized
            except Exception as e:
                self._logger.warning(f"Camera {camera._camera_info.name} failure. Reconnecting...")
                time.sleep(2)
                self._close_cameras()
                self._open_cameras()
                return self._get_camera_frames()

        self.camera_player.put_frame(display_frames)
        return frames

    def _move_action(self, action: np.ndarray):
        if not self.config.is_dummy:
            self._controller.move_arm(action).wait()
        else:
            self._logger.debug(f"Dummy move: {action}")

    def _get_observation(self) -> dict:
        if not self.config.is_dummy:
            frames = self._get_camera_frames()
            state = {
                "joint_position": self._piper_state.arm_joint_position,
                "joint_velocity": self._piper_state.arm_joint_velocity,
                "gripper_position": np.array([self._piper_state.gripper_position]),
            }
            return {
                "state": state,
                "frames": frames,
            }
        else:
            return self._base_observation_space.sample()
