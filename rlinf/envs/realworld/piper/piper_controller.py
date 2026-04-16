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

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from piper_sdk import C_PiperInterface_V2, LogLevel

from rlinf.scheduler import Cluster, NodePlacementStrategy, Worker
from rlinf.utils.logging import get_logger

from .piper_robot_state import PiperRobotState


def _enum_to_name(x: Any) -> Any:
    return getattr(x, "name", x)

def _obj_bool_dict(obj: Any) -> Optional[Dict[str, bool]]:
    if obj is None:
        return None
    d = getattr(obj, "__dict__", None)
    if isinstance(d, dict):
        out = {}
        for k, v in d.items():
            if isinstance(v, bool):
                out[k] = v
        return out
    return None

class PiperController(Worker):
    """Piper robot arm controller running as a Ray Worker."""

    @staticmethod
    def launch_controller(
        can_name: str | list[str] | tuple[str, ...],
        env_idx: int = 0,
        node_rank: int = 0,
        worker_rank: int = 0,
        io_unit_mode: Literal["radian_norm", "evo_rl_unit"] = "radian_norm",
        dry_run_commands: bool = True,
        speed_ratio: int = 5,
        high_follow: bool = True,
        mode_refresh_interval_s: float = 1.0,
        reset_duration_s: float = 3.0,
        reset_interpolation_steps: int = 90,
    ):
        """Launch a PiperController on the specified worker's node."""
        cluster = Cluster()
        placement = NodePlacementStrategy(node_ranks=[node_rank])
        return PiperController.create_group(
            can_name,
            io_unit_mode=io_unit_mode,
            dry_run_commands=dry_run_commands,
            speed_ratio=speed_ratio,
            high_follow=high_follow,
            mode_refresh_interval_s=mode_refresh_interval_s,
            reset_duration_s=reset_duration_s,
            reset_interpolation_steps=reset_interpolation_steps,
        ).launch(
            cluster=cluster,
            placement_strategy=placement,
            name=f"PiperController-{worker_rank}-{env_idx}",
        )

    def __init__(
        self,
        can_name: str | list[str] | tuple[str, ...],
        enable_for_sampling: bool = True,
        set_can_ctrl_mode: bool = True,
        io_unit_mode: Literal["radian_norm", "evo_rl_unit"] = "radian_norm",
        dry_run_commands: bool = True,
        speed_ratio: int = 5,
        high_follow: bool = True,
        mode_refresh_interval_s: float = 1.0,
        reset_duration_s: float = 3.0,
        reset_interpolation_steps: int = 90,
    ):
        """Initialize the Piper robot arm controller worker."""
        super().__init__()
        self._logger = get_logger()
        self._can_names = self._normalize_can_names(can_name)
        self._can_name = ",".join(self._can_names)
        self._num_arms = len(self._can_names)
        self._io_unit_mode = io_unit_mode
        self._dry_run_commands = dry_run_commands
        self._speed_ratio = int(speed_ratio)
        self._high_follow = bool(high_follow)
        self._mode_refresh_interval_s = float(mode_refresh_interval_s)
        self._reset_duration_s = float(reset_duration_s)
        self._reset_interpolation_steps = int(reset_interpolation_steps)
        self._state = PiperRobotState()
        self.drivers: list[C_PiperInterface_V2] = []
        self._connected = False
        self._enabled = False
        self._last_mode_refresh_t = 0.0

        for can in self._can_names:
            driver = C_PiperInterface_V2(
                can_name=can,
                judge_flag=True,
                can_auto_init=True,
                logger_level=LogLevel.WARNING,
            )
            driver.ConnectPort(can_init=False, piper_init=True, start_thread=True)
            self.drivers.append(driver)
        self._connected = True

        if enable_for_sampling:
            for driver in self.drivers:
                driver.EnableArm(motor_num=7, enable_flag=0x02)
            self._enabled = True

        if set_can_ctrl_mode:
            self._send_motion_mode()

        status = [
            {
                "can_name": can,
                "connected": drv.get_connect_status(),
                "is_ok": drv.isOk(),
                "can_fps": drv.GetCanFps(),
            }
            for can, drv in zip(self._can_names, self.drivers, strict=True)
        ]
        self.log_info(
            f"[Piper] num_arms={self._num_arms}, can_names={self._can_names}, "
            f"io_unit_mode={self._io_unit_mode}, dry_run_commands={self._dry_run_commands}, "
            f"speed_ratio={self._speed_ratio}, high_follow={self._high_follow}, "
            f"mode_refresh_interval_s={self._mode_refresh_interval_s}, "
            f"reset_duration_s={self._reset_duration_s}, "
            f"reset_interpolation_steps={self._reset_interpolation_steps}, status={status}"
        )

    def _send_motion_mode(self) -> None:
        """Send Piper motion mode, matching Evo-RL's follower configuration."""
        mit_mode = 0xAD if self._high_follow else 0x00
        for driver in self.drivers:
            driver.MotionCtrl_2(0x01, 0x01, self._speed_ratio, mit_mode)
        self._last_mode_refresh_t = time.monotonic()

    def _refresh_motion_mode_if_needed(self) -> None:
        interval_s = self._mode_refresh_interval_s
        if interval_s <= 0:
            return
        now = time.monotonic()
        if now - self._last_mode_refresh_t >= interval_s:
            self._send_motion_mode()

    @staticmethod
    def _normalize_can_names(
        can_name: str | list[str] | tuple[str, ...],
    ) -> list[str]:
        """Normalize can_name input to one/two CAN names.

        Accepts:
        - "can0"
        - "can0,can1"
        - ["can0", "can1"]
        """
        if isinstance(can_name, str):
            can_names = [x.strip() for x in can_name.split(",") if x.strip()]
        else:
            can_names = [str(x).strip() for x in can_name if str(x).strip()]
        if not can_names:
            raise ValueError("can_name is empty")
        if len(can_names) > 2:
            raise ValueError(
                f"PiperController supports up to 2 arms, got {len(can_names)} CAN names: {can_names}"
            )
        return can_names


    def is_robot_up(self) -> bool:
        """Check if connection is ready."""
        return self._connected and all(driver.isOk() for driver in self.drivers)

    def get_state(self) -> PiperRobotState:
        """Get the current state of the Piper robot in radians and meters."""
        DEG2RAD = math.pi / 180.0

        all_joint_rad = []
        all_joint_vel_rads = []
        all_gripper_pos = []
        all_gripper_effort = []
        for driver in self.drivers:
            jmsg = driver.GetArmJointMsgs()
            j = jmsg.joint_state
            # driver provides values in 0.001 deg
            joint_raw_mdeg = [
                j.joint_1,
                j.joint_2,
                j.joint_3,
                j.joint_4,
                j.joint_5,
                j.joint_6,
            ]
            if self._io_unit_mode == "evo_rl_unit":
                all_joint_rad.append(np.array([v * 1e-3 for v in joint_raw_mdeg], dtype=np.float64))
            else:
                all_joint_rad.append(np.array([v * 1e-3 * DEG2RAD for v in joint_raw_mdeg], dtype=np.float64))

            hmsg = driver.GetArmHighSpdInfoMsgs()
            joint_vel_rads = []
            for i in range(1, 7):
                m = getattr(hmsg, f"motor_{i}")
                # 0.001 rad/s -> rad/s
                joint_vel_rads.append(m.motor_speed * 1e-3)
            all_joint_vel_rads.append(np.array(joint_vel_rads))

            gmsg = driver.GetArmGripperMsgs()
            g = gmsg.gripper_state
            if self._io_unit_mode == "evo_rl_unit":
                # Match Evo-RL/LeRobot Piper follower: abs(milli_to_unit(grippers_angle)).
                all_gripper_pos.append(abs(g.grippers_angle * 1e-3))
            else:
                # Match the control_your_robot normalized gripper ratio path.
                all_gripper_pos.append(g.grippers_angle * 1e-3 / 70.0)
            all_gripper_effort.append(g.grippers_effort * 1e-3)

        self._state.arm_joint_position = np.concatenate(all_joint_rad, axis=0)
        self._state.arm_joint_velocity = np.concatenate(all_joint_vel_rads, axis=0)
        
        self._state.gripper_position = np.array(all_gripper_pos)
        self._state.gripper_effort = np.array(all_gripper_effort)

        # Note: we need Forward Kinematics (FK) integration here if target task needs tcp_pose
        # Alternatively, if Piper SDK starts providing task-space (TCP) pos, we'd fill tcp_pose here.
        # Currently leaving tcp_pose as exactly joint-driven or placeholder unless FK installed.
        # self._state.tcp_pose = ... 
        return self._state

    def move_arm(self, action: np.ndarray):
        """
        Execute an action. Here `action` is interpreted as a 7-DoF joint command:
        [j0, j1, j2, j3, j4, j5, gripper].
        Units for joints are radians, and gripper is a continuous value [0, 1].
        """
        assert len(action) == 7, f"Piper action requires 7 dims (6 joints + gripper), got {len(action)}"
        action = np.asarray(action, dtype=np.float64)
        self._execute_single_arm(self.drivers[0], action, arm_idx=0, sleep_after=False)
        time.sleep(1/30)

    def move_arm_dual(self, action: np.ndarray):
        """Execute dual-arm action: 14 dims = left 7 + right 7."""
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        assert action.shape == (14,), (
            f"Piper dual-arm action requires 14 dims (left7+right7), got {action.shape}"
        )
        if len(self.drivers) < 2:
            raise RuntimeError(
                "Dual-arm action provided, but only one CAN driver is configured. "
                "Set can_name to 'can0,can1' (or equivalent two CAN names)."
            )
        self._execute_single_arm(self.drivers[0], action[:7], arm_idx=0, sleep_after=False)
        self._execute_single_arm(self.drivers[1], action[7:], arm_idx=1, sleep_after=False)
        # time.sleep(1/30)

    def _execute_single_arm(
        self,
        driver: C_PiperInterface_V2,
        action: np.ndarray,
        arm_idx: int,
        sleep_after: bool = True,
    ):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        action_str = np.array2string(
            action,
            precision=4,
            floatmode="fixed",
        )
        # self.log_info(
        #     f"piper execute action (arm={arm_idx}, can={self._can_names[arm_idx]}): {action_str}"
        # )

        start_time = time.time()
        while not driver.EnablePiper():
            time.sleep(0.01)
            if time.time() - start_time > 3.0:
                raise RuntimeError(
                    f"[Piper arm={arm_idx}] EnablePiper timeout (>3s). "
                    "Please check e-stop, motor enable state, and CAN communication."
                )

        if self._io_unit_mode == "evo_rl_unit":
            # Match Evo-RL PiperFollower.send_action():
            # upstream values are already in SDK upper-layer units (deg / gripper unit),
            # and are converted to milli-units only at the final SDK boundary.
            joint_0 = round(action[0] * 1000.0)
            joint_1 = round(action[1] * 1000.0)
            joint_2 = round(action[2] * 1000.0)
            joint_3 = round(action[3] * 1000.0)
            joint_4 = round(action[4] * 1000.0)
            joint_5 = round(action[5] * 1000.0)
            joint_6 = round(action[6] * 1000.0)
        else:
            # conversion from rad -> 0.001 deg
            factor = 57295.7795
            joint_0 = round(action[0] * factor)
            joint_1 = round(action[1] * factor)
            joint_2 = round(action[2] * factor)
            joint_3 = round(action[3] * factor)
            joint_4 = round(action[4] * factor)
            joint_5 = round(action[5] * factor)
            joint_6 = round(action[6] * 70 * 1000)

        self._refresh_motion_mode_if_needed()

        if self._dry_run_commands:
            return

        driver.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
        driver.GripperCtrl(joint_6, 1000, 0x01, 0)
        if sleep_after:
            time.sleep(1/30)

    def _current_action_vector(self) -> np.ndarray:
        """Read current dual-arm joint+gripper state in the controller's action units."""
        state = self.get_state()
        current = np.zeros(14, dtype=np.float64)
        current[:6] = np.asarray(state.arm_joint_position[:6], dtype=np.float64)
        current[6] = float(state.gripper_position[0])
        if self._num_arms >= 2:
            current[7:13] = np.asarray(state.arm_joint_position[6:12], dtype=np.float64)
            current[13] = float(state.gripper_position[1])
        return current

    def _expand_reset_target(self, reset: np.ndarray) -> np.ndarray:
        """Normalize reset input to a full target vector while preserving omitted grippers."""
        current = self._current_action_vector()
        dim = reset.shape[0]
        target = current.copy()
        if dim == 6:
            target[:6] = reset[:6]
            return target[:7]
        if dim == 7:
            target[:7] = reset[:7]
            return target[:7]
        if dim == 12:
            target[:6] = reset[:6]
            target[7:13] = reset[6:12]
            return target
        if dim == 14:
            target[:] = reset
            return target
        raise ValueError(
            "Reset pos must be one of [6, 7, 12, 14] dims "
            f"(got {dim})."
        )

    def _interpolate_reset(self, target: np.ndarray) -> None:
        """Move to reset pose with small steps instead of a single large jump."""
        start = self._current_action_vector()
        if target.shape == (7,):
            start = start[:7]

        steps = max(self._reset_interpolation_steps, 1)
        duration_s = max(self._reset_duration_s, 0.0)
        sleep_s = duration_s / steps if steps > 0 else 0.0

        for idx in range(1, steps + 1):
            alpha = idx / steps
            cmd = start + (target - start) * alpha
            tick_start = time.perf_counter()
            if target.shape == (7,):
                self._execute_single_arm(self.drivers[0], cmd, arm_idx=0, sleep_after=False)
            else:
                self._execute_single_arm(self.drivers[0], cmd[:7], arm_idx=0, sleep_after=False)
                self._execute_single_arm(self.drivers[1], cmd[7:], arm_idx=1, sleep_after=False)
            if sleep_s > 0:
                elapsed = time.perf_counter() - tick_start
                time.sleep(max(0.0, sleep_s - elapsed))

    def clear_errors(self):
        # Implementation for clearing driver errors can be mapped here.
        pass

    def reset_joint(self, reset_pos: list[float] | np.ndarray):
        """Unified reset for single/dual arm.

        Supported input dimensions:
        - Single arm: 6 joints or 7 (6 joints + gripper)
        - Dual arm: 12 joints or 14 (left7 + right7)
        """
        reset = np.asarray(reset_pos, dtype=np.float64).reshape(-1)
        target = self._expand_reset_target(reset)
        self._interpolate_reset(target)

    # def reset_joint_dual(self, reset_pos: list[float] | np.ndarray):
    #     """Backward-compatible alias to the unified ``reset_joint``."""
    #     self.reset_joint(reset_pos)
    
    def disconnect(self):
        """Disconnect driver."""
        if self._connected:
            for driver in self.drivers:
                driver.DisconnectPort()
            self._connected = False
            self.log_info("[Piper] Disconnected port.")
