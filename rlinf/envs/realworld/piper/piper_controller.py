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

# "quantiles" | "min_max" — override with env PIPER_ACTION_UNNORMALIZE_MODE=min_max
_DEFAULT_UNNORM = os.environ.get("PIPER_ACTION_UNNORMALIZE_MODE", "quantiles").strip().lower()
PIPER_ACTION_UNNORMALIZE_MODE: Literal["quantiles", "min_max"] = (
    "min_max" if _DEFAULT_UNNORM == "min_max" else "quantiles"
)


def _resolve_norm_stats_json_path() -> Path:
    """Resolve OpenPI ``norm_stats.json`` (under e.g. ``<ckpt>/physical-intelligence/libero/``)."""
    explicit = os.environ.get("PIPER_NORM_STATS_JSON", "").strip()
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"PIPER_NORM_STATS_JSON is not a file: {p}")
        return p.resolve()
    for env_key in ("PIPER_MODEL_PATH", "OPENPI_MODEL_PATH"):
        base = os.environ.get(env_key, "").strip()
        if not base:
            continue
        p = (
            Path(base).expanduser().resolve()
            / "physical-intelligence"
            / "libero"
            / "norm_stats.json"
        )
        if p.is_file():
            return p
    raise FileNotFoundError(
        "Set PIPER_NORM_STATS_JSON to the OpenPI norm_stats.json path "
        "(e.g. <checkpoint>/physical-intelligence/libero/norm_stats.json), "
        "or set PIPER_MODEL_PATH / OPENPI_MODEL_PATH to the checkpoint root "
        "that contains that file."
    )


def load_openpi_action_norm_stats(
    path: Path, action_dim: int
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Load ``actions`` q01/q99 and optional min/max from OpenPI ``norm_stats.json``.

    OpenPI schema (``norm_stats.actions``) always includes ``q01`` / ``q99``; ``min`` / ``max``
    are optional. If absent, MIN_MAX unnormalization uses ``q01``/``q99`` as endpoints
    (same affine map as QUANTILES).
    """
    raw = json.loads(path.read_text())
    ns = raw.get("norm_stats", raw)
    if "actions" not in ns:
        raise KeyError(f"{path}: missing norm_stats.actions; keys={list(ns.keys())!r}")
    act = ns["actions"]
    for key in ("q01", "q99"):
        if key not in act:
            raise KeyError(f"{path}: norm_stats.actions missing {key!r}")

    def _take_vec(name: str) -> np.ndarray:
        v = np.asarray(act[name], dtype=np.float64).reshape(-1)
        if v.size < action_dim:
            raise ValueError(
                f"{path}: actions.{name} len {v.size} < action_dim {action_dim}"
            )
        return v[:action_dim].copy()

    q01 = _take_vec("q01")
    q99 = _take_vec("q99")
    amin: Optional[np.ndarray] = None
    amax: Optional[np.ndarray] = None
    if "min" in act and "max" in act:
        amin = _take_vec("min")
        amax = _take_vec("max")
    return q01, q99, amin, amax


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
        can_name: str,
        env_idx: int = 0,
        node_rank: int = 0,
        worker_rank: int = 0,
    ):
        """Launch a PiperController on the specified worker's node."""
        cluster = Cluster()
        placement = NodePlacementStrategy(node_ranks=[node_rank])
        return PiperController.create_group(can_name).launch(
            cluster=cluster,
            placement_strategy=placement,
            name=f"PiperController-{worker_rank}-{env_idx}",
        )

    def __init__(
        self, 
        can_name: str,
        enable_for_sampling: bool = True,
        set_can_ctrl_mode: bool = True
    ):
        """Initialize the Piper robot arm controller worker."""
        super().__init__()
        self._logger = get_logger()
        self._can_name = can_name
        self._state = PiperRobotState()
        
        self.driver = C_PiperInterface_V2(
            can_name=can_name,
            judge_flag=True,
            can_auto_init=True,
            logger_level=LogLevel.WARNING,
        )
        self._connected = False
        self._enabled = False

        self.driver.ConnectPort(can_init=False, piper_init=True, start_thread=True)
        self._connected = True

        if set_can_ctrl_mode:
            # 0x01 is CAN control mode, 0x01 is Joint position, 30 is max speed
            self.driver.ModeCtrl(ctrl_mode=0x01, move_mode=0x01, move_spd_rate_ctrl=30, is_mit_mode=0x00)

        if enable_for_sampling:
            self.driver.EnableArm(motor_num=7, enable_flag=0x02)
            self._enabled = True

        action_dim = int(os.environ.get("PIPER_ACTION_NORM_DIM", "7"))
        norm_path = _resolve_norm_stats_json_path()
        q01, q99, amin, amax = load_openpi_action_norm_stats(norm_path, action_dim)
        self._action_q01 = q01
        self._action_q99 = q99
        if amin is not None and amax is not None:
            self._action_min = amin
            self._action_max = amax
            self._min_max_from_file = True
        else:
            self._action_min = q01.copy()
            self._action_max = q99.copy()
            self._min_max_from_file = False

        self.log_info(f"[Piper] Connected: {self.driver.get_connect_status()}, isOk: {self.driver.isOk()}, CAN FPS: {self.driver.GetCanFps()}")
        self.log_info(
            f"[Piper] Loaded action norm stats from {norm_path} "
            f"(action_dim={action_dim}, min_max_uses_dataset_minmax={self._min_max_from_file})"
        )
        if not self._min_max_from_file and PIPER_ACTION_UNNORMALIZE_MODE == "min_max":
            self.log_info(
                "[Piper] norm_stats has no actions.min/max; MIN_MAX unnorm uses q01/q99 "
                "(same linear map as QUANTILES)."
            )

    def _unnormalize_action_quantiles(self, normalized_action: np.ndarray) -> np.ndarray:
        """Unnormalize action from [-1, 1] with OpenPI QUANTILES stats (q01/q99)."""
        denom = self._action_q99 - self._action_q01
        denom = np.where(np.abs(denom) < 1e-8, 1e-8, denom)
        return (normalized_action + 1.0) * denom / 2.0 + self._action_q01

    def _unnormalize_action_min_max(
        self, normalized_action: np.ndarray, eps: float = 1e-8
    ) -> np.ndarray:
        """Unnormalize action from [-1, 1] with MIN_MAX (inverse of LeRobot NormalizationMode.MIN_MAX).

        Uses ``actions.min`` / ``actions.max`` from ``norm_stats.json`` when present; otherwise
        uses ``q01`` / ``q99`` (same endpoints as quantile-based linear map).
        """
        a = np.asarray(normalized_action, dtype=np.float64)
        min_val = self._action_min
        max_val = self._action_max
        if a.shape[-1] != min_val.shape[0]:
            raise ValueError(
                f"action dim {a.shape[-1]} != norm min len {min_val.shape[0]}"
            )
        denom = max_val - min_val
        denom = np.where(denom == 0, eps, denom)
        return (a + 1.0) / 2.0 * denom + min_val

    def is_robot_up(self) -> bool:
        """Check if connection is ready."""
        return self._connected and self.driver.isOk()

    def get_state(self) -> PiperRobotState:
        """Get the current state of the Piper robot in radians and meters."""
        DEG2RAD = math.pi / 180.0

        jmsg = self.driver.GetArmJointMsgs()  
        j = jmsg.joint_state  
        # driver provides values in 0.001 deg
        joint_raw_mdeg = [j.joint_1, j.joint_2, j.joint_3, j.joint_4, j.joint_5, j.joint_6]
        joint_rad = np.array([v * 1e-3 * DEG2RAD for v in joint_raw_mdeg])

        hmsg = self.driver.GetArmHighSpdInfoMsgs()  
        joint_vel_rads = []
        for i in range(1, 7):
            m = getattr(hmsg, f"motor_{i}")  
            # 0.001 rad/s -> rad/s
            joint_vel_rads.append(m.motor_speed * 1e-3)

        gmsg = self.driver.GetArmGripperMsgs()  
        g = gmsg.gripper_state

        stroke_mm = g.grippers_angle * 1e-3
        effort_Nm = g.grippers_effort * 1e-3 

        self._state.arm_joint_position = joint_rad
        self._state.arm_joint_velocity = np.array(joint_vel_rads)
        self._state.gripper_position = stroke_mm
        self._state.gripper_effort = effort_Nm

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
        self.log_info(f"action before unnormalize: {action}")
        if PIPER_ACTION_UNNORMALIZE_MODE == "min_max":
            action = self._unnormalize_action_min_max(action)
        else:
            action = self._unnormalize_action_quantiles(action)

        self.log_info(
            f"action after unnormalize ({PIPER_ACTION_UNNORMALIZE_MODE}): {action}"
        )
        
        start_time = time.time()
        while not self.driver.EnablePiper():
            # # Avoid hanging forever when arm cannot be enabled.
            # if time.time() - start_time > 3.0:
            #     raise RuntimeError(
            #         "[Piper] EnablePiper timeout (>3s). Please check e-stop, motor enable state, "
            #         "and CAN communication on robot controller."
            #     )
            time.sleep(0.01)
            
        # Hardcoded 1000 threshold mapping to CAN standard values
        self.driver.GripperCtrl(0, 1000, 0x01, 0)
        
        # conversion from rad -> 0.001 deg
        # 1 rad = 180 / pi deg = 57.2957795 deg
        # 1 rad -> 57295.7795 * (0.001 deg)
        factor = 57295.7795
            
        joint_0 = round(action[0] * factor)
        joint_1 = round(action[1] * factor)
        joint_2 = round(action[2] * factor)
        joint_3 = round(action[3] * factor)
        joint_4 = round(action[4] * factor)
        joint_5 = round(action[5] * factor)
        
        # Using snippet's formula directly: `round(action[6]*70*1000)`
        joint_6 = round(action[6] * 70 * 1000)
        
        self.driver.MotionCtrl_2(0x01, 0x01, 100, 0x00)
        # self.driver.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
        self.driver.GripperCtrl(abs(joint_6), 1000, 0x01, 0)
        
        # self.log_debug(f"Piper Move: args={[joint_0, joint_1, joint_2, joint_3, joint_4, joint_5]}")

    def clear_errors(self):
        # Implementation for clearing driver errors can be mapped here.
        pass

    def reset_joint(self, reset_pos: list[float]):
        """
        Reset joints to a specific position in radians.
        """
        assert len(reset_pos) == 6 or len(reset_pos) == 7, "Reset pos must be 6 joints (+ gripper optionally)."
        # Extract 6-dim joints, pad with default gripper 0 if needed
        action = np.zeros(7)
        action[:6] = reset_pos[:6]
        if len(reset_pos) == 7:
            action[6] = reset_pos[6]
            
        self.move_arm(action)
        time.sleep(1.0) # sleep for completion
    
    def disconnect(self):
        """Disconnect driver."""
        if self._connected:
            self.driver.DisconnectPort()
            self._connected = False
            self.log_info("[Piper] Disconnected port.")
