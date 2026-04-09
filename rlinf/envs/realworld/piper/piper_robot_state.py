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

from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class PiperRobotState:
    """Dataclass to hold the state of the Piper Robot."""
    
    # TCP Pose [x, y, z, rx, ry, rz]
    # Note: Depending on SDK support, we might need to compute this via FK if SDK doesn't provide TCP.
    tcp_pose: np.ndarray = field(
        default_factory=lambda: np.zeros(6)
    )
    
    # TCP Velocity
    tcp_vel: np.ndarray = field(default_factory=lambda: np.zeros(6))
    
    # Joint states for 6 DoF arm
    arm_joint_position: np.ndarray = field(
        default_factory=lambda: np.zeros(6)
    )
    
    arm_joint_velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(6)
    )
    
    # Gripper information (stroke_mm and effort)
    gripper_position: float = 0.0
    gripper_effort: float = 0.0
    gripper_open: bool = False

    def to_dict(self):
        """Convert the dataclass to a serializable dictionary."""
        return asdict(self)
