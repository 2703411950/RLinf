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

import importlib
import warnings
from dataclasses import dataclass
from typing import Optional

from ..hardware import (
    Hardware,
    HardwareConfig,
    HardwareInfo,
    HardwareResource,
    NodeHardwareConfig,
)


@dataclass
class PiperHWInfo(HardwareInfo):
    """Hardware information for a Songling Piper robotic system."""

    config: "PiperConfig"


@Hardware.register()
class PiperRobot(Hardware):
    """Hardware policy for Piper robotic systems."""

    HW_TYPE = "Piper"

    @classmethod
    def enumerate(
        cls, node_rank: int, configs: Optional[list["PiperConfig"]] = None
    ) -> Optional[HardwareResource]:
        """Enumerate the Piper robot resources on a node.

        Args:
            node_rank: The rank of the node being enumerated.
            configs: The configurations for the hardware on a node.

        Returns:
            Optional[HardwareResource]: An object representing the hardware resources. None if no hardware is found.
        """
        assert configs is not None, (
            "Piper robot hardware requires explicit configurations for CAN port and camera serials for its controller nodes."
        )
        robot_configs: list["PiperConfig"] = []
        for config in configs:
            if isinstance(config, PiperConfig) and config.node_rank == node_rank:
                robot_configs.append(config)

        if robot_configs:
            piper_infos = []
            cameras = cls.enumerate_cameras()

            for config in robot_configs:
                # Use auto detected cameras if not provided
                if config.camera_serials is None:
                    config.camera_serials = list(cameras)

                piper_infos.append(
                    PiperHWInfo(
                        type=cls.HW_TYPE,
                        model=cls.HW_TYPE,
                        config=config,
                    )
                )

                if config.disable_validate:
                    continue

                # Validate camera serials
                try:
                    importlib.import_module("pyrealsense2")
                except ModuleNotFoundError:
                    raise ModuleNotFoundError(
                        f"pyrealsense2 is required for Piper robot camera serials check, but it is not installed on the node with rank {node_rank}."
                    )
                if not cameras:
                    warnings.warn(
                        f"No Realsense cameras are connected to node rank {node_rank}."
                    )
                else:
                    for serial in config.camera_serials:
                        if serial not in cameras:
                            raise ValueError(
                                f"Camera with serial {serial} for Piper robot at is not connected to node rank {node_rank}. Available cameras are: {cameras}."
                            )

            return HardwareResource(type=cls.HW_TYPE, infos=piper_infos)
        return None

    @classmethod
    def enumerate_cameras(cls):
        """Enumerate connected camera serial numbers using pyrealsense2."""
        cameras: set[str] = set()
        try:
            import pyrealsense2 as rs
        except ImportError:
            return cameras
        for device in rs.context().devices:
            cameras.add(device.get_info(rs.camera_info.serial_number))
        return cameras


@NodeHardwareConfig.register_hardware_config(PiperRobot.HW_TYPE)
@dataclass
class PiperConfig(HardwareConfig):
    """Configuration for a Piper robotic system."""

    can_name: str
    """CAN bus port name parameter for piper_sdk (e.g. 'can0')."""

    camera_serials: Optional[list[str]] = None
    """List of RealSense camera serial numbers associated with the robot."""

    disable_validate: bool = False
    """Whether to disable validation of camera serials and hardware connection."""

    def __post_init__(self):
        """Post-initialization to validate the configuration."""
        assert isinstance(self.node_rank, int), (
            f"'node_rank' in piper config must be an integer. But got {type(self.node_rank)}."
        )

        if self.camera_serials:
            self.camera_serials = list(self.camera_serials)
