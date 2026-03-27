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
class PiperDualHWInfo(HardwareInfo):
    """Hardware information for a dual-arm Piper robotic system."""

    config: "PiperDualConfig"


@Hardware.register()
class PiperDualRobot(Hardware):
    """Hardware policy for dual-arm Piper robotic systems."""

    HW_TYPE = "PiperDual"

    @classmethod
    def enumerate(
        cls, node_rank: int, configs: Optional[list["PiperDualConfig"]] = None
    ) -> Optional[HardwareResource]:
        assert configs is not None, (
            "PiperDual hardware requires explicit configurations for two CAN ports "
            "and optional camera serials."
        )
        robot_configs: list["PiperDualConfig"] = []
        for config in configs:
            if isinstance(config, PiperDualConfig) and config.node_rank == node_rank:
                robot_configs.append(config)

        if not robot_configs:
            return None

        dual_infos = []
        cameras = cls.enumerate_cameras()
        for config in robot_configs:
            if config.camera_serials is None:
                config.camera_serials = list(cameras)

            dual_infos.append(
                PiperDualHWInfo(
                    type=cls.HW_TYPE,
                    model=cls.HW_TYPE,
                    config=config,
                )
            )

            if config.disable_validate:
                continue

            try:
                importlib.import_module("pyrealsense2")
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    "pyrealsense2 is required for PiperDual camera serials check, "
                    f"but it is not installed on node rank {node_rank}."
                )
            if not cameras:
                warnings.warn(
                    f"No Realsense cameras are connected to node rank {node_rank}."
                )
            else:
                for serial in config.camera_serials:
                    if serial not in cameras:
                        raise ValueError(
                            f"Camera with serial {serial} for PiperDual is not connected "
                            f"to node rank {node_rank}. Available cameras: {cameras}."
                        )

        return HardwareResource(type=cls.HW_TYPE, infos=dual_infos)

    @classmethod
    def enumerate_cameras(cls):
        cameras: set[str] = set()
        try:
            import pyrealsense2 as rs
        except ImportError:
            return cameras
        for device in rs.context().devices:
            cameras.add(device.get_info(rs.camera_info.serial_number))
        return cameras


@NodeHardwareConfig.register_hardware_config(PiperDualRobot.HW_TYPE)
@dataclass
class PiperDualConfig(HardwareConfig):
    """Configuration for a dual-arm Piper robotic system."""

    can_names: list[str]
    """Two CAN bus port names for left/right Piper (e.g. ['can0', 'can1'])."""

    camera_serials: Optional[list[str]] = None
    """List of RealSense camera serial numbers associated with the robot setup."""

    disable_validate: bool = False
    """Whether to disable validation of camera serials and hardware connection."""

    def __post_init__(self):
        assert isinstance(self.node_rank, int), (
            f"'node_rank' in piper dual config must be an integer. But got {type(self.node_rank)}."
        )
        try:
            can_names = list(self.can_names)
        except TypeError as e:
            raise AssertionError(
                f"'can_names' must be an iterable with 2 entries, but got: {self.can_names}"
            ) from e
        assert len(can_names) == 2, (
            f"'can_names' must contain exactly 2 entries, but got: {can_names}"
        )
        self.can_names = [str(x) for x in can_names]
        if self.camera_serials:
            self.camera_serials = list(self.camera_serials)
