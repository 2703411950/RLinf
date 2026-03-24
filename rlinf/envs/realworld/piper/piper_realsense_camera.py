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

import time

import numpy as np

from rlinf.envs.realworld.common.camera import Camera, CameraInfo
from rlinf.utils.logging import get_logger

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


class PiperRealsenseCamera(Camera):
    """Specific Realsense wrapper provided by the user for Piper with aggressive temporal filters."""

    def __init__(self, camera_info: CameraInfo, width=640, height=480, fps=30, laser_on=True):
        super().__init__(camera_info)
        self.width = width
        self.height = height
        self.fps = fps
        self.laser_on = laser_on
        
        self.pipeline = None
        self.config = None
        self.align = None 
        self._logger = get_logger()
        
        self.is_connected = False
        self.depth_scale = 0.001

    def open(self):
        if rs is None:
            raise ImportError(
                "pyrealsense2 is not installed. Please install it using `pip install pyrealsense2`"
            )
        self._logger.info(f"[RealSense] Connecting to: {self._serial_number} ...")
        
        try:
            self.temporal_filter = rs.temporal_filter()
            self.temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.4) 
            self.temporal_filter.set_option(rs.option.filter_smooth_delta, 20)
            self.temporal_filter.set_option(rs.option.holes_fill, 7)
            
            self.pipeline = rs.pipeline()
            self.config = rs.config()

            if self._serial_number:
                self.config.enable_device(str(self._serial_number))
            
            # Using z16 and bgr8
            self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            
            profile = self.pipeline.start(self.config)
            device = profile.get_device()
            depth_sensor = device.first_depth_sensor()
            rgb_sensor = device.first_color_sensor()

            # --- Laser control logic ---
            if depth_sensor.supports(rs.option.emitter_enabled):
                if self.laser_on:
                    self._logger.info(f"   🔦 [{self._serial_number}] Laser ON")
                    depth_sensor.set_option(rs.option.emitter_enabled, 1.0)
                    if depth_sensor.supports(rs.option.laser_power):
                        depth_sensor.set_option(rs.option.laser_power, 360) 
                else:
                    self._logger.info(f"   🌑 [{self._serial_number}] Laser OFF")
                    depth_sensor.set_option(rs.option.emitter_enabled, 0.0)

            # High Density preset
            if depth_sensor.supports(rs.option.visual_preset):
                depth_sensor.set_option(rs.option.visual_preset, 4)

            # 50Hz powerline frequency
            if rgb_sensor.supports(rs.option.power_line_frequency):
                rgb_sensor.set_option(rs.option.power_line_frequency, 1)

            self.depth_scale = depth_sensor.get_depth_scale()
            
            # Create RGB alignment object
            self.align = rs.align(rs.stream.color)
            
            self.is_connected = True
            self._logger.info(f"[RealSense] {self._serial_number} warming up...")
            for _ in range(30): # Warm up
                self.pipeline.wait_for_frames()
            self._logger.info(f"[RealSense] {self._serial_number} ready.")
            
        except Exception as e:
            self._logger.error(f"[RealSense] {self._serial_number} connection failed: {e}")
            self.is_connected = False
            raise e

    def get_frame(self) -> dict:
        """Returns the aligned RGB and Depth arrays."""
        if not self.is_connected:
            raise RuntimeError("Camera not initialized.")
        try:
            # wait 2000ms avoiding USB jitter timeout
            frames = self.pipeline.wait_for_frames(timeout_ms=2000)
            
            aligned_frames = self.align.process(frames)
            
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not aligned_depth_frame or not color_frame: 
                raise RuntimeError("Failed to retrieve frames.")
            
            aligned_depth_frame = self.temporal_filter.process(aligned_depth_frame)

            depth_image = np.array(aligned_depth_frame.get_data())
            color_image = np.array(color_frame.get_data())
            timestamp_s = frames.get_timestamp() / 1000.0

            return {
                'color_img': color_image,
                'depth_img': depth_image,
                'timestamp': timestamp_s
            }
        except Exception as e:
            raise RuntimeError(f"Error getting frame from RealSense: {e}")

    def close(self):
        if self.is_connected and self.pipeline:
            try: 
                self.pipeline.stop()
            except Exception: 
                pass
            self.is_connected = False
            self._logger.info(f"[RealSense] {self._serial_number} disconnected.")
