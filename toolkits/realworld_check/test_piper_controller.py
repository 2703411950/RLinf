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


import os
import time

import numpy as np

from rlinf.envs.realworld.piper.piper_controller import PiperController


def main():
    can_name = os.environ.get("PIPER_CAN_NAME", "can1")
    controller = PiperController.launch_controller(can_name=can_name, node_rank=0, env_idx=0, worker_rank=0)

    start_time = time.time()
    while not controller.is_robot_up().wait()[0]:
        time.sleep(0.5)
        if time.time() - start_time > 30:
            print(
                f"Waited {time.time() - start_time} seconds for Piper robot to be ready."
            )
            break
    while True:
        try:
            cmd_str = input("Please input cmd:")
            if cmd_str == "q":
                break
            elif cmd_str == "getpos":
                state = controller.get_state().wait()[0]
                print(state.arm_joint_position)
            elif cmd_str == "getgripper":
                state = controller.get_state().wait()[0]
                print(f"Gripper position: {state.gripper_position}, effort: {state.gripper_effort}")
            elif cmd_str == "getstate":
                state = controller.get_state().wait()[0]
                print(state.to_dict())
            elif cmd_str == "reset":
                controller.reset_joint([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).wait()
            else:
                print(f"Unknown cmd: {cmd_str}")
        except KeyboardInterrupt:
            break
        time.sleep(1.0)


if __name__ == "__main__":
    main()