#!/usr/bin/env python3

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from piper_sdk import C_PiperInterface_V2, LogLevel


OUT_PATH = Path("/tmp/piper_state_evorl_rlinf_compare.json")
CAN_NAMES = ("can0", "can1")


def _make_driver(can_name: str) -> C_PiperInterface_V2:
    driver = C_PiperInterface_V2(
        can_name=can_name,
        judge_flag=True,
        can_auto_init=True,
        logger_level=LogLevel.WARNING,
    )
    driver.ConnectPort(can_init=False, piper_init=True, start_thread=True)
    return driver


def _sdk_sample(driver: C_PiperInterface_V2) -> dict:
    jmsg = driver.GetArmJointMsgs()
    gmsg = driver.GetArmGripperMsgs()
    j = getattr(jmsg, "joint_state", None)
    g = getattr(gmsg, "gripper_state", None)
    joint_raw = [
        int(getattr(j, "joint_1", 0)),
        int(getattr(j, "joint_2", 0)),
        int(getattr(j, "joint_3", 0)),
        int(getattr(j, "joint_4", 0)),
        int(getattr(j, "joint_5", 0)),
        int(getattr(j, "joint_6", 0)),
    ]
    gripper_raw = int(getattr(g, "grippers_angle", 0))
    return {
        "joint_raw_milli": joint_raw,
        "gripper_raw_milli": gripper_raw,
        "joint_timestamp": float(getattr(jmsg, "time_stamp", 0.0) or 0.0),
        "gripper_timestamp": float(getattr(gmsg, "time_stamp", 0.0) or 0.0),
    }


def _evorl_arm_units(sample: dict) -> np.ndarray:
    joints = np.asarray(sample["joint_raw_milli"], dtype=np.float64) * 1e-3
    gripper = np.asarray([abs(sample["gripper_raw_milli"] * 1e-3)], dtype=np.float64)
    return np.concatenate([joints, gripper], axis=0)


def _rlinf_controller_units(samples: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    joint_position = np.concatenate(
        [np.asarray(sample["joint_raw_milli"], dtype=np.float64) * 1e-3 for sample in samples],
        axis=0,
    )
    gripper_position = np.asarray(
        [abs(sample["gripper_raw_milli"] * 1e-3) for sample in samples],
        dtype=np.float64,
    )
    return joint_position, gripper_position


def _rlinf_realworld_order(joint_position: np.ndarray, gripper_position: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            joint_position[:6],
            gripper_position[:1],
            joint_position[6:12],
            gripper_position[1:2],
        ],
        axis=0,
    )


def main() -> None:
    drivers = [_make_driver(can_name) for can_name in CAN_NAMES]
    try:
        time.sleep(0.3)
        samples = [_sdk_sample(driver) for driver in drivers]
    finally:
        for driver in drivers:
            try:
                driver.DisconnectPort()
            except Exception:
                pass

    evorl_left = _evorl_arm_units(samples[0])
    evorl_right = _evorl_arm_units(samples[1])
    evorl_bimanual = np.concatenate([evorl_left, evorl_right], axis=0)

    rlinf_joint_position, rlinf_gripper_position = _rlinf_controller_units(samples)
    rlinf_bimanual = _rlinf_realworld_order(rlinf_joint_position, rlinf_gripper_position)

    diff = evorl_bimanual - rlinf_bimanual
    report = {
        "can_names": list(CAN_NAMES),
        "sdk_samples": samples,
        "evorl_bimanual_state_14": evorl_bimanual.tolist(),
        "rlinf_controller_joint_position_12": rlinf_joint_position.tolist(),
        "rlinf_controller_gripper_position_2": rlinf_gripper_position.tolist(),
        "rlinf_realworld_state_14": rlinf_bimanual.tolist(),
        "diff_evorl_minus_rlinf_14": diff.tolist(),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
    }
    OUT_PATH.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
