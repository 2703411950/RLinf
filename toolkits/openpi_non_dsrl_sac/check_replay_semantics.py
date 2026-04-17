#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
"""Inspect PI05 replay data semantics before adapting non-DSRL SAC."""

from __future__ import annotations

import argparse
import json
from typing import Any

import torch

from toolkits.openpi_non_dsrl_sac.common import (
    DEFAULT_ENV_ACTION_DIM,
    DEFAULT_REPLAY_DIR,
    extract_policy_obs_from_batch,
    infer_action_layout,
    load_trajectory,
    nested_tensor_shapes,
    tensor_stats,
)


def _compare_actions(traj: Any) -> dict[str, Any]:
    actions = traj.actions
    recorded = traj.forward_inputs.get("action") if traj.forward_inputs else None
    if not torch.is_tensor(actions) or not torch.is_tensor(recorded):
        return {"available": False}
    diff = (actions - recorded).abs()
    return {
        "available": True,
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.float().mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", default=DEFAULT_REPLAY_DIR)
    parser.add_argument("--trajectory-id", type=int, default=0)
    parser.add_argument("--env-action-dim", type=int, default=DEFAULT_ENV_ACTION_DIM)
    args = parser.parse_args()

    traj = load_trajectory(args.replay_dir, args.trajectory_id)
    obs_summary = nested_tensor_shapes(traj.curr_obs or {})
    forward_summary = nested_tensor_shapes(traj.forward_inputs or {})

    action_tensor = traj.actions
    if not torch.is_tensor(action_tensor):
        raise RuntimeError("Trajectory does not contain tensor actions.")

    action_layout = infer_action_layout(action_tensor.shape[-1], args.env_action_dim)
    sampled_obs = extract_policy_obs_from_batch({"curr_obs": traj.curr_obs})

    report = {
        "replay_dir": args.replay_dir,
        "trajectory_id": args.trajectory_id,
        "action_layout": action_layout,
        "actions": tensor_stats(action_tensor),
        "curr_obs_shapes": obs_summary,
        "forward_input_shapes": forward_summary,
        "sampled_policy_obs": nested_tensor_shapes(sampled_obs),
        "actions_vs_forward_inputs_action": _compare_actions(traj),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if not action_layout["is_divisible"]:
        raise SystemExit(
            f"Flat action dim {action_tensor.shape[-1]} is not divisible by env action dim {args.env_action_dim}."
        )


if __name__ == "__main__":
    main()
