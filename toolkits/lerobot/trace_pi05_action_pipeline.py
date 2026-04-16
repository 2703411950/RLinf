#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

from openpi.models import model as openpi_model
from rlinf.envs.action_utils import prepare_actions
from rlinf.envs.realworld.realworld_env import RealWorldEnv
from rlinf.models import get_model


OUT_PATH = Path("/tmp/pi05_action_trace.json")


def _to_np(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        t = x.detach().cpu()
        if t.dtype == torch.bfloat16:
            t = t.to(torch.float32)
        return t.numpy()
    return np.asarray(x)


def _summary(x: Any, head: int = 16) -> dict[str, Any]:
    arr = _to_np(x)
    flat = arr.astype(np.float32).reshape(-1)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(flat.min()) if flat.size else None,
        "max": float(flat.max()) if flat.size else None,
        "mean": float(flat.mean()) if flat.size else None,
        "std": float(flat.std()) if flat.size else None,
        "head": flat[:head].tolist(),
    }


def _diff(a: Any, b: Any, head: int = 16) -> dict[str, Any]:
    aa = torch.as_tensor(_to_np(a)).float()
    bb = torch.as_tensor(_to_np(b)).float()
    diff = aa - bb
    return {
        "shape_a": list(aa.shape),
        "shape_b": list(bb.shape),
        "max_abs_diff": float(diff.abs().max()),
        "mean_abs_diff": float(diff.abs().mean()),
        "head_a": aa.reshape(-1)[:head].tolist(),
        "head_b": bb.reshape(-1)[:head].tolist(),
        "head_diff": diff.reshape(-1)[:head].tolist(),
    }


def _controller_unit_convert(actions_14: np.ndarray, io_unit_mode: str) -> np.ndarray:
    actions = np.asarray(actions_14, dtype=np.float64)
    if io_unit_mode == "evo_rl_unit":
        return np.round(actions * 1000.0).astype(np.int64)
    out = np.zeros_like(actions, dtype=np.int64)
    out[:6] = np.round(actions[:6] * 57295.7795).astype(np.int64)
    out[6] = int(round(actions[6] * 70 * 1000))
    out[7:13] = np.round(actions[7:13] * 57295.7795).astype(np.int64)
    out[13] = int(round(actions[13] * 70 * 1000))
    return out


def _ensure_cfg(cfg) -> None:
    override_cfg = cfg.env.eval.get("override_cfg", None)
    if override_cfg is None:
        with open_dict(cfg.env.eval):
            cfg.env.eval.override_cfg = OmegaConf.create({})
    with open_dict(cfg.env.eval.override_cfg):
        cfg.env.eval.override_cfg.dry_run_commands = True
    if not cfg.env.eval.override_cfg.get("can_name", None):
        node_groups = cfg.get("cluster", {}).get("node_groups", [])
        if node_groups:
            hardware = node_groups[0].get("hardware", {})
            configs = hardware.get("configs", [])
            if configs and configs[0].get("can_name", None):
                with open_dict(cfg.env.eval.override_cfg):
                    cfg.env.eval.override_cfg.can_name = str(configs[0].can_name)


def _obs_for_policy(policy, obs: dict) -> dict:
    device = next(policy.parameters()).device
    out = {}
    for key, val in obs.items():
        if key == "task_descriptions":
            out[key] = val
        elif isinstance(val, np.ndarray):
            out[key] = torch.from_numpy(val).to(device)
        elif torch.is_tensor(val):
            out[key] = val.to(device, non_blocking=True)
        else:
            out[key] = val
    if out.get("wrist_images") is None:
        ev = out.get("extra_view_images")
        if isinstance(ev, torch.Tensor) and ev.dim() >= 2 and ev.shape[1] > 0:
            out["wrist_images"] = ev[:, 0].contiguous()
        elif "main_images" in out:
            out["wrist_images"] = out["main_images"].clone()
    return out


def main() -> None:
    with hydra.initialize_config_dir(
        config_dir=str(Path("examples/embodiment/config").resolve()),
        version_base=None,
    ):
        cfg = hydra.compose(config_name="realworld_piper_pi05_infer")

    _ensure_cfg(cfg)
    env = RealWorldEnv(
        cfg.env.eval,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    try:
        obs, _ = env.reset()
    finally:
        env.close()

    rollout_model_config = copy.deepcopy(cfg.actor.model)
    with open_dict(rollout_model_config):
        rollout_model_config.precision = cfg.rollout.model.precision
        rollout_model_config.model_path = cfg.rollout.model.model_path
        if hasattr(cfg.actor, "openpi_data"):
            rollout_model_config.openpi_data = copy.deepcopy(cfg.actor.openpi_data)
    policy = get_model(rollout_model_config)
    policy.eval()

    obs = _obs_for_policy(policy, obs)
    to_process_obs = policy.obs_processor(obs)
    processed_obs = policy.input_transform(to_process_obs, transpose=False)
    processed_obs = policy.precision_processor(processed_obs)
    observation = openpi_model.Observation.from_dict(processed_obs)

    outputs = policy.sample_actions(observation, mode="eval", compute_values=False)
    raw_actions = outputs["actions"]
    denorm_actions = policy.output_transform(
        {"actions": raw_actions, "state": observation.state}
    )["actions"]
    prepared_actions = prepare_actions(
        raw_chunk_actions=denorm_actions.detach().cpu().numpy(),
        env_type=cfg.env.eval.env_type,
        model_type=cfg.actor.model.model_type,
        num_action_chunks=cfg.actor.model.num_action_chunks,
        action_dim=cfg.actor.model.action_dim,
        policy="piper",
    )

    raw_np = _to_np(raw_actions)[0]
    denorm_np = _to_np(denorm_actions)[0]
    prepared_np = _to_np(prepared_actions)[0]
    controller_units = np.stack(
        [
            _controller_unit_convert(step, cfg.env.eval.override_cfg.io_unit_mode)
            for step in prepared_np
        ],
        axis=0,
    )

    payload = {
        "config": {
            "model_path": str(rollout_model_config.model_path),
            "io_unit_mode": str(cfg.env.eval.override_cfg.io_unit_mode),
            "num_action_chunks": int(cfg.actor.model.num_action_chunks),
            "model_action_dim": int(policy.config.action_dim),
            "action_horizon": int(policy.config.action_horizon),
            "env_action_dim": int(policy.config.action_env_dim),
        },
        "raw_sample_actions": _summary(raw_np),
        "raw_sample_actions_first_step_14d": _summary(raw_np[0, :14]),
        "after_output_transform": _summary(denorm_np),
        "after_output_transform_first_step_14d": _summary(denorm_np[0]),
        "after_prepare_actions": _summary(prepared_np),
        "after_prepare_actions_first_step_14d": _summary(prepared_np[0]),
        "after_controller_unit_conversion": _summary(controller_units),
        "after_controller_unit_conversion_first_step_14d": _summary(controller_units[0]),
        "diff_raw_vs_denorm_first_step_14d": _diff(raw_np[0, :14], denorm_np[0]),
        "diff_denorm_vs_prepared_first_step_14d": _diff(denorm_np[0], prepared_np[0]),
        "raw_head_steps_3": raw_np[:3, :14].tolist(),
        "denorm_head_steps_3": denorm_np[:3].tolist(),
        "prepared_head_steps_3": prepared_np[:3].tolist(),
        "controller_units_head_steps_3": controller_units[:3].tolist(),
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(json.dumps(
        {
            "out_path": str(OUT_PATH),
            "raw_sample_actions_first_step_14d": payload["raw_sample_actions_first_step_14d"],
            "after_output_transform_first_step_14d": payload["after_output_transform_first_step_14d"],
            "after_controller_unit_conversion_first_step_14d": payload[
                "after_controller_unit_conversion_first_step_14d"
            ],
            "diff_raw_vs_denorm_first_step_14d": payload["diff_raw_vs_denorm_first_step_14d"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
