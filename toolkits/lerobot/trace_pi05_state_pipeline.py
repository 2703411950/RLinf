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
from safetensors import safe_open

from rlinf.envs.realworld.realworld_env import RealWorldEnv
from rlinf.models import get_model
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config


OUT_PATH = Path("/tmp/pi05_state_trace.json")


def _to_np(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        t = x.detach().cpu()
        if t.dtype == torch.bfloat16:
            t = t.to(torch.float32)
        return t.numpy()
    return np.asarray(x)


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


def _load_preprocessor_stats(checkpoint_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    candidates = sorted(checkpoint_dir.glob("policy_preprocessor_step_*_normalizer_processor.safetensors"))
    if not candidates:
        raise FileNotFoundError(f"No preprocessor stats found under {checkpoint_dir}")
    stats_path = candidates[-1]
    with safe_open(str(stats_path), framework="pt", device="cpu") as handle:
        q01 = handle.get_tensor("observation.state.q01").numpy()
        q99 = handle.get_tensor("observation.state.q99").numpy()
    return q01.astype(np.float32), q99.astype(np.float32)


def _digitize_state(state32: np.ndarray) -> np.ndarray:
    return np.digitize(state32, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1


def _dim_report(raw: np.ndarray, norm: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for i in range(raw.shape[0]):
        denom = float(q99[i] - q01[i])
        out_of_range = bool(raw[i] < q01[i] or raw[i] > q99[i])
        if denom == 0:
            z = None
        else:
            z = float(norm[i])
        rows.append(
            {
                "dim": i,
                "raw": float(raw[i]),
                "q01": float(q01[i]),
                "q99": float(q99[i]),
                "normalized": z,
                "below_q01": bool(raw[i] < q01[i]),
                "above_q99": bool(raw[i] > q99[i]),
                "out_of_range": out_of_range,
            }
        )
    return rows


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

    ckpt_dir = Path(str(rollout_model_config.model_path)).resolve()
    q01, q99 = _load_preprocessor_stats(ckpt_dir)

    actor_train_config = get_openpi_config(
        rollout_model_config.openpi.config_name,
        model_path=rollout_model_config.model_path,
        data_kwargs=OmegaConf.to_container(cfg.actor.openpi_data, resolve=True),
    )
    data_config = actor_train_config.data.create(actor_train_config.assets_dirs, actor_train_config.model)
    transforms_seq = [
        ("data_transform_0_piper_inputs", data_config.data_transforms.inputs[0]),
        ("normalize", policy._input_transform.transforms[2]),
        ("resize_images_like_evorl", data_config.model_transforms.inputs[1]),
        ("pad_state_for_pi05_prompt", data_config.model_transforms.inputs[2]),
        ("tokenize_prompt", data_config.model_transforms.inputs[3]),
        ("pad_states_and_actions", data_config.model_transforms.inputs[4]),
    ]

    sample = {
        "observation/image": _to_np(to_process_obs["observation/image"][0]),
        "observation/wrist_image": _to_np(to_process_obs["observation/wrist_image"][0]),
        "observation/extra_view_image": _to_np(to_process_obs["observation/extra_view_image"][0]),
        "observation/state": _to_np(to_process_obs["observation/state"][0]),
        "prompt": to_process_obs["prompt"][0],
    }

    trace: dict[str, Any] = {
        "checkpoint": str(ckpt_dir),
        "raw_state_14": sample["observation/state"].tolist(),
        "q01_14": q01[:14].tolist(),
        "q99_14": q99[:14].tolist(),
    }

    current = sample
    stage_states: dict[str, Any] = {}
    for name, transform in transforms_seq:
        current = transform(current)
        if "state" in current:
            stage_states[name] = _to_np(current["state"]).tolist()
        if "tokenized_prompt" in current:
            stage_states["tokenized_prompt_head"] = _to_np(current["tokenized_prompt"])[:64].tolist()
            stage_states["tokenized_prompt_mask_mean"] = float(_to_np(current["tokenized_prompt_mask"]).mean())

    raw14 = np.asarray(sample["observation/state"], dtype=np.float32)
    norm14 = np.asarray(stage_states["normalize"], dtype=np.float32)
    pad32 = np.asarray(stage_states["pad_state_for_pi05_prompt"], dtype=np.float32)
    digit32 = _digitize_state(pad32)

    trace["normalized_state_14"] = norm14.tolist()
    trace["padded_state_32"] = pad32.tolist()
    trace["digitized_state_32"] = digit32.tolist()
    trace["dimension_report"] = _dim_report(raw14, norm14, q01[:14], q99[:14])
    trace["out_of_range_dims"] = [row["dim"] for row in trace["dimension_report"] if row["out_of_range"]]
    trace["severely_out_of_range_dims"] = [
        row["dim"] for row in trace["dimension_report"] if row["normalized"] is not None and abs(row["normalized"]) > 1.5
    ]
    trace["tokenized_prompt_head"] = stage_states.get("tokenized_prompt_head", [])
    trace["tokenized_prompt_mask_mean"] = stage_states.get("tokenized_prompt_mask_mean")

    OUT_PATH.write_text(json.dumps(trace, indent=2))
    print(
        json.dumps(
            {
                "out_path": str(OUT_PATH),
                "out_of_range_dims": trace["out_of_range_dims"],
                "severely_out_of_range_dims": trace["severely_out_of_range_dims"],
                "raw_state_14": trace["raw_state_14"],
                "normalized_state_14": trace["normalized_state_14"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
