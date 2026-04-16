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
from PIL import Image

from openpi.models import model as openpi_model
from rlinf.envs.realworld.realworld_env import RealWorldEnv
from rlinf.models import get_model
from rlinf.models.embodiment.openpi import _load_lerobot_norm_stats
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config


OUT_DIR = Path("/tmp/pi05_image_trace")


def _to_numpy(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        t = x.detach().cpu()
        if t.dtype == torch.bfloat16:
            t = t.to(torch.float32)
        return t.numpy()
    return np.asarray(x)


def _chw_or_hwc_to_hwc_uint8(x: Any) -> np.ndarray:
    arr = _to_numpy(x)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D image, got shape {arr.shape}")
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 1.0) if arr.max() <= 1.0 else np.clip(arr, 0.0, 255.0)
            arr = (arr * 255.0).round().astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _save_image(path: Path, x: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_chw_or_hwc_to_hwc_uint8(x)).save(path)


def _summary(x: Any) -> dict[str, Any]:
    arr = _to_numpy(x)
    flat = arr.astype(np.float32).reshape(-1)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(flat.min()) if flat.size else None,
        "max": float(flat.max()) if flat.size else None,
        "mean": float(flat.mean()) if flat.size else None,
        "std": float(flat.std()) if flat.size else None,
    }


def _same(a: Any, b: Any) -> dict[str, Any]:
    aa = torch.as_tensor(_to_numpy(a)).float()
    bb = torch.as_tensor(_to_numpy(b)).float()
    diff = aa - bb
    return {
        "shape_a": list(aa.shape),
        "shape_b": list(bb.shape),
        "max_abs_diff": float(diff.abs().max()),
        "mean_abs_diff": float(diff.abs().mean()),
        "allclose_1e-6": bool(torch.allclose(aa, bb, atol=1e-6, rtol=1e-6)),
    }


def _ensure_cfg_camera_serials(cfg) -> None:
    override_cfg = cfg.env.eval.get("override_cfg", None)
    if override_cfg is None:
        with open_dict(cfg.env.eval):
            cfg.env.eval.override_cfg = OmegaConf.create({})
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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with hydra.initialize_config_dir(
        config_dir=str(Path("examples/embodiment/config").resolve()),
        version_base=None,
    ):
        cfg = hydra.compose(config_name="realworld_piper_pi05_infer")

    _ensure_cfg_camera_serials(cfg)
    with open_dict(cfg.env.eval.override_cfg):
        cfg.env.eval.override_cfg.dry_run_commands = True
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

    report: dict[str, Any] = {"stages": {}, "comparisons": {}}

    # Stage 0: env obs
    env_main = obs["main_images"][0]
    env_left = obs["wrist_images"][0]
    env_right = obs["extra_view_images"][0, 1]
    _save_image(OUT_DIR / "00_env_main.png", env_main)
    _save_image(OUT_DIR / "00_env_left.png", env_left)
    _save_image(OUT_DIR / "00_env_right.png", env_right)
    report["stages"]["env"] = {
        "main_images": _summary(env_main),
        "wrist_images": _summary(env_left),
        "extra_view_right": _summary(env_right),
    }

    # Stage 1: obs_processor
    processed = policy.obs_processor(obs)
    proc_main = processed["observation/image"][0]
    proc_left = processed["observation/wrist_image"][0]
    proc_extra = processed["observation/extra_view_image"][0]
    proc_right = proc_extra[1]
    _save_image(OUT_DIR / "01_obs_processor_main.png", proc_main)
    _save_image(OUT_DIR / "01_obs_processor_left.png", proc_left)
    _save_image(OUT_DIR / "01_obs_processor_right.png", proc_right)
    report["stages"]["obs_processor"] = {
        "observation/image": _summary(proc_main),
        "observation/wrist_image": _summary(proc_left),
        "observation/extra_view_image[1]": _summary(proc_right),
    }
    report["comparisons"]["env_vs_obs_processor"] = {
        "main": _same(env_main, proc_main),
        "left": _same(env_left, proc_left),
        "right": _same(env_right, proc_right),
    }

    # Stage 2+: every transform
    actor_train_config = get_openpi_config(
        rollout_model_config.openpi.config_name,
        model_path=rollout_model_config.model_path,
        data_kwargs=OmegaConf.to_container(cfg.actor.openpi_data, resolve=True),
    )
    data_config = actor_train_config.data.create(
        actor_train_config.assets_dirs, actor_train_config.model
    )
    norm_stats = _load_lerobot_norm_stats(str(rollout_model_config.model_path))
    transforms_seq = [
        ("data_transform_0_piper_inputs", data_config.data_transforms.inputs[0]),
        ("normalize", __import__("openpi.transforms", fromlist=["Normalize"]).Normalize(
            norm_stats, use_quantiles=data_config.use_quantile_norm
        )),
        *[
            (f"model_transform_{idx}_{type(t).__name__}", t)
            for idx, t in enumerate(data_config.model_transforms.inputs)
        ],
    ]

    sample = {
        "observation/image": _to_numpy(proc_main),
        "observation/wrist_image": _to_numpy(proc_left),
        "observation/extra_view_image": _to_numpy(proc_extra),
        "observation/state": _to_numpy(processed["observation/state"][0]),
        "prompt": processed["prompt"][0],
    }

    for idx, (name, transform) in enumerate(transforms_seq, start=2):
        sample = transform(sample)
        stage_key = f"{idx:02d}_{name}"
        report["stages"][stage_key] = {"keys": sorted(sample.keys())}
        if "image" in sample:
            base = sample["image"]["base_0_rgb"]
            left = sample["image"]["left_wrist_0_rgb"]
            right = sample["image"]["right_wrist_0_rgb"]
            _save_image(OUT_DIR / f"{stage_key}_main.png", base)
            _save_image(OUT_DIR / f"{stage_key}_left.png", left)
            _save_image(OUT_DIR / f"{stage_key}_right.png", right)
            report["stages"][stage_key]["image/base_0_rgb"] = _summary(base)
            report["stages"][stage_key]["image/left_wrist_0_rgb"] = _summary(left)
            report["stages"][stage_key]["image/right_wrist_0_rgb"] = _summary(right)
        if "state" in sample:
            report["stages"][stage_key]["state"] = _summary(sample["state"])
        if "tokenized_prompt" in sample:
            report["stages"][stage_key]["tokenized_prompt"] = _summary(sample["tokenized_prompt"])
            report["stages"][stage_key]["tokenized_prompt_mask"] = _summary(
                sample["tokenized_prompt_mask"]
            )

    # Final observation into model preprocess
    final_batch = {}
    for key, value in sample.items():
        if isinstance(value, dict):
            final_batch[key] = {k: torch.as_tensor(v)[None].to(next(policy.parameters()).device) for k, v in value.items()}
        elif key == "prompt":
            continue
        else:
            final_batch[key] = torch.as_tensor(value)[None].to(next(policy.parameters()).device)
    obs_obj = openpi_model.Observation.from_dict(final_batch)
    images, img_masks, lang_tokens, lang_masks, state = policy._preprocess_observation(
        obs_obj, train=False
    )
    preprocess_stage = "99_model_preprocess"
    for name, image in zip(
        ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"), images, strict=True
    ):
        _save_image(OUT_DIR / f"{preprocess_stage}_{name}.png", image[0])
    report["stages"][preprocess_stage] = {
        "image/base_0_rgb": _summary(images[0][0]),
        "image/left_wrist_0_rgb": _summary(images[1][0]),
        "image/right_wrist_0_rgb": _summary(images[2][0]),
        "image_masks": [bool(x.item()) for x in img_masks],
        "tokenized_prompt": _summary(lang_tokens[0]),
        "tokenized_prompt_mask": _summary(lang_masks[0]),
        "state": _summary(state[0]),
    }

    # Embed prefix entry confirmation
    prefix_embs, prefix_pad_masks, prefix_att_masks = policy.embed_prefix(
        images, img_masks, lang_tokens, lang_masks
    )
    report["stages"]["100_embed_prefix"] = {
        "prefix_embs": _summary(prefix_embs[0]),
        "prefix_pad_masks": _summary(prefix_pad_masks[0]),
        "prefix_att_masks": _summary(prefix_att_masks[0]),
    }

    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "out_dir": str(OUT_DIR),
        "env_vs_obs_processor": report["comparisons"]["env_vs_obs_processor"],
        "model_preprocess_masks": report["stages"]["99_model_preprocess"]["image_masks"],
    }, indent=2))


if __name__ == "__main__":
    main()
