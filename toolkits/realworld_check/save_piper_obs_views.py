#!/usr/bin/env python
# Copyright 2026 The RLinf Authors.
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

from __future__ import annotations

import argparse
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from PIL import Image

from rlinf.envs.realworld.realworld_env import RealWorldEnv


def _to_rgb_hwc_uint8(x: torch.Tensor | np.ndarray) -> np.ndarray:
    arr = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D image array, got shape {arr.shape}")
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected CHW or HWC RGB image, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _save_image(path: Path, image: torch.Tensor | np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_to_rgb_hwc_uint8(image)).save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save one realworld Piper observation frame as main/wrist0/wrist1 PNGs."
    )
    parser.add_argument(
        "--config-name",
        default="realworld_piper_pi05_infer",
        help="Hydra config name under examples/embodiment/config.",
    )
    parser.add_argument(
        "--config-path",
        default="examples/embodiment/config",
        help="Hydra config path.",
    )
    parser.add_argument(
        "--out-dir",
        default="/tmp/piper_obs_views",
        help="Directory for main.png / wrist0.png / wrist1.png.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()

    with hydra.initialize_config_dir(
        config_dir=str(Path(args.config_path).resolve()),
        version_base=None,
    ):
        cfg = hydra.compose(config_name=args.config_name)

    override_cfg = cfg.env.eval.get("override_cfg", None)
    if override_cfg is None:
        with open_dict(cfg.env.eval):
            cfg.env.eval.override_cfg = OmegaConf.create({})
        override_cfg = cfg.env.eval.override_cfg

    if not cfg.env.eval.override_cfg.get("can_name", None):
        node_groups = cfg.get("cluster", {}).get("node_groups", [])
        if node_groups:
            hardware = node_groups[0].get("hardware", {})
            configs = hardware.get("configs", [])
            if configs and configs[0].get("can_name", None):
                with open_dict(cfg.env.eval.override_cfg):
                    cfg.env.eval.override_cfg.can_name = str(configs[0].can_name)

    env = RealWorldEnv(
        cfg.env.eval,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    try:
        obs, _ = env.reset()

        main = obs["main_images"][0]
        _save_image(out_dir / "main.png", main)

        extra = obs.get("extra_view_images")
        if extra is not None:
            if extra.shape[1] > 0:
                _save_image(out_dir / "wrist0.png", extra[0, 0])
            if extra.shape[1] > 1:
                _save_image(out_dir / "wrist1.png", extra[0, 1])

        print(f"saved_dir={out_dir}")
        print(f"main_image_key={cfg.env.eval.main_image_key}")
        print(
            "camera_serials="
            f"{OmegaConf.to_container(cfg.env.eval.override_cfg.camera_serials, resolve=True)}"
        )
        print(f"main_shape={tuple(obs['main_images'].shape)}")
        if extra is None:
            print("extra_view_images=None")
        else:
            print(f"extra_shape={tuple(extra.shape)}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
