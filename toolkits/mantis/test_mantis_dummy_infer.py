#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.models import get_model


def build_dummy_obs(batch_size: int = 1) -> dict:
    main_images = np.random.randint(0, 256, size=(batch_size, 128, 128, 3), dtype=np.uint8)
    left_wrist = np.random.randint(0, 256, size=(batch_size, 128, 128, 3), dtype=np.uint8)
    right_wrist = np.random.randint(0, 256, size=(batch_size, 128, 128, 3), dtype=np.uint8)
    states = np.zeros((batch_size, 14), dtype=np.float32)
    return {
        "main_images": torch.from_numpy(main_images),
        "wrist_images": torch.from_numpy(left_wrist),
        "extra_view_images": torch.from_numpy(np.stack([left_wrist, right_wrist], axis=1)),
        "states": torch.from_numpy(states),
        "task_descriptions": ["put object into bag"] * batch_size,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="examples/embodiment/config/model/mantis.yaml",
        help="Path to the RLinf actor model config.",
    )
    parser.add_argument("--save-json", default="", help="Optional path to save summary json.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    model = get_model(cfg)
    model.eval()

    device = next(model.parameters()).device
    print(f"model device={device}")

    obs = build_dummy_obs()
    obs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in obs.items()}
    with torch.no_grad():
        env_actions, result = model.predict_action_batch(env_obs=obs, mode="eval")

    raw = result["forward_inputs"]["raw_normalized_action"].numpy()
    summary = {
        "raw_shape": list(raw.shape),
        "raw_min": float(raw.min()),
        "raw_max": float(raw.max()),
        "raw_mean": float(raw.mean()),
        "raw_std": float(raw.std()),
        "env_shape": list(env_actions.shape),
        "env_min": float(env_actions.min()),
        "env_max": float(env_actions.max()),
        "env_mean": float(env_actions.mean()),
        "env_std": float(env_actions.std()),
    }
    print(json.dumps(summary, indent=2))

    if args.save_json:
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
