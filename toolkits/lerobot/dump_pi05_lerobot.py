#!/usr/bin/env python3

"""Run one PI05 sample through LeRobot native inference and dump JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample-npz", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-steps", type=int, default=None)
    return parser.parse_args()


def summarize_tensor(name: str, value: Any, max_items: int = 8) -> dict[str, Any]:
    tensor = value.detach().cpu() if torch.is_tensor(value) else torch.as_tensor(value)
    flat = tensor.reshape(-1).float()
    return {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "min": float(flat.min()) if flat.numel() else None,
        "max": float(flat.max()) if flat.numel() else None,
        "mean": float(flat.mean()) if flat.numel() else None,
        "std": float(flat.std(unbiased=False)) if flat.numel() else None,
        "head": flat[:max_items].tolist(),
    }


def ensure_hwc_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr


def hwc_to_chw_float01(image_hwc: np.ndarray) -> torch.Tensor:
    arr = image_hwc.astype(np.float32) / 255.0
    return torch.from_numpy(np.transpose(arr, (2, 0, 1)))


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    lerobot_root = repo_root.parent / "lerobot"
    sys.path.insert(0, str(lerobot_root / "src"))

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.processor.pipeline import DataProcessorPipeline
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    args = parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    with np.load(args.sample_npz, allow_pickle=True) as data:
        sample = {k: data[k] for k in data.files}

    config = PreTrainedConfig.from_pretrained(str(checkpoint))
    config.device = args.device
    policy = PI05Policy.from_pretrained(str(checkpoint), config=config, strict=True).eval()
    preprocessor = DataProcessorPipeline.from_pretrained(
        str(checkpoint),
        config_filename="policy_preprocessor.json",
        overrides={"device_processor": {"device": args.device}},
    )
    postprocessor = DataProcessorPipeline.from_pretrained(
        str(checkpoint),
        config_filename="policy_postprocessor.json",
        overrides={"device_processor": {"device": "cpu"}},
    )

    batch = {
        "observation.images.cam_high": hwc_to_chw_float01(ensure_hwc_uint8(sample["main_image"]))[None].to(
            args.device
        ),
        "observation.images.cam_left_wrist": hwc_to_chw_float01(
            ensure_hwc_uint8(sample["left_wrist_image"])
        )[None].to(args.device),
        "observation.images.cam_right_wrist": hwc_to_chw_float01(
            ensure_hwc_uint8(sample["right_wrist_image"])
        )[None].to(args.device),
        "observation.state": torch.from_numpy(np.asarray(sample["state"], dtype=np.float32))[None].to(
            args.device
        ),
        "task": [str(sample["prompt"].tolist() if hasattr(sample["prompt"], "tolist") else sample["prompt"])],
    }

    processed = preprocessor(batch)
    images, img_masks = policy._preprocess_images(processed)
    tokens = processed[OBS_LANGUAGE_TOKENS]
    masks = processed[OBS_LANGUAGE_ATTENTION_MASK]

    torch.manual_seed(args.seed)
    noise = torch.randn(
        (tokens.shape[0], policy.config.chunk_size, policy.config.max_action_dim),
        dtype=torch.float32,
        device=tokens.device,
    )

    with torch.no_grad():
        raw_actions = policy.model.sample_actions(
            images,
            img_masks,
            tokens,
            masks,
            noise=noise,
            num_steps=args.num_steps,
        )
    raw_actions = raw_actions[:, :, : policy.config.output_features["action"].shape[0]]
    denorm_actions = postprocessor({"action": raw_actions.detach().cpu()})["action"]

    result = {
        "framework": "lerobot",
        "checkpoint": str(checkpoint),
        "sample_npz": str(Path(args.sample_npz).resolve()),
        "device": args.device,
        "seed": args.seed,
        "processed": {
            "observation.state": summarize_tensor("observation.state", processed["observation.state"]),
            "tokenized_prompt": summarize_tensor("tokenized_prompt", processed[OBS_LANGUAGE_TOKENS]),
            "tokenized_prompt_mask": summarize_tensor(
                "tokenized_prompt_mask", processed[OBS_LANGUAGE_ATTENTION_MASK]
            ),
        },
        "raw_actions": summarize_tensor("raw_actions", raw_actions),
        "denorm_actions": summarize_tensor("denorm_actions", denorm_actions),
        "raw_arrays": {
            "processed_state": processed["observation.state"].detach().cpu().tolist(),
            "tokenized_prompt": processed[OBS_LANGUAGE_TOKENS].detach().cpu().tolist(),
            "tokenized_prompt_mask": processed[OBS_LANGUAGE_ATTENTION_MASK].detach().cpu().tolist(),
            "raw_actions": raw_actions.detach().cpu().tolist(),
            "denorm_actions": denorm_actions.detach().cpu().tolist(),
        },
    }

    out_path = Path(args.output_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ["framework", "checkpoint", "sample_npz", "device", "seed"]}, indent=2))
    print(json.dumps(result["processed"], indent=2))
    print(json.dumps({"raw_actions": result["raw_actions"], "denorm_actions": result["denorm_actions"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
