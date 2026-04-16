#!/usr/bin/env python3

"""Compare PI05 observation preprocessing between Evo-RL semantics and RLinf."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from safetensors import safe_open


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openpi.models.tokenizer import PaligemmaTokenizer
import openpi.transforms as openpi_transforms
from rlinf.models.embodiment.openpi import (
    _load_lerobot_checkpoint_config,
    _load_lerobot_norm_stats,
)
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="PI05 checkpoint directory.")
    parser.add_argument("--sample-npz", help="Optional sample npz with main/left/right/state/prompt.")
    parser.add_argument("--save-json", help="Optional json output.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_image(height: int, width: int, offset: int) -> np.ndarray:
    y = np.linspace(0, 255, height, dtype=np.float32)[:, None]
    x = np.linspace(0, 255, width, dtype=np.float32)[None, :]
    r = (x + offset) % 256
    g = (y + offset * 2) % 256
    b = ((x * 0.35 + y * 0.65) + offset * 3) % 256
    return np.stack([r.repeat(height, axis=0), g.repeat(width, axis=1), b], axis=-1).astype(np.uint8)


def default_sample(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    return {
        "main_image": make_image(480, 640, 0),
        "left_wrist_image": make_image(480, 640, 17),
        "right_wrist_image": make_image(480, 640, 33),
        "state": rng.uniform(-1.0, 1.0, size=(14,)).astype(np.float32),
        "prompt": "unscrew the bottle cap",
    }


def load_sample(sample_npz: str | None, seed: int) -> dict[str, Any]:
    if sample_npz is None:
        return default_sample(seed)
    sample: dict[str, Any] = {}
    with np.load(sample_npz, allow_pickle=True) as data:
        for key in data.files:
            value = data[key]
            if key == "prompt":
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                if isinstance(value, list):
                    value = value[0]
                sample[key] = str(value)
            else:
                sample[key] = value
    return sample


def ensure_hwc_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D image, got {arr.shape}")
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return np.ascontiguousarray(arr)


def resize_like_evorl(image: np.ndarray, height: int = 224, width: int = 224) -> np.ndarray:
    arr = ensure_hwc_uint8(image)
    chw = np.transpose(arr, (2, 0, 1))
    image_t = torch.as_tensor(chw).unsqueeze(0)
    resized = F.interpolate(
        image_t,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    resized = resized.to(torch.float32) / 255.0
    return resized.contiguous().cpu().numpy()


def load_norm_state_stats(checkpoint: Path) -> tuple[np.ndarray, np.ndarray]:
    candidates = sorted(checkpoint.glob("policy_preprocessor_step_*_normalizer_processor.safetensors"))
    if not candidates:
        raise FileNotFoundError(f"No normalizer safetensors found under {checkpoint}")
    stats_path = candidates[-1]
    with safe_open(stats_path, framework="pt", device="cpu") as handle:
        q01 = handle.get_tensor("observation.state.q01").numpy()
        q99 = handle.get_tensor("observation.state.q99").numpy()
    return q01, q99


def quantile_normalize(state: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    denom = q99 - q01
    denom = np.where(denom == 0, 1e-6, denom)
    return 2.0 * (state - q01) / denom - 1.0


def pad_state(state: np.ndarray, target_dim: int) -> np.ndarray:
    if state.shape[-1] >= target_dim:
        return state
    return np.pad(state, [(0, target_dim - state.shape[-1])], mode="constant")


def build_rlinf_input_sample(sample: dict[str, Any]) -> dict[str, Any]:
    main = ensure_hwc_uint8(sample["main_image"])
    left = ensure_hwc_uint8(sample["left_wrist_image"])
    right = ensure_hwc_uint8(sample["right_wrist_image"])
    return {
        "observation/image": main,
        "observation/wrist_image": left,
        "observation/extra_view_image": np.stack([left, right], axis=0),
        "observation/state": np.asarray(sample["state"], dtype=np.float32),
        "prompt": str(sample["prompt"]),
    }


def manual_evorl_preprocess(sample: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
    config = json.loads((checkpoint / "config.json").read_text())
    q01, q99 = load_norm_state_stats(checkpoint)
    state = np.asarray(sample["state"], dtype=np.float32)
    norm_state = quantile_normalize(state, q01, q99).astype(np.float32)
    padded_state = pad_state(norm_state, int(config["max_state_dim"])).astype(np.float32)
    tokenizer = PaligemmaTokenizer(int(config["tokenizer_max_length"]))
    tokens, masks = tokenizer.tokenize(str(sample["prompt"]), padded_state)
    return {
        "state": padded_state,
        "tokenized_prompt": tokens,
        "tokenized_prompt_mask": masks,
        "image/base_0_rgb": resize_like_evorl(sample["main_image"]),
        "image/left_wrist_0_rgb": resize_like_evorl(sample["left_wrist_image"]),
        "image/right_wrist_0_rgb": resize_like_evorl(sample["right_wrist_image"]),
    }


def build_rlinf_preprocess_pipeline(checkpoint: Path):
    actor_train_config = get_openpi_config(
        "pi05_piper",
        model_path=str(checkpoint),
        data_kwargs={"env_action_dim": 14, "align_evorl_observation": True},
    )
    model_config = actor_train_config.model
    lerobot_checkpoint_config = _load_lerobot_checkpoint_config(str(checkpoint))
    if lerobot_checkpoint_config is not None:
        object.__setattr__(model_config, "max_state_dim", lerobot_checkpoint_config["max_state_dim"])
        object.__setattr__(
            model_config, "max_token_len", lerobot_checkpoint_config["tokenizer_max_length"]
        )
        object.__setattr__(model_config, "action_dim", lerobot_checkpoint_config["max_action_dim"])
    data_config = actor_train_config.data.create(actor_train_config.assets_dirs, model_config)
    norm_stats = _load_lerobot_norm_stats(str(checkpoint))
    return openpi_transforms.compose(
        [
            openpi_transforms.InjectDefaultPrompt(None),
            *data_config.data_transforms.inputs,
            openpi_transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ]
    )


def summarize_diff(a: np.ndarray | torch.Tensor, b: np.ndarray | torch.Tensor) -> dict[str, Any]:
    ta = torch.as_tensor(a).float()
    tb = torch.as_tensor(b).float()
    diff = ta - tb
    return {
        "shape_a": list(ta.shape),
        "shape_b": list(tb.shape),
        "max_abs_diff": float(diff.abs().max()),
        "mean_abs_diff": float(diff.abs().mean()),
        "allclose_1e-6": bool(torch.allclose(ta, tb, atol=1e-6, rtol=1e-6)),
        "allclose_1e-4": bool(torch.allclose(ta, tb, atol=1e-4, rtol=1e-4)),
    }


def main() -> int:
    args = parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    sample = load_sample(args.sample_npz, args.seed)

    manual = manual_evorl_preprocess(sample, checkpoint)
    preprocess = build_rlinf_preprocess_pipeline(checkpoint)
    processed_obs = preprocess(build_rlinf_input_sample(sample))

    rlinf = {
        "state": np.asarray(processed_obs["state"]),
        "tokenized_prompt": np.asarray(processed_obs["tokenized_prompt"]),
        "tokenized_prompt_mask": np.asarray(processed_obs["tokenized_prompt_mask"]),
        "image/base_0_rgb": np.asarray(processed_obs["image"]["base_0_rgb"]),
        "image/left_wrist_0_rgb": np.asarray(processed_obs["image"]["left_wrist_0_rgb"]),
        "image/right_wrist_0_rgb": np.asarray(processed_obs["image"]["right_wrist_0_rgb"]),
    }

    diffs = {
        "state": summarize_diff(rlinf["state"], manual["state"]),
        "tokenized_prompt": summarize_diff(rlinf["tokenized_prompt"], manual["tokenized_prompt"]),
        "tokenized_prompt_mask": summarize_diff(
            rlinf["tokenized_prompt_mask"], manual["tokenized_prompt_mask"]
        ),
        "image/base_0_rgb": summarize_diff(rlinf["image/base_0_rgb"], manual["image/base_0_rgb"]),
        "image/left_wrist_0_rgb": summarize_diff(
            rlinf["image/left_wrist_0_rgb"], manual["image/left_wrist_0_rgb"]
        ),
        "image/right_wrist_0_rgb": summarize_diff(
            rlinf["image/right_wrist_0_rgb"], manual["image/right_wrist_0_rgb"]
        ),
    }

    print(json.dumps(diffs, indent=2))

    if args.save_json:
        payload = {
            "diffs": diffs,
            "manual_state_head": np.asarray(manual["state"])[:8].tolist(),
            "rlinf_state_head": np.asarray(rlinf["state"])[:8].tolist(),
            "manual_tokens_head": np.asarray(manual["tokenized_prompt"])[:16].tolist(),
            "rlinf_tokens_head": np.asarray(rlinf["tokenized_prompt"])[:16].tolist(),
        }
        Path(args.save_json).write_text(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
