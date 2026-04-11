#!/usr/bin/env python3

"""Compare one-frame PI05 inference between LeRobot and RLinf/OpenPI.

This script loads the same PI05 checkpoint through:
1. LeRobot's native PI05 policy + saved pre/postprocessors.
2. RLinf's OpenPI model wrapper.

It then runs the same observation through both paths and prints:
- Preprocessed inputs.
- Raw sampled actions.
- Denormalized actions.
- Per-tensor differences.

Input can be provided either as a single `.npz` sample or as separate image/state args.
The `.npz` file may contain either HWC or CHW images. Supported keys:
- `main_image`
- `left_wrist_image`
- `right_wrist_image`
- `state`
- `prompt`
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
LEROBOT_ROOT = REPO_ROOT.parent / "lerobot"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(LEROBOT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(LEROBOT_ROOT / "src"))

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.processor.pipeline import DataProcessorPipeline
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from rlinf.models.embodiment.openpi import get_model as get_rlinf_openpi_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare PI05 inference between LeRobot and RLinf/OpenPI on one sample."
    )
    parser.add_argument("--checkpoint", required=True, help="PI05 checkpoint directory.")
    parser.add_argument(
        "--sample-npz",
        help="Optional `.npz` containing main_image/left_wrist_image/right_wrist_image/state/prompt.",
    )
    parser.add_argument("--main-image", help="Path to main camera image.")
    parser.add_argument("--left-wrist-image", help="Path to left wrist image.")
    parser.add_argument("--right-wrist-image", help="Path to right wrist image.")
    parser.add_argument(
        "--state",
        help="Inline JSON array for state, e.g. '[...]'. If omitted, loaded from --state-json or --sample-npz.",
    )
    parser.add_argument("--state-json", help="Path to JSON file containing state array.")
    parser.add_argument("--prompt", help="Task prompt. Overrides prompt from npz.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for both policies.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for explicit noise generation.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Override denoise steps for both paths. Defaults to checkpoint config num_inference_steps.",
    )
    parser.add_argument(
        "--rlinf-config-name",
        default="pi05_piper",
        help="RLinf/OpenPI dataconfig name.",
    )
    parser.add_argument(
        "--rlinf-env-action-dim",
        type=int,
        default=None,
        help="Override RLinf openpi_data.env_action_dim. Defaults to checkpoint action dim.",
    )
    parser.add_argument(
        "--rlinf-num-action-chunks",
        type=int,
        default=None,
        help="Override RLinf action_chunk/num_action_chunks. Defaults to checkpoint chunk size.",
    )
    parser.add_argument(
        "--rlinf-train-expert-only",
        choices=["true", "false"],
        default=None,
        help="Optionally override RLinf openpi.train_expert_only.",
    )
    parser.add_argument(
        "--save-json",
        help="Optional path to save all numeric summaries as JSON.",
    )
    return parser.parse_args()


def load_image_array(path: str | Path) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("RGB"))
    return image


def load_state_array(args: argparse.Namespace, sample: dict[str, Any]) -> np.ndarray:
    if args.state is not None:
        return np.asarray(json.loads(args.state), dtype=np.float32)
    if args.state_json is not None:
        return np.asarray(json.loads(Path(args.state_json).read_text()), dtype=np.float32)
    if "state" in sample:
        return np.asarray(sample["state"], dtype=np.float32)
    raise ValueError("State is required. Provide --state, --state-json, or --sample-npz with `state`.")


def load_prompt(args: argparse.Namespace, sample: dict[str, Any]) -> str:
    if args.prompt is not None:
        return args.prompt
    prompt = sample.get("prompt")
    if prompt is None:
        raise ValueError("Prompt is required. Provide --prompt or include `prompt` in --sample-npz.")
    if isinstance(prompt, np.ndarray):
        prompt = prompt.tolist()
    if isinstance(prompt, bytes):
        prompt = prompt.decode("utf-8")
    if isinstance(prompt, list):
        if len(prompt) != 1:
            raise ValueError(f"Expected single prompt, got list of length {len(prompt)}.")
        prompt = prompt[0]
    return str(prompt)


def load_sample(args: argparse.Namespace) -> dict[str, Any]:
    sample: dict[str, Any] = {}
    if args.sample_npz is not None:
        with np.load(args.sample_npz, allow_pickle=True) as data:
            for key in data.files:
                sample[key] = data[key]
    if args.main_image is not None:
        sample["main_image"] = load_image_array(args.main_image)
    if args.left_wrist_image is not None:
        sample["left_wrist_image"] = load_image_array(args.left_wrist_image)
    if args.right_wrist_image is not None:
        sample["right_wrist_image"] = load_image_array(args.right_wrist_image)
    if "main_image" not in sample:
        raise ValueError("Main image is required. Provide --main-image or include `main_image` in --sample-npz.")
    sample["state"] = load_state_array(args, sample)
    sample["prompt"] = load_prompt(args, sample)
    return sample


def ensure_hwc_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D image array, got shape {arr.shape}.")
    if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got shape {arr.shape}.")
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 1.0) if arr.max() <= 1.0 else np.clip(arr, 0.0, 255.0)
            arr = (arr * 255.0).round().astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    return arr


def hwc_to_chw_float01(image_hwc: np.ndarray) -> torch.Tensor:
    arr = image_hwc.astype(np.float32) / 255.0
    return torch.from_numpy(np.transpose(arr, (2, 0, 1)))


def hwc_batch_to_torch(image_hwc: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(image_hwc.copy())


def summarize_tensor(name: str, value: Any, max_items: int = 8) -> dict[str, Any]:
    if isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value)
    elif torch.is_tensor(value):
        tensor = value.detach().cpu()
    else:
        return {"name": name, "type": str(type(value)), "value": value}

    flat = tensor.reshape(-1).float()
    summary = {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "min": float(flat.min()) if flat.numel() else None,
        "max": float(flat.max()) if flat.numel() else None,
        "mean": float(flat.mean()) if flat.numel() else None,
        "std": float(flat.std(unbiased=False)) if flat.numel() else None,
        "head": flat[:max_items].tolist(),
    }
    return summary


def print_tensor_summary(title: str, value: Any) -> None:
    summary = summarize_tensor(title, value)
    if "shape" not in summary:
        print(f"{title}: {summary}")
        return
    print(
        f"{title}: shape={summary['shape']} dtype={summary['dtype']} "
        f"min={summary['min']:.6f} max={summary['max']:.6f} "
        f"mean={summary['mean']:.6f} std={summary['std']:.6f}"
    )
    print(f"{title} head: {summary['head']}")


def tensor_diff_summary(name: str, a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    a_cpu = a.detach().cpu().float()
    b_cpu = b.detach().cpu().float()
    diff = a_cpu - b_cpu
    return {
        "name": name,
        "shape_a": list(a_cpu.shape),
        "shape_b": list(b_cpu.shape),
        "max_abs_diff": float(diff.abs().max()),
        "mean_abs_diff": float(diff.abs().mean()),
        "allclose_atol_1e-4": bool(torch.allclose(a_cpu, b_cpu, atol=1e-4, rtol=1e-4)),
        "allclose_atol_1e-3": bool(torch.allclose(a_cpu, b_cpu, atol=1e-3, rtol=1e-3)),
        "allclose_atol_1e-2": bool(torch.allclose(a_cpu, b_cpu, atol=1e-2, rtol=1e-2)),
    }


def print_diff_summary(title: str, a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    summary = tensor_diff_summary(title, a, b)
    print(
        f"{title}: shape_a={summary['shape_a']} shape_b={summary['shape_b']} "
        f"max_abs_diff={summary['max_abs_diff']:.6f} mean_abs_diff={summary['mean_abs_diff']:.6f} "
        f"allclose(1e-4)={summary['allclose_atol_1e-4']} "
        f"allclose(1e-3)={summary['allclose_atol_1e-3']} "
        f"allclose(1e-2)={summary['allclose_atol_1e-2']}"
    )
    return summary


def load_lerobot_policy_and_processors(
    checkpoint: Path, device: str
) -> tuple[PI05Policy, DataProcessorPipeline, DataProcessorPipeline]:
    config = PreTrainedConfig.from_pretrained(str(checkpoint))
    config.device = device
    policy = PI05Policy.from_pretrained(str(checkpoint), config=config, strict=True)
    preprocessor = DataProcessorPipeline.from_pretrained(
        str(checkpoint),
        config_filename="policy_preprocessor.json",
        overrides={"device_processor": {"device": device}},
    )
    postprocessor = DataProcessorPipeline.from_pretrained(
        str(checkpoint),
        config_filename="policy_postprocessor.json",
        overrides={"device_processor": {"device": "cpu"}},
    )
    return policy.eval(), preprocessor, postprocessor


def build_rlinf_model(args: argparse.Namespace, checkpoint: Path, action_dim: int, chunk_size: int):
    openpi_cfg: dict[str, Any] = {
        "config_name": args.rlinf_config_name,
        "action_chunk": args.rlinf_num_action_chunks or chunk_size,
        "action_env_dim": args.rlinf_env_action_dim or action_dim,
    }
    if args.num_steps is not None:
        openpi_cfg["num_steps"] = args.num_steps
    if args.rlinf_train_expert_only is not None:
        openpi_cfg["train_expert_only"] = args.rlinf_train_expert_only == "true"

    cfg = OmegaConf.create(
        {
            "model_path": str(checkpoint),
            "precision": None,
            "openpi": openpi_cfg,
            "openpi_data": {
                "env_action_dim": args.rlinf_env_action_dim or action_dim,
            },
        }
    )
    model = get_rlinf_openpi_model(cfg)
    model.eval()
    return model


def build_lerobot_batch(sample: dict[str, Any], device: str) -> dict[str, Any]:
    main = ensure_hwc_uint8(sample["main_image"])
    left = ensure_hwc_uint8(sample["left_wrist_image"]) if sample.get("left_wrist_image") is not None else None
    right = (
        ensure_hwc_uint8(sample["right_wrist_image"]) if sample.get("right_wrist_image") is not None else None
    )
    state = torch.from_numpy(np.asarray(sample["state"], dtype=np.float32))[None, :]
    batch = {
        "observation.images.cam_high": hwc_to_chw_float01(main)[None].to(device),
        "observation.state": state.to(device),
        "task": [sample["prompt"]],
    }
    if left is not None:
        batch["observation.images.cam_left_wrist"] = hwc_to_chw_float01(left)[None].to(device)
    if right is not None:
        batch["observation.images.cam_right_wrist"] = hwc_to_chw_float01(right)[None].to(device)
    return batch


def build_rlinf_env_obs(sample: dict[str, Any]) -> dict[str, Any]:
    main = ensure_hwc_uint8(sample["main_image"])
    left = ensure_hwc_uint8(sample["left_wrist_image"]) if sample.get("left_wrist_image") is not None else None
    right = (
        ensure_hwc_uint8(sample["right_wrist_image"]) if sample.get("right_wrist_image") is not None else None
    )
    wrist = None
    if left is not None and right is not None:
        wrist = np.stack([left, right], axis=1)
    elif left is not None:
        wrist = left[:, None, ...]
    elif right is not None:
        wrist = right[:, None, ...]

    env_obs = {
        "main_images": hwc_batch_to_torch(main[None]),
        "wrist_images": None if wrist is None else hwc_batch_to_torch(wrist),
        "extra_view_images": None,
        "states": torch.from_numpy(np.asarray(sample["state"], dtype=np.float32))[None, :],
        "task_descriptions": [sample["prompt"]],
    }
    return env_obs


def capture_lerobot_inputs(
    preprocessor: DataProcessorPipeline, batch: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    transitions = list(preprocessor.step_through(batch))
    snapshots: list[dict[str, Any]] = []
    for idx, transition in enumerate(transitions):
        batch_view = preprocessor.to_output(transition)
        snapshot = {"step_index": idx}
        if "observation.state" in batch_view:
            snapshot["observation.state"] = batch_view["observation.state"]
        if OBS_LANGUAGE_TOKENS in batch_view:
            snapshot[OBS_LANGUAGE_TOKENS] = batch_view[OBS_LANGUAGE_TOKENS]
        if OBS_LANGUAGE_ATTENTION_MASK in batch_view:
            snapshot[OBS_LANGUAGE_ATTENTION_MASK] = batch_view[OBS_LANGUAGE_ATTENTION_MASK]
        snapshots.append(snapshot)
    return preprocessor(batch), snapshots


def extract_lerobot_raw_and_denorm(
    policy: PI05Policy,
    preprocessor: DataProcessorPipeline,
    postprocessor: DataProcessorPipeline,
    batch: dict[str, Any],
    seed: int,
    num_steps: int | None,
) -> dict[str, Any]:
    processed_batch, snapshots = capture_lerobot_inputs(preprocessor, batch)
    images, img_masks = policy._preprocess_images(processed_batch)
    tokens = processed_batch[OBS_LANGUAGE_TOKENS]
    masks = processed_batch[OBS_LANGUAGE_ATTENTION_MASK]

    torch.manual_seed(seed)
    noise_shape = (tokens.shape[0], policy.config.chunk_size, policy.config.max_action_dim)
    noise = torch.randn(noise_shape, dtype=torch.float32, device=tokens.device)

    with torch.no_grad():
        raw_actions = policy.model.sample_actions(
            images,
            img_masks,
            tokens,
            masks,
            noise=noise,
            num_steps=num_steps,
        )
    raw_actions = raw_actions[:, :, : policy.config.output_features["action"].shape[0]]
    denorm_actions = postprocessor({"action": raw_actions.detach().cpu()})["action"]
    return {
        "processed_batch": processed_batch,
        "snapshots": snapshots,
        "noise": noise.detach().cpu(),
        "raw_actions": raw_actions.detach().cpu(),
        "denorm_actions": denorm_actions.detach().cpu(),
    }


def extract_rlinf_raw_and_denorm(
    model: Any,
    env_obs: dict[str, Any],
    seed: int,
    num_steps: int | None,
) -> dict[str, Any]:
    to_process_obs = model.obs_processor(env_obs)
    processed_obs = model.input_transform(to_process_obs, transpose=False)
    processed_obs = model.precision_processor(processed_obs)

    observation = model.obs_processor(env_obs)
    observation = model.input_transform(observation, transpose=False)
    observation = model.precision_processor(observation)

    from openpi.models import model as openpi_model

    obs_obj = openpi_model.Observation.from_dict(observation)

    torch.manual_seed(seed)
    noise_shape = (
        obs_obj.state.shape[0],
        model.config.action_horizon,
        model.config.action_dim,
    )
    noise = torch.randn(noise_shape, dtype=torch.float32, device=obs_obj.state.device)

    with torch.no_grad():
        outputs = model.sample_actions(
            obs_obj,
            noise=noise,
            mode="eval",
            compute_values=False,
        )
    raw_actions = outputs["actions"].detach().cpu()
    denorm_actions = model.output_transform(
        {"actions": outputs["actions"], "state": obs_obj.state}
    )["actions"].detach().cpu()
    return {
        "to_process_obs": to_process_obs,
        "processed_obs": processed_obs,
        "noise": noise.detach().cpu(),
        "raw_actions": raw_actions,
        "denorm_actions": denorm_actions,
        "denoise_inds": outputs["denoise_inds"].detach().cpu(),
    }


def normalize_for_json(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: normalize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalize_for_json(v) for v in obj]
    if is_dataclass(obj):
        return normalize_for_json(asdict(obj))
    return obj


def main() -> int:
    args = parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    sample = load_sample(args)

    checkpoint_config = json.loads((checkpoint / "config.json").read_text())
    action_dim = int(checkpoint_config["output_features"]["action"]["shape"][0])
    chunk_size = int(checkpoint_config["chunk_size"])
    default_num_steps = int(checkpoint_config["num_inference_steps"])
    num_steps = args.num_steps if args.num_steps is not None else default_num_steps

    print("=== Config Summary ===")
    print(f"checkpoint: {checkpoint}")
    print(f"device: {args.device}")
    print(f"checkpoint action_dim: {action_dim}")
    print(f"checkpoint chunk_size: {chunk_size}")
    print(f"checkpoint num_inference_steps: {default_num_steps}")
    print(f"comparison num_steps: {num_steps}")
    print(f"RLinf config_name: {args.rlinf_config_name}")

    lerobot_policy, lerobot_pre, lerobot_post = load_lerobot_policy_and_processors(
        checkpoint, args.device
    )
    rlinf_model = build_rlinf_model(args, checkpoint, action_dim, chunk_size)

    lerobot_batch = build_lerobot_batch(sample, args.device)
    rlinf_env_obs = build_rlinf_env_obs(sample)

    lerobot_out = extract_lerobot_raw_and_denorm(
        lerobot_policy,
        lerobot_pre,
        lerobot_post,
        lerobot_batch,
        seed=args.seed,
        num_steps=num_steps,
    )
    rlinf_out = extract_rlinf_raw_and_denorm(
        rlinf_model,
        rlinf_env_obs,
        seed=args.seed,
        num_steps=num_steps,
    )

    print("\n=== LeRobot Preprocessed Inputs ===")
    print_tensor_summary(
        "lerobot processed observation.state",
        lerobot_out["processed_batch"].get("observation.state"),
    )
    print_tensor_summary(
        "lerobot tokenized_prompt",
        lerobot_out["processed_batch"].get(OBS_LANGUAGE_TOKENS),
    )
    print_tensor_summary(
        "lerobot tokenized_prompt_mask",
        lerobot_out["processed_batch"].get(OBS_LANGUAGE_ATTENTION_MASK),
    )

    print("\n=== RLinf Preprocessed Inputs ===")
    print_tensor_summary("rlinf processed state", rlinf_out["processed_obs"].get("state"))
    print_tensor_summary(
        "rlinf tokenized_prompt",
        rlinf_out["processed_obs"].get("tokenized_prompt"),
    )
    print_tensor_summary(
        "rlinf tokenized_prompt_mask",
        rlinf_out["processed_obs"].get("tokenized_prompt_mask"),
    )

    print("\n=== Input Diffs ===")
    input_diffs: dict[str, Any] = {}
    if (
        "observation.state" in lerobot_out["processed_batch"]
        and "state" in rlinf_out["processed_obs"]
        and tuple(lerobot_out["processed_batch"]["observation.state"].shape)
        == tuple(rlinf_out["processed_obs"]["state"].shape)
    ):
        input_diffs["state"] = print_diff_summary(
            "processed state diff",
            lerobot_out["processed_batch"]["observation.state"],
            rlinf_out["processed_obs"]["state"],
        )
    if (
        OBS_LANGUAGE_TOKENS in lerobot_out["processed_batch"]
        and "tokenized_prompt" in rlinf_out["processed_obs"]
        and tuple(lerobot_out["processed_batch"][OBS_LANGUAGE_TOKENS].shape)
        == tuple(rlinf_out["processed_obs"]["tokenized_prompt"].shape)
    ):
        input_diffs["tokenized_prompt"] = print_diff_summary(
            "processed tokenized_prompt diff",
            lerobot_out["processed_batch"][OBS_LANGUAGE_TOKENS],
            rlinf_out["processed_obs"]["tokenized_prompt"],
        )
    if (
        OBS_LANGUAGE_ATTENTION_MASK in lerobot_out["processed_batch"]
        and "tokenized_prompt_mask" in rlinf_out["processed_obs"]
        and tuple(lerobot_out["processed_batch"][OBS_LANGUAGE_ATTENTION_MASK].shape)
        == tuple(rlinf_out["processed_obs"]["tokenized_prompt_mask"].shape)
    ):
        input_diffs["tokenized_prompt_mask"] = print_diff_summary(
            "processed tokenized_prompt_mask diff",
            lerobot_out["processed_batch"][OBS_LANGUAGE_ATTENTION_MASK],
            rlinf_out["processed_obs"]["tokenized_prompt_mask"],
        )

    print("\n=== Raw Actions ===")
    print_tensor_summary("lerobot raw_actions", lerobot_out["raw_actions"])
    print_tensor_summary("rlinf raw_actions", rlinf_out["raw_actions"])

    print("\n=== Denormalized Actions ===")
    print_tensor_summary("lerobot denorm_actions", lerobot_out["denorm_actions"])
    print_tensor_summary("rlinf denorm_actions", rlinf_out["denorm_actions"])

    print("\n=== Action Diffs ===")
    action_diffs: dict[str, Any] = {}
    common_raw = min(
        lerobot_out["raw_actions"].shape[1],
        rlinf_out["raw_actions"].shape[1],
    )
    common_denorm = min(
        lerobot_out["denorm_actions"].shape[1],
        rlinf_out["denorm_actions"].shape[1],
    )
    common_dim = min(
        lerobot_out["raw_actions"].shape[2],
        rlinf_out["raw_actions"].shape[2],
    )
    action_diffs["raw_actions_common_prefix"] = print_diff_summary(
        "raw action diff",
        lerobot_out["raw_actions"][:, :common_raw, :common_dim],
        rlinf_out["raw_actions"][:, :common_raw, :common_dim],
    )
    action_diffs["denorm_actions_common_prefix"] = print_diff_summary(
        "denorm action diff",
        lerobot_out["denorm_actions"][:, :common_denorm, :common_dim],
        rlinf_out["denorm_actions"][:, :common_denorm, :common_dim],
    )

    results = {
        "config": {
            "checkpoint": str(checkpoint),
            "device": args.device,
            "seed": args.seed,
            "num_steps": num_steps,
            "rlinf_config_name": args.rlinf_config_name,
        },
        "lerobot": {
            "processed_state": summarize_tensor(
                "lerobot processed observation.state",
                lerobot_out["processed_batch"].get("observation.state"),
            ),
            "tokenized_prompt": summarize_tensor(
                "lerobot tokenized_prompt",
                lerobot_out["processed_batch"].get(OBS_LANGUAGE_TOKENS),
            ),
            "tokenized_prompt_mask": summarize_tensor(
                "lerobot tokenized_prompt_mask",
                lerobot_out["processed_batch"].get(OBS_LANGUAGE_ATTENTION_MASK),
            ),
            "raw_actions": summarize_tensor("lerobot raw_actions", lerobot_out["raw_actions"]),
            "denorm_actions": summarize_tensor(
                "lerobot denorm_actions", lerobot_out["denorm_actions"]
            ),
        },
        "rlinf": {
            "processed_state": summarize_tensor(
                "rlinf processed state", rlinf_out["processed_obs"].get("state")
            ),
            "tokenized_prompt": summarize_tensor(
                "rlinf tokenized_prompt", rlinf_out["processed_obs"].get("tokenized_prompt")
            ),
            "tokenized_prompt_mask": summarize_tensor(
                "rlinf tokenized_prompt_mask",
                rlinf_out["processed_obs"].get("tokenized_prompt_mask"),
            ),
            "raw_actions": summarize_tensor("rlinf raw_actions", rlinf_out["raw_actions"]),
            "denorm_actions": summarize_tensor("rlinf denorm_actions", rlinf_out["denorm_actions"]),
        },
        "diffs": {
            "inputs": input_diffs,
            "actions": action_diffs,
        },
        "raw": {
            "lerobot_snapshots": normalize_for_json(lerobot_out["snapshots"]),
            "rlinf_processed_obs": normalize_for_json(rlinf_out["processed_obs"]),
        },
    }

    if args.save_json:
        out_path = Path(args.save_json).resolve()
        out_path.write_text(json.dumps(normalize_for_json(results), indent=2))
        print(f"\nSaved JSON report to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
