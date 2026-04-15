#!/usr/bin/env python3

"""Run one PI05 sample through RLinf/OpenPI inference and dump JSON."""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample-npz", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--config-name", default="pi05_piper")
    parser.add_argument("--env-action-dim", type=int, default=None)
    parser.add_argument("--num-action-chunks", type=int, default=None)
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


def build_temp_model_view(checkpoint: Path, asset_id: str) -> Path:
    tmp_root = Path("/tmp/pi05_compare_logs/rlinf_model_view").resolve()
    tmp_root.mkdir(parents=True, exist_ok=True)
    for name in ["config.json", "model.safetensors"]:
        src = checkpoint / name
        dst = tmp_root / name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)

    def read_stats(safetensor_path: Path, prefix: str) -> dict[str, list[float]]:
        with safe_open(safetensor_path, framework="pt", device="cpu") as f:
            return {
                "mean": f.get_tensor(f"{prefix}.mean").tolist(),
                "std": f.get_tensor(f"{prefix}.std").tolist(),
                "q01": f.get_tensor(f"{prefix}.q01").tolist(),
                "q99": f.get_tensor(f"{prefix}.q99").tolist(),
            }

    norm_stats = {
        "state": read_stats(checkpoint / "policy_preprocessor_step_3_normalizer_processor.safetensors", "observation.state"),
        "actions": read_stats(checkpoint / "policy_preprocessor_step_3_normalizer_processor.safetensors", "action"),
    }
    asset_dir = tmp_root / asset_id
    asset_dir.mkdir(parents=True, exist_ok=True)
    (asset_dir / "norm_stats.json").write_text(json.dumps({"norm_stats": norm_stats}, indent=2))
    return tmp_root


def install_piper_stub_module(asset_id: str) -> None:
    import dataclasses

    import openpi.models.model as _model
    import openpi.transforms as _transforms
    from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
    from typing_extensions import override

    module_name = "rlinf.models.embodiment.openpi.dataconfig.piper_dataconfig"
    if module_name in sys.modules:
        return

    @dataclasses.dataclass(frozen=True)
    class LeRobotPiperDataConfig(DataConfigFactory):
        default_prompt: str | None = None
        env_action_dim: int = 14
        use_quantile_norm: bool = True
        override_use_quantile_norm: bool | None = None
        repack_transforms: _transforms.Group = dataclasses.field(
            default_factory=lambda: _transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "image": {
                                "cam_high": "observation/image",
                                "cam_left_wrist": "observation/wrist_image/0",
                                "cam_right_wrist": "observation/wrist_image/1",
                            },
                            "state": "observation/state",
                            "actions": "actions",
                            "prompt": "prompt",
                        }
                    )
                ]
            )
        )

        @override
        def create(self, assets_dirs: Path, model_config: _model.BaseModelConfig) -> DataConfig:
            model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
            use_quantile_norm = (
                self.override_use_quantile_norm
                if self.override_use_quantile_norm is not None
                else self.use_quantile_norm
            )
            return dataclasses.replace(
                self.create_base_config(assets_dirs, model_config),
                repack_transforms=self.repack_transforms,
                data_transforms=_transforms.Group(),
                model_transforms=model_transforms,
                action_sequence_keys=("action",),
                use_quantile_norm=use_quantile_norm,
                asset_id=asset_id,
            )

    def coerce_piper_data_kwargs(data_kwargs: dict | None) -> dict | None:
        return data_kwargs

    stub = types.ModuleType(module_name)
    stub.LeRobotPiperDataConfig = LeRobotPiperDataConfig
    stub.coerce_piper_data_kwargs = coerce_piper_data_kwargs
    sys.modules[module_name] = stub


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

    args = parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    with np.load(args.sample_npz, allow_pickle=True) as data:
        sample = {k: data[k] for k in data.files}

    from openpi.models import model as openpi_model
    from rlinf.models.embodiment.openpi import get_model as get_rlinf_openpi_model

    checkpoint_config = json.loads((checkpoint / "config.json").read_text())
    action_dim = int(checkpoint_config["output_features"]["action"]["shape"][0])
    chunk_size = int(checkpoint_config["chunk_size"])
    openpi_cfg = {
        "config_name": args.config_name,
        "action_chunk": args.num_action_chunks or chunk_size,
        "action_env_dim": args.env_action_dim or action_dim,
        "lerobot_compat": True,
    }
    if args.num_steps is not None:
        openpi_cfg["num_steps"] = args.num_steps

    cfg = OmegaConf.create(
        {
            "model_path": str(checkpoint),
            "precision": None,
            "openpi": openpi_cfg,
            "openpi_data": {"env_action_dim": args.env_action_dim or action_dim},
        }
    )
    model = get_rlinf_openpi_model(cfg).eval()
    model.to(args.device)

    main_image = ensure_hwc_uint8(sample["main_image"])
    left = ensure_hwc_uint8(sample["left_wrist_image"])
    right = ensure_hwc_uint8(sample["right_wrist_image"])
    prompt = str(sample["prompt"].tolist() if hasattr(sample["prompt"], "tolist") else sample["prompt"])

    to_process = {
        "observation/image": torch.from_numpy(main_image[None].copy()),
        "observation/wrist_image": torch.from_numpy(left[None].copy()),
        "observation/extra_view_image": torch.from_numpy(right[None].copy()),
        "observation/state": torch.from_numpy(np.asarray(sample["state"], dtype=np.float32))[None],
        "prompt": [prompt],
    }
    processed = model.input_transform(to_process, transpose=False)
    processed = model.precision_processor(processed)
    if "image" in processed and "image_mask" not in processed:
        processed["image_mask"] = {
            key: torch.ones(value.shape[0], dtype=torch.bool, device=value.device)
            for key, value in processed["image"].items()
        }
    observation = openpi_model.Observation.from_dict(processed)

    torch.manual_seed(args.seed)
    noise = torch.randn(
        (observation.state.shape[0], model.config.action_horizon, model.config.action_dim),
        dtype=torch.float32,
        device=observation.state.device,
    )

    with torch.no_grad():
        outputs = model.sample_actions(
            observation,
            noise=noise,
            mode="eval",
            compute_values=False,
        )
    raw_actions = outputs["actions"].detach().cpu()
    denorm_actions = model.output_transform(
        {"actions": outputs["actions"], "state": observation.state}
    )["actions"].detach().cpu()

    result = {
        "framework": "rlinf",
        "checkpoint": str(checkpoint),
        "sample_npz": str(Path(args.sample_npz).resolve()),
        "device": args.device,
        "seed": args.seed,
        "processed": {
            "state": summarize_tensor("state", processed["state"]),
            "tokenized_prompt": summarize_tensor("tokenized_prompt", processed["tokenized_prompt"]),
            "tokenized_prompt_mask": summarize_tensor(
                "tokenized_prompt_mask", processed["tokenized_prompt_mask"]
            ),
        },
        "raw_actions": summarize_tensor("raw_actions", raw_actions),
        "denorm_actions": summarize_tensor("denorm_actions", denorm_actions),
        "raw_arrays": {
            "processed_state": processed["state"].detach().cpu().tolist(),
            "tokenized_prompt": processed["tokenized_prompt"].detach().cpu().tolist(),
            "tokenized_prompt_mask": processed["tokenized_prompt_mask"].detach().cpu().tolist(),
            "raw_actions": raw_actions.tolist(),
            "denorm_actions": denorm_actions.tolist(),
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
