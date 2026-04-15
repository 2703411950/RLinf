# Copyright 2025 The RLinf Authors.
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
# openpi model configs

import json
import os
from pathlib import Path

import torch
from omegaconf import DictConfig


def _load_lerobot_checkpoint_config(checkpoint_dir: str) -> dict | None:
    config_path = Path(checkpoint_dir) / "config.json"
    if not config_path.exists():
        return None
    config = json.loads(config_path.read_text())
    if config.get("policy_type") != "pi05" and config.get("type") != "pi05":
        return None
    return config


def _load_lerobot_norm_stats(checkpoint_dir: str) -> dict | None:
    from openpi.shared import normalize as _normalize
    from safetensors import safe_open

    stats_path = Path(checkpoint_dir) / "policy_preprocessor_step_3_normalizer_processor.safetensors"
    if not stats_path.exists():
        return None

    def _read_stats(prefix: str) -> _normalize.NormStats:
        with safe_open(stats_path, framework="pt", device="cpu") as handle:
            return _normalize.NormStats(
                mean=handle.get_tensor(f"{prefix}.mean").numpy(),
                std=handle.get_tensor(f"{prefix}.std").numpy(),
                q01=handle.get_tensor(f"{prefix}.q01").numpy(),
                q99=handle.get_tensor(f"{prefix}.q99").numpy(),
            )

    return {
        "state": _read_stats("observation.state"),
        "actions": _read_stats("action"),
    }


def get_model(cfg: DictConfig, torch_dtype=None):
    import glob

    import openpi.shared.download as download
    import openpi.transforms as transforms
    import safetensors
    from safetensors.torch import load_file
    from openpi.training import checkpoints as _checkpoints

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from rlinf.models.embodiment.openpi.openpi_action_model import (
        OpenPi0Config,
        OpenPi0ForRLActionPrediction,
    )

    # config
    config_name = getattr(cfg.openpi, "config_name", None)
    data_kwargs = getattr(cfg, "openpi_data", None)
    actor_train_config = get_openpi_config(
        config_name, model_path=cfg.model_path, data_kwargs=data_kwargs
    )

    print(f"actor_train_config: {actor_train_config}")

    actor_model_config = actor_train_config.model
    actor_model_config = OpenPi0Config(**actor_model_config.__dict__)
    override_model_config_kwargs = cfg.openpi
    if override_model_config_kwargs is not None:
        for key, val in override_model_config_kwargs.items():
            actor_model_config.__dict__[key] = val

    # load model
    checkpoint_dir = download.maybe_download(str(cfg.model_path))
    lerobot_checkpoint_config = None
    if getattr(actor_model_config, "lerobot_compat", False):
        lerobot_checkpoint_config = _load_lerobot_checkpoint_config(checkpoint_dir)
        if lerobot_checkpoint_config is not None:
            object.__setattr__(
                actor_model_config, "action_horizon", lerobot_checkpoint_config["chunk_size"]
            )
            object.__setattr__(
                actor_model_config, "action_chunk", lerobot_checkpoint_config["n_action_steps"]
            )
            object.__setattr__(
                actor_model_config,
                "action_env_dim",
                lerobot_checkpoint_config["output_features"]["action"]["shape"][0],
            )
            object.__setattr__(
                actor_model_config, "num_steps", lerobot_checkpoint_config["num_inference_steps"]
            )
            object.__setattr__(
                actor_model_config,
                "tokenizer_max_length",
                lerobot_checkpoint_config["tokenizer_max_length"],
            )
            object.__setattr__(
                actor_model_config, "max_state_dim", lerobot_checkpoint_config["max_state_dim"]
            )
            object.__setattr__(
                actor_model_config, "max_action_dim", lerobot_checkpoint_config["max_action_dim"]
            )

    # Check if this is a checkpoint directory (saved by FSDP)
    # Check for model_state_dict/full_weights.pt (direct checkpoint) or actor/model_state_dict/full_weights.pt (from runner)
    full_weights_path = os.path.join(
        checkpoint_dir, "model_state_dict", "full_weights.pt"
    )
    actor_full_weights_path = os.path.join(
        checkpoint_dir, "actor", "model_state_dict", "full_weights.pt"
    )

    model: OpenPi0ForRLActionPrediction = OpenPi0ForRLActionPrediction(
        actor_model_config
    )
    # train expert only
    if actor_model_config.train_expert_only:
        model.freeze_vlm()

    # Load weights from checkpoint if it's a checkpoint directory, otherwise load from safetensors
    if os.path.exists(full_weights_path):
        # Direct checkpoint directory
        model_state_dict = torch.load(full_weights_path, map_location="cpu")
        model.load_state_dict(model_state_dict, strict=False)
    elif os.path.exists(actor_full_weights_path):
        # Checkpoint directory from runner
        model_state_dict = torch.load(actor_full_weights_path, map_location="cpu")
        model.load_state_dict(model_state_dict, strict=False)
    else:
        # Original model directory with safetensors files
        weight_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
        if not weight_paths:
            weight_paths = [os.path.join(checkpoint_dir, "model.safetensors")]
        for weight_path in weight_paths:
            if getattr(actor_model_config, "lerobot_compat", False):
                state_dict = load_file(weight_path)
                remapped_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith("model."):
                        remapped_state_dict[key[len("model.") :]] = value
                    else:
                        remapped_state_dict[key] = value
                model.load_state_dict(remapped_state_dict, strict=False)
            else:
                safetensors.torch.load_model(model, weight_path, strict=False)

    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    # fsdp replace
    # model.paligemma_with_expert.replace_gemma_decoder_layers()
    # load data stats
    data_config = actor_train_config.data.create(
        actor_train_config.assets_dirs, actor_model_config
    )
    norm_stats = None
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        try:
            norm_stats = _checkpoints.load_norm_stats(checkpoint_dir, data_config.asset_id)
        except FileNotFoundError:
            if not getattr(actor_model_config, "lerobot_compat", False):
                raise
            norm_stats = _load_lerobot_norm_stats(checkpoint_dir)
            if norm_stats is None:
                raise
    # wrappers
    repack_transforms = transforms.Group()
    default_prompt = None
    post_normalize_input_transforms = getattr(
        data_config, "post_normalize_input_transforms", ()
    )
    model.setup_wrappers(
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *post_normalize_input_transforms,
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
    )

    return model
