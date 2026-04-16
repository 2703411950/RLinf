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

import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional
from torchvision.transforms import v2
from torchvision.transforms.functional import to_tensor

from rlinf.models.embodiment.base_policy import BasePolicy
from rlinf.utils.logging import get_logger


def resize_with_pad(img: torch.Tensor, size: int) -> torch.Tensor:
    h, w = functional.get_image_size(img)[::-1]
    scale = size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img = v2.Resize((new_h, new_w))(img)
    return functional.pad(
        img,
        [
            (size - new_w) // 2,
            (size - new_h) // 2,
            (size - new_w + 1) // 2,
            (size - new_h + 1) // 2,
        ],
        fill=0,
    )


def make_transform(size: int):
    return v2.Compose([lambda img: resize_with_pad(img, size)])


def pad_to_dim(x: torch.Tensor, target_dim: int, value: float = 0.0):
    current_dim = x.shape[-1]
    mask = torch.ones_like(x)
    if current_dim < target_dim:
        pad_dim = target_dim - current_dim
        x = F.pad(x, (0, pad_dim), mode="constant", value=value)
        mask = F.pad(mask, (0, pad_dim), mode="constant", value=0.0)
    return x, mask


def normalize_and_pad(data: torch.Tensor, norm_stats_key: dict[str, Any], max_dim: int):
    data_max = torch.tensor(norm_stats_key["max"], dtype=torch.float32, device=data.device)
    data_min = torch.tensor(norm_stats_key["min"], dtype=torch.float32, device=data.device)
    normalized_data = 2 * (data - data_min) / (data_max - data_min + 1e-6) - 1
    return pad_to_dim(normalized_data, max_dim)


def unnormalize_and_unpad(
    data: torch.Tensor, norm_stats_key: dict[str, Any], original_dim: int
) -> torch.Tensor:
    data_max = torch.tensor(norm_stats_key["max"], dtype=torch.float32, device=data.device)
    data_min = torch.tensor(norm_stats_key["min"], dtype=torch.float32, device=data.device)
    unpadded = data[..., :original_dim]
    return (unpadded + 1) * (data_max - data_min + 1e-6) / 2 + data_min


@dataclass
class Observation:
    image: Any
    state: Any
    prompt: Any
    wrist_images: Any = None
    extra_view_images: Any = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]):
        return cls(
            image=d.get("image"),
            state=d.get("state"),
            prompt=d.get("prompt"),
            wrist_images=d.get("wrist_images"),
            extra_view_images=d.get("extra_view_images"),
        )


class MantisActionModel(nn.Module, BasePolicy):
    def __init__(self, config, torch_dtype=torch.bfloat16):
        super().__init__()
        self.config = config
        self.torch_dtype = torch_dtype
        self.logger = get_logger()

        mantis_cfg = getattr(config, "mantis", config)
        self.mantis_cfg = mantis_cfg
        self.model_path = str(config.model_path)
        self.num_action_chunks = int(getattr(config, "num_action_chunks", 32))
        self.action_dim = int(getattr(config, "action_dim", 14))
        self.state_dim = int(getattr(config, "state_dim", 14))
        self.primary_image_size = int(getattr(mantis_cfg, "primary_image_size", 256))
        self.auxiliary_image_size = int(getattr(mantis_cfg, "auxiliary_image_size", 256))
        self.enable_fast_action = bool(getattr(mantis_cfg, "enable_fast_action", True))
        self.instruction = str(getattr(mantis_cfg, "instruction", "robot manipulation"))
        self.unnorm_key = str(getattr(mantis_cfg, "unnorm_key", "piper_put_object_into_bag"))
        self.norm_stats_path = str(
            getattr(
                mantis_cfg,
                "norm_stats_path",
                "/data/cyy/control_your_robot/src/robot/policy/Mantis/mantis_models/norm_stats.json",
            )
        )
        self.source_root = str(
            getattr(mantis_cfg, "source_root", "/data/cyy/piper-vla")
        )
        self.backbone_path = getattr(
            mantis_cfg, "backbone_path", "/data/cyy/control_your_robot/.cache/models/RynnBrain-2B"
        )
        self.diffusion_model_path = getattr(
            mantis_cfg,
            "diffusion_model_path",
            "/data/cyy/control_your_robot/.cache/models/Sana_600M_512px_diffusers_64channels",
        )
        self.ignore_mismatched_sizes = bool(
            getattr(mantis_cfg, "ignore_mismatched_sizes", True)
        )
        self.input_size = int(getattr(mantis_cfg, "input_size", 16))

        if self.source_root not in sys.path:
            sys.path.insert(0, self.source_root)

        from inference.mantis_fast.models import Mantis, MantisConfig

        self._mantis_model_cls = Mantis
        self._mantis_config_cls = MantisConfig

        with open(os.path.join(self.model_path, "config.json"), "r") as f:
            raw_model_cfg = json.load(f)
        if self.backbone_path:
            raw_model_cfg["mllm_id"] = self.backbone_path
        if self.diffusion_model_path:
            raw_model_cfg["vae_id"] = self.diffusion_model_path
            raw_model_cfg["noise_scheduler_id"] = self.diffusion_model_path
            raw_model_cfg["scheduler_id"] = self.diffusion_model_path
            raw_model_cfg["diffusion_model_id"] = self.diffusion_model_path
        self.max_state_dim = int(raw_model_cfg.get("max_state_dim", 32) or 32)
        self.max_action_dim = int(raw_model_cfg.get("max_action_dim", 32) or 32)
        self.model_chunk_size = int(raw_model_cfg.get("chunk_size", self.num_action_chunks) or self.num_action_chunks)
        self.num_action_chunks = min(self.num_action_chunks, self.model_chunk_size)

        mantis_model_cfg = self._mantis_config_cls(**raw_model_cfg)
        self.policy = self._mantis_model_cls.from_pretrained(
            self.model_path,
            config=mantis_model_cfg,
            input_size=self.input_size,
            ignore_mismatched_sizes=self.ignore_mismatched_sizes,
        )
        if hasattr(self.policy.model, "transformer"):
            del self.policy.model.transformer
        if hasattr(self.policy.model, "connector"):
            del self.policy.model.connector

        self.policy.to(dtype=self.torch_dtype)
        self.policy.eval()
        if self.enable_fast_action:
            self.policy.enable_fast_inference()

        with open(self.norm_stats_path, "r") as f:
            norm_stats = json.load(f)
        self.norm_stats = norm_stats[self.unnorm_key]

        self.primary_image_transform = make_transform(self.primary_image_size)
        self.auxiliary_image_transform = make_transform(self.auxiliary_image_size)

        self.logger.info(
            "[Mantis] Loaded model_path=%s chunk=%s action_dim=%s state_dim=%s unnorm_key=%s",
            self.model_path,
            self.num_action_chunks,
            self.action_dim,
            self.state_dim,
            self.unnorm_key,
        )

    def gradient_checkpointing_enable(self, **kwargs):
        return None

    def default_forward(self, **kwargs):
        raise NotImplementedError("Mantis RL training forward is not implemented yet.")

    @staticmethod
    def _to_hwc_uint8(image: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(image, torch.Tensor):
            arr = image.detach().cpu().numpy()
        else:
            arr = np.asarray(image)
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D image, got shape {arr.shape}")
        if arr.shape[0] == 3 and arr.shape[-1] != 3:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(arr)

    def obs_processor(self, env_obs: dict[str, Any]) -> dict[str, Any]:
        return {
            "image": env_obs.get("main_images"),
            "state": env_obs.get("states"),
            "prompt": env_obs.get("task_descriptions", env_obs.get("prompt")),
            "wrist_images": env_obs.get("wrist_images"),
            "extra_view_images": env_obs.get("extra_view_images"),
        }

    def _build_input_images(self, observation: Observation) -> list[list[torch.Tensor]]:
        main_images = observation.image
        wrist_images = observation.wrist_images
        extra_view_images = observation.extra_view_images
        batch_size = main_images.shape[0]
        image_list: list[list[torch.Tensor]] = []

        for idx in range(batch_size):
            main = to_tensor(self._to_hwc_uint8(main_images[idx]))
            left = main
            right = main
            if wrist_images is not None:
                left = to_tensor(self._to_hwc_uint8(wrist_images[idx]))
            elif extra_view_images is not None and extra_view_images.shape[1] > 0:
                left = to_tensor(self._to_hwc_uint8(extra_view_images[idx, 0]))
            if extra_view_images is not None and extra_view_images.shape[1] > 1:
                right = to_tensor(self._to_hwc_uint8(extra_view_images[idx, 1]))
            elif wrist_images is not None:
                right = left

            # Match control_your_robot Mantis inference order:
            # history, main, left_wrist, right_wrist.
            image_list.append(
                [
                    self.auxiliary_image_transform(main),
                    self.primary_image_transform(main),
                    self.auxiliary_image_transform(left),
                    self.auxiliary_image_transform(right),
                ]
            )
        return image_list

    def _build_states(self, observation: Observation, device: torch.device) -> torch.Tensor:
        states = observation.state
        if not isinstance(states, torch.Tensor):
            states = torch.as_tensor(states, dtype=torch.float32)
        states = states.to(device=device, dtype=torch.float32)
        states = states[..., : self.state_dim]
        normalized_states, _ = normalize_and_pad(
            states,
            self.norm_stats["observation.state"],
            self.max_state_dim,
        )
        return normalized_states.to(dtype=self.torch_dtype)

    def _build_prompts(self, observation: Observation, batch_size: int) -> list[str]:
        prompt = observation.prompt
        if prompt is None:
            return [self.instruction] * batch_size
        if isinstance(prompt, str):
            return [prompt] * batch_size
        return [str(p) for p in prompt]

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs,
        mode: Literal["train", "eval"] = "eval",
        compute_values: bool = False,
        **kwargs,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del mode, compute_values, kwargs
        processed_obs = self.obs_processor(env_obs)
        observation = Observation.from_dict(processed_obs)

        device = next(self.parameters()).device
        prompts = self._build_prompts(observation, observation.state.shape[0])
        input_images = self._build_input_images(observation)
        normalized_states = self._build_states(observation, device)

        sample_fn = (
            self.policy.sample_actions_fast
            if self.enable_fast_action
            else self.policy.sample_actions
        )
        pred_actions = sample_fn(
            caption=prompts,
            input_images=input_images,
            states=normalized_states,
            num_images_per_prompt=1,
        )
        pred_actions = torch.as_tensor(pred_actions, dtype=torch.float32, device=device)
        if pred_actions.ndim == 2:
            pred_actions = pred_actions.unsqueeze(0)

        raw_normalized_actions = pred_actions[:, : self.num_action_chunks, : self.max_action_dim]
        env_actions = unnormalize_and_unpad(
            raw_normalized_actions,
            self.norm_stats["action"],
            self.action_dim,
        )
        env_actions = env_actions[:, : self.num_action_chunks, : self.action_dim]

        self.logger.info(
            "[Mantis] raw normalized action stats: shape=%s min=%.4f max=%.4f mean=%.4f std=%.4f",
            tuple(raw_normalized_actions.shape),
            raw_normalized_actions.min().item(),
            raw_normalized_actions.max().item(),
            raw_normalized_actions.mean().item(),
            raw_normalized_actions.std().item(),
        )
        self.logger.info(
            "[Mantis] env action stats: shape=%s min=%.4f max=%.4f mean=%.4f std=%.4f",
            tuple(env_actions.shape),
            env_actions.min().item(),
            env_actions.max().item(),
            env_actions.mean().item(),
            env_actions.std().item(),
        )

        result = {
            "prev_logprobs": torch.zeros(env_actions.shape[:2], dtype=torch.float32),
            "prev_values": torch.zeros(env_actions.shape[:2], dtype=torch.float32),
            "forward_inputs": {
                "action": env_actions.detach().cpu(),
                "raw_normalized_action": raw_normalized_actions.detach().cpu(),
                "normalized_state": normalized_states.detach().cpu(),
            },
        }
        return env_actions.detach().cpu().numpy(), result
