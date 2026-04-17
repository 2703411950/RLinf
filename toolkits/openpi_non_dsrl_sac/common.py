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

"""Common helpers for PI05 non-DSRL offline SAC validation."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from rlinf.data.replay_buffer import TrajectoryReplayBuffer

DEFAULT_REPLAY_DIR = "/data/cyy/RLinf/logs/20260416-22:38:11/demos"
DEFAULT_MODEL_PATH = "/data/cyy/ckpts/adk111/piper_pi05_SFT_right_place_cup_on_book_left_insert_30000"
DEFAULT_ENV_ACTION_DIM = 14
DEFAULT_ACTION_CHUNK = 50
DEFAULT_STATE_DIM = 14
DEFAULT_NUM_STEPS = 10


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    """Return a compact statistics summary for a tensor."""
    tensor = tensor.detach().cpu()
    if tensor.numel() == 0:
        return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "numel": 0}

    stats_tensor = tensor.float()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": int(tensor.numel()),
        "min": float(stats_tensor.min().item()),
        "max": float(stats_tensor.max().item()),
        "mean": float(stats_tensor.mean().item()),
        "std": float(stats_tensor.std(unbiased=False).item()),
    }


def nested_tensor_shapes(data: dict[str, Any], prefix: str = "") -> dict[str, list[int]]:
    """Flatten nested tensor keys into a shape summary."""
    out: dict[str, list[int]] = {}
    for key, value in data.items():
        flat_key = f"{prefix}.{key}" if prefix else key
        if torch.is_tensor(value):
            out[flat_key] = list(value.shape)
        elif isinstance(value, dict):
            out.update(nested_tensor_shapes(value, flat_key))
    return out


def infer_action_layout(flat_action_dim: int, env_action_dim: int) -> dict[str, int | bool]:
    """Infer whether flattened actions are divisible by the env action dimension."""
    if env_action_dim <= 0:
        raise ValueError(f"env_action_dim must be positive, got {env_action_dim}")
    divisible = flat_action_dim % env_action_dim == 0
    return {
        "flat_action_dim": int(flat_action_dim),
        "env_action_dim": int(env_action_dim),
        "is_divisible": divisible,
        "action_chunk": int(flat_action_dim // env_action_dim) if divisible else -1,
    }


def _validate_replay_dir(replay_dir: str | Path) -> Path:
    path = Path(replay_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"ReplayBuffer directory not found: {path}")
    for filename in ("metadata.json", "trajectory_index.json"):
        if not (path / filename).is_file():
            raise FileNotFoundError(f"Missing {filename} under replay dir: {path}")
    return path


def load_replay_buffer(replay_dir: str | Path) -> TrajectoryReplayBuffer:
    """Load a TrajectoryReplayBuffer checkpoint from disk."""
    replay_path = _validate_replay_dir(replay_dir)
    buffer = TrajectoryReplayBuffer(
        seed=1234,
        enable_cache=False,
        auto_save=False,
        trajectory_format="pt",
    )
    buffer.load_checkpoint(str(replay_path), is_distributed=False)
    return buffer


def load_trajectory(replay_dir: str | Path, trajectory_id: int = 0):
    """Load one trajectory from the replay buffer checkpoint."""
    buffer = load_replay_buffer(replay_dir)
    trajectory_info = buffer._trajectory_index[trajectory_id]
    model_weights_id = trajectory_info["model_weights_id"]
    return buffer._load_trajectory(trajectory_id, model_weights_id)


def sample_batch(replay_dir: str | Path, batch_size: int) -> dict[str, Any]:
    """Sample one transition batch from the replay buffer."""
    buffer = load_replay_buffer(replay_dir)
    return buffer.sample(batch_size)


def extract_policy_obs_from_batch(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Build the policy observation dict used by realworld PI05 inference.

    The current Piper replay stores wrist cameras inside ``extra_view_images``.
    For non-DSRL SAC adaptation we keep that exact observation contract.
    """
    curr_obs = batch["curr_obs"]
    obs = {
        "main_images": curr_obs["main_images"],
        "states": curr_obs["states"],
    }
    if "wrist_images" in curr_obs:
        obs["wrist_images"] = curr_obs["wrist_images"]
    elif "extra_view_images" in curr_obs:
        obs["wrist_images"] = curr_obs["extra_view_images"]
    return obs


def extract_behavior_actions_from_batch(batch: dict[str, Any]) -> torch.Tensor:
    """Return the action tensor that the offline actor should learn from."""
    if "actions" not in batch:
        raise KeyError("Sample batch does not contain 'actions'.")
    return batch["actions"]


def build_minimal_openpi_cfg(
    *,
    model_path: str = DEFAULT_MODEL_PATH,
    env_action_dim: int = DEFAULT_ENV_ACTION_DIM,
    action_chunk: int = DEFAULT_ACTION_CHUNK,
    state_dim: int = DEFAULT_STATE_DIM,
    num_steps: int = DEFAULT_NUM_STEPS,
    use_dsrl: bool = False,
    use_non_dsrl_sac: bool = True,
) -> Any:
    """Build the minimal config required by ``rlinf.models.embodiment.openpi.get_model``."""
    cfg = {
        "model_path": model_path,
        "state_dim": state_dim,
        "action_dim": env_action_dim,
        "num_action_chunks": action_chunk,
        "num_steps": 5,
        "openpi": {
            "config_name": "pi05_piper",
            "lerobot_compat": True,
            "action_chunk": action_chunk,
            "action_env_dim": env_action_dim,
            "num_steps": num_steps,
            "use_dsrl": use_dsrl,
            "use_non_dsrl_sac": use_non_dsrl_sac,
            "train_expert_only": True,
            "non_dsrl_state_dim": state_dim,
        },
        "openpi_data": {
            "align_evorl_observation": True,
            "env_action_dim": env_action_dim,
        },
    }
    return OmegaConf.create(cfg)


def format_command(command: Iterable[str]) -> str:
    """Join command parts for README output."""
    return " ".join(command)
