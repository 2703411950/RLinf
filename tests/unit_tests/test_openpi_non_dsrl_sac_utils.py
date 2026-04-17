# Copyright 2026 The RLinf Authors.

"""Unit tests for PI05 non-DSRL SAC validation helpers."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from toolkits.openpi_non_dsrl_sac.common import (
    extract_policy_obs_from_batch,
    infer_action_layout,
    nested_tensor_shapes,
    tensor_stats,
)


def test_infer_action_layout_divisible():
    layout = infer_action_layout(700, 14)
    assert layout["is_divisible"]
    assert layout["action_chunk"] == 50


def test_infer_action_layout_non_divisible():
    layout = infer_action_layout(701, 14)
    assert not layout["is_divisible"]
    assert layout["action_chunk"] == -1


def test_extract_policy_obs_prefers_extra_view_as_wrist():
    batch = {
        "curr_obs": {
            "main_images": torch.zeros(2, 480, 640, 3, dtype=torch.uint8),
            "extra_view_images": torch.zeros(2, 2, 480, 640, 3, dtype=torch.uint8),
            "states": torch.zeros(2, 14),
        }
    }
    obs = extract_policy_obs_from_batch(batch)
    assert set(obs.keys()) == {"main_images", "wrist_images", "states"}
    assert list(obs["wrist_images"].shape) == [2, 2, 480, 640, 3]


def test_nested_tensor_shapes_and_stats():
    data = {
        "a": torch.ones(2, 3),
        "b": {"c": torch.zeros(4, dtype=torch.float32)},
    }
    shapes = nested_tensor_shapes(data)
    assert shapes == {"a": [2, 3], "b.c": [4]}

    stats = tensor_stats(torch.tensor([1.0, 2.0, 3.0]))
    assert stats["shape"] == [3]
    assert stats["min"] == 1.0
    assert stats["max"] == 3.0
