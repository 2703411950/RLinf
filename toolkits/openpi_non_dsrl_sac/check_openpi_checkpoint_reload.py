#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
"""Checkpoint reload skeleton for the future non-DSRL PI05 SAC actor."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.openpi import get_model
from toolkits.openpi_non_dsrl_sac.common import (
    DEFAULT_ENV_ACTION_DIM,
    DEFAULT_MODEL_PATH,
    DEFAULT_REPLAY_DIR,
    build_minimal_openpi_cfg,
    extract_policy_obs_from_batch,
    sample_batch,
)


def _parse_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "y"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", default=DEFAULT_REPLAY_DIR)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--env-action-dim", type=int, default=DEFAULT_ENV_ACTION_DIM)
    parser.add_argument("--expect-failure", default="false")
    args = parser.parse_args()
    expect_failure = _parse_bool(args.expect_failure)

    batch = sample_batch(args.replay_dir, batch_size=1)
    policy_obs = extract_policy_obs_from_batch(batch)
    cfg = build_minimal_openpi_cfg(
        model_path=args.model_path,
        env_action_dim=args.env_action_dim,
        use_dsrl=False,
        use_non_dsrl_sac=True,
    )

    model = get_model(cfg).eval()
    report = {"expect_failure": expect_failure}

    try:
        out_before = model(forward_type=ForwardType.SAC, obs=policy_obs, mode="eval")
    except ValueError as exc:
        report["status"] = "raised"
        report["error_message"] = str(exc)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if expect_failure and "use_dsrl=False" in str(exc):
            return
        raise

    with tempfile.TemporaryDirectory(prefix="pi05_non_dsrl_reload_") as tmp_dir:
        tmp_path = Path(tmp_dir) / "state_dict.pt"
        torch.save(model.state_dict(), tmp_path)
        reloaded = get_model(cfg).eval()
        reloaded.load_state_dict(torch.load(tmp_path, map_location="cpu"), strict=False)
        out_after = reloaded(forward_type=ForwardType.SAC, obs=policy_obs, mode="eval")

    actions_before = out_before[0]
    actions_after = out_after[0]
    diff = (actions_before - actions_after).abs().max().item()
    report["status"] = "passed"
    report["max_abs_diff"] = float(diff)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
