#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
"""Single-batch overfit skeleton for PI05 non-DSRL offline SAC.

This script is intentionally staged:
1. Today it proves the current failure mode on the real replay buffer.
2. After non-DSRL SAC is implemented, fill in the ``TODO(agent)`` sections and
   switch ``--expect-failure`` to ``false``.
"""

from __future__ import annotations

import argparse
import json

import torch

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.openpi import get_model
from toolkits.openpi_non_dsrl_sac.common import (
    DEFAULT_ENV_ACTION_DIM,
    DEFAULT_MODEL_PATH,
    DEFAULT_REPLAY_DIR,
    build_minimal_openpi_cfg,
    extract_behavior_actions_from_batch,
    extract_policy_obs_from_batch,
    sample_batch,
)


def _parse_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "y"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", default=DEFAULT_REPLAY_DIR)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--env-action-dim", type=int, default=DEFAULT_ENV_ACTION_DIM)
    parser.add_argument("--expect-failure", default="false")
    args = parser.parse_args()
    expect_failure = _parse_bool(args.expect_failure)

    batch = sample_batch(args.replay_dir, args.batch_size)
    policy_obs = extract_policy_obs_from_batch(batch)
    behavior_actions = extract_behavior_actions_from_batch(batch)
    cfg = build_minimal_openpi_cfg(
        model_path=args.model_path,
        env_action_dim=args.env_action_dim,
        use_dsrl=False,
        use_non_dsrl_sac=True,
    )
    model = get_model(cfg).train()

    report = {
        "batch_size": args.batch_size,
        "steps": args.steps,
        "expect_failure": expect_failure,
    }

    try:
        model(forward_type=ForwardType.SAC, obs=policy_obs, mode="train")
    except ValueError as exc:
        report["status"] = "raised"
        report["error_message"] = str(exc)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if expect_failure and "use_dsrl=False" in str(exc):
            return
        raise

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    losses: list[float] = []
    for _step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        sampled_actions, log_pi, _ = model(
            forward_type=ForwardType.SAC,
            obs=policy_obs,
            mode="train",
        )
        q_values = model(
            forward_type=ForwardType.SAC_Q,
            obs=policy_obs,
            actions=behavior_actions,
            train=True,
        )
        # TODO(agent): replace this placeholder objective with the real non-DSRL
        # SAC actor + critic losses once the model path is implemented.
        loss = (sampled_actions.float() ** 2).mean() - q_values.float().mean() + log_pi.float().mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    report["status"] = "passed"
    report["losses"] = losses
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if expect_failure:
        raise SystemExit(
            "Overfit skeleton unexpectedly passed under --expect-failure=true. Tighten the script and switch the flag to false."
        )


if __name__ == "__main__":
    main()
