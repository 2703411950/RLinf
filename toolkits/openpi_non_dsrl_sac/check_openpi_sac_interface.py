#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
"""Smoke-test the OpenPI SAC interface against real PI05 replay data.

Current expected state:
- After enabling ``use_non_dsrl_sac=True``, the script should pass on real PI05 replay data.
- Use ``--expect-failure=true`` only when bisecting regressions before the side branch is wired up.
"""

from __future__ import annotations

import argparse
import json

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
    tensor_stats,
)


def _parse_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "y"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", default=DEFAULT_REPLAY_DIR)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--env-action-dim", type=int, default=DEFAULT_ENV_ACTION_DIM)
    parser.add_argument(
        "--expect-failure",
        default="false",
        help="Whether non-DSRL SAC forward is expected to raise.",
    )
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
    model = get_model(cfg).eval()

    report = {
        "replay_dir": args.replay_dir,
        "batch_size": args.batch_size,
        "expect_failure": expect_failure,
    }

    try:
        sampled_actions, log_pi, shared_feature = model(
            forward_type=ForwardType.SAC,
            obs=policy_obs,
            mode="eval",
        )
        q_values = model(
            forward_type=ForwardType.SAC_Q,
            obs=policy_obs,
            actions=behavior_actions,
            train=False,
        )
    except ValueError as exc:
        report["status"] = "raised"
        report["error_type"] = type(exc).__name__
        report["error_message"] = str(exc)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if expect_failure and "use_dsrl=False" in str(exc):
            return
        raise

    report["status"] = "passed"
    report["sampled_actions"] = tensor_stats(sampled_actions)
    report["log_pi"] = tensor_stats(log_pi)
    report["shared_feature_present"] = shared_feature is not None
    report["q_values"] = tensor_stats(q_values)
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if expect_failure:
        raise SystemExit(
            "SAC interface already passed while --expect-failure=true. Switch the flag to false and tighten the assertions."
        )


if __name__ == "__main__":
    main()
