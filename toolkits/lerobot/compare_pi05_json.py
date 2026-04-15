#!/usr/bin/env python3

"""Compare LeRobot and RLinf PI05 JSON dumps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lerobot-json", required=True)
    parser.add_argument("--rlinf-json", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def diff_stats(a: np.ndarray, b: np.ndarray) -> dict:
    diff = a.astype(np.float64) - b.astype(np.float64)
    return {
        "shape_a": list(a.shape),
        "shape_b": list(b.shape),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
    }


def main() -> int:
    args = parse_args()
    lerobot = json.loads(Path(args.lerobot_json).read_text())
    rlinf = json.loads(Path(args.rlinf_json).read_text())

    results = {
        "processed_state": diff_stats(
            np.asarray(lerobot["raw_arrays"]["processed_state"]),
            np.asarray(rlinf["raw_arrays"]["processed_state"]),
        ),
        "tokenized_prompt": diff_stats(
            np.asarray(lerobot["raw_arrays"]["tokenized_prompt"]),
            np.asarray(rlinf["raw_arrays"]["tokenized_prompt"]),
        ),
        "tokenized_prompt_mask": diff_stats(
            np.asarray(lerobot["raw_arrays"]["tokenized_prompt_mask"]),
            np.asarray(rlinf["raw_arrays"]["tokenized_prompt_mask"]),
        ),
        "raw_actions": diff_stats(
            np.asarray(lerobot["raw_arrays"]["raw_actions"]),
            np.asarray(rlinf["raw_arrays"]["raw_actions"]),
        ),
        "denorm_actions": diff_stats(
            np.asarray(lerobot["raw_arrays"]["denorm_actions"]),
            np.asarray(rlinf["raw_arrays"]["denorm_actions"]),
        ),
    }

    out = Path(args.output_json).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
