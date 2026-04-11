#!/usr/bin/env python3

"""Generate a tiny synthetic single-frame PI05 sample for cross-stack debugging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Output npz path.")
    parser.add_argument("--state-dim", type=int, default=14)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default="unscrew the bottle cap")
    return parser.parse_args()


def make_image(height: int, width: int, offset: int) -> np.ndarray:
    y = np.linspace(0, 255, height, dtype=np.float32)[:, None]
    x = np.linspace(0, 255, width, dtype=np.float32)[None, :]
    r = (x + offset) % 256
    g = (y + offset * 2) % 256
    b = ((x * 0.35 + y * 0.65) + offset * 3) % 256
    img = np.stack([r.repeat(height, axis=0), g.repeat(width, axis=1), b], axis=-1)
    return img.astype(np.uint8)


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    state = rng.uniform(low=-1.0, high=1.0, size=(args.state_dim,)).astype(np.float32)

    sample = {
        "main_image": make_image(args.height, args.width, offset=0),
        "left_wrist_image": make_image(args.height, args.width, offset=17),
        "right_wrist_image": make_image(args.height, args.width, offset=33),
        "state": state,
        "prompt": np.asarray(args.prompt),
    }

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **sample)

    meta = {
        "output": str(out_path),
        "state_dim": args.state_dim,
        "image_shape": [args.height, args.width, 3],
        "seed": args.seed,
        "prompt": args.prompt,
        "state_head": state[:8].tolist(),
    }
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
