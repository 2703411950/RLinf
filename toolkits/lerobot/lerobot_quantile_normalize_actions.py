#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# LeRobot 数据集：用 meta/stats 中的 q01/q99 对每一帧的 action 做 QUANTILES 归一化
# （与常见 NormalizationMode.QUANTILES 一致：映射到约 [-1, 1]）。
#
# 用法示例：
#   python toolkits/lerobot/lerobot_quantile_normalize_actions.py \
#     --dataset /data/cyy/RLinf/datasets/testpiper \
#     --preview 5 --summary
#
#   python toolkits/lerobot/lerobot_quantile_normalize_actions.py \
#     --dataset /data/cyy/RLinf/datasets/testpiper \
#     --out-npz /tmp/testpiper_action_quantile_norm.npz
#
#   python toolkits/lerobot/lerobot_quantile_normalize_actions.py \
#     --dataset /data/cyy/RLinf/datasets/testpiper \
#     --write-dataset /data/cyy/RLinf/datasets/testpiper_action_qnorm

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def _flatten_q(stats_list: list) -> np.ndarray:
    return np.asarray(stats_list, dtype=np.float64).reshape(-1)


def load_action_quantiles_from_stats(
    stats_path: Path, action_key: str = "action"
) -> tuple[np.ndarray, np.ndarray]:
    """从 LeRobot ``stats.json`` 读取 ``action`` 的 q01 / q99（每维一条）。"""
    data = json.loads(stats_path.read_text())
    if isinstance(data.get("stats"), dict):
        stats = data["stats"]
    else:
        stats = data
    if action_key not in stats:
        raise KeyError(f"Missing {action_key!r} in stats. Keys: {list(stats)[:30]}")
    block = stats[action_key]
    if "q01" not in block or "q99" not in block:
        raise KeyError(f"stats[{action_key!r}] must contain 'q01' and 'q99'")
    q01 = _flatten_q(block["q01"])
    q99 = _flatten_q(block["q99"])
    if q01.shape != q99.shape:
        raise ValueError(f"q01 shape {q01.shape} != q99 shape {q99.shape}")
    return q01, q99


def quantile_normalize_actions(
    actions: np.ndarray,
    q01: np.ndarray,
    q99: np.ndarray,
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    """QUANTILES 归一化：``2 * (x - q01) / (q99 - q01) - 1``，按最后一维逐维计算。

    Args:
        actions: 形状 ``(..., D)``，与 ``q01`` / ``q99`` 长度 ``D`` 一致。
        q01, q99: 形状 ``(D,)``。
        eps: 分母为 0 时用 ``eps`` 替代（与 PyTorch 版逻辑一致）。

    Returns:
        与 ``actions`` 同形状的 float64 数组。
    """
    x = np.asarray(actions, dtype=np.float64)
    lo = np.asarray(q01, dtype=np.float64).reshape(1, -1)
    hi = np.asarray(q99, dtype=np.float64).reshape(1, -1)
    if x.shape[-1] != lo.shape[-1]:
        raise ValueError(f"action dim {x.shape[-1]} != q01 len {lo.shape[-1]}")
    denom = hi - lo
    denom = np.where(np.abs(denom) < eps, eps, denom)
    return 2.0 * (x - lo) / denom - 1.0


def _iter_data_parquet_files(dataset_root: Path) -> list[Path]:
    data_dir = dataset_root / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"No data/ under {dataset_root}")
    files = sorted(data_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet under {data_dir}")
    return files


def collect_actions_from_dataset(
    dataset_root: Path, action_key: str = "action"
) -> tuple[list[Path], np.ndarray, pd.DataFrame]:
    """按文件顺序读取所有帧的 ``action``，并拼成 ``(N, D)``；同时返回纵向合并的 DataFrame。"""
    paths = _iter_data_parquet_files(dataset_root)
    chunks: list[np.ndarray] = []
    dfs: list[pd.DataFrame] = []
    for p in paths:
        df = pd.read_parquet(p)
        if action_key not in df.columns:
            raise KeyError(f"{p}: missing column {action_key!r}")
        arr = np.stack(df[action_key].values)
        chunks.append(arr)
        dfs.append(df)
    stacked = np.concatenate(chunks, axis=0)
    big_df = pd.concat(dfs, ignore_index=True)
    return paths, stacked, big_df


def main() -> None:
    p = argparse.ArgumentParser(
        description="LeRobot: normalize per-frame action with q01/q99 from meta/stats.json."
    )
    p.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="LeRobot dataset root (contains meta/ and data/).",
    )
    p.add_argument(
        "--stats",
        type=Path,
        default=None,
        help="Path to stats JSON (default: <dataset>/meta/stats.json).",
    )
    p.add_argument("--action-key", default="action", help="Parquet + stats feature key.")
    p.add_argument(
        "--eps",
        type=float,
        default=1e-6,
        help="Epsilon when q99 - q01 is ~0.",
    )
    p.add_argument(
        "--preview",
        type=int,
        default=0,
        help="Print this many first rows: raw vs normalized (0 = skip).",
    )
    p.add_argument(
        "--summary",
        action="store_true",
        help="Print per-dimension min/max/mean/std of normalized actions.",
    )
    p.add_argument(
        "--out-npz",
        type=Path,
        default=None,
        help="Save action_normalized, episode_index, frame_index, global_index as .npz.",
    )
    p.add_argument(
        "--write-dataset",
        type=Path,
        default=None,
        help="Write a new dataset tree with normalized actions (copies meta + data parquets).",
    )
    args = p.parse_args()

    root = args.dataset.expanduser().resolve()
    stats_path = (args.stats or root / "meta" / "stats.json").expanduser().resolve()
    q01, q99 = load_action_quantiles_from_stats(stats_path, args.action_key)

    paths, actions, big_df = collect_actions_from_dataset(root, args.action_key)
    normalized = quantile_normalize_actions(actions, q01, q99, eps=args.eps)

    print(f"Dataset: {root}")
    print(f"Stats:   {stats_path}")
    print(f"Files:   {len(paths)}, total frames: {len(actions)}, action_dim: {actions.shape[1]}")

    if args.preview > 0:
        n = min(args.preview, len(actions))
        print(f"\nFirst {n} frames (raw -> normalized, each row one frame):")
        for i in range(n):
            print(f"  [{i}] raw:    {np.array2string(actions[i], precision=4, suppress_small=True)}")
            print(
                f"       norm: {np.array2string(normalized[i], precision=4, suppress_small=True)}"
            )

    if args.summary:
        print("\nNormalized action statistics (per dim: min, max, mean, std):")
        d = actions.shape[1]
        for j in range(d):
            col = normalized[:, j]
            print(
                f"  dim {j}: min={col.min():.6f} max={col.max():.6f} "
                f"mean={col.mean():.6f} std={col.std():.6f}"
            )

    if args.out_npz is not None:
        ep = (
            big_df["episode_index"].to_numpy()
            if "episode_index" in big_df.columns
            else None
        )
        fr = (
            big_df["frame_index"].to_numpy()
            if "frame_index" in big_df.columns
            else None
        )
        gl = (
            big_df["index"].to_numpy()
            if "index" in big_df.columns
            else np.arange(len(normalized))
        )
        out: dict[str, np.ndarray] = {
            "action_normalized": normalized.astype(np.float32),
            "q01": q01.astype(np.float32),
            "q99": q99.astype(np.float32),
            "global_index": gl,
        }
        if ep is not None:
            out["episode_index"] = ep
        if fr is not None:
            out["frame_index"] = fr
        args.out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out_npz, **out)
        print(f"\nWrote {args.out_npz}")

    if args.write_dataset is not None:
        out_root = args.write_dataset.expanduser().resolve()
        if out_root == root:
            raise ValueError("--write-dataset must differ from --dataset")
        meta_src = root / "meta"
        if meta_src.is_dir():
            shutil.copytree(meta_src, out_root / "meta", dirs_exist_ok=True)
        offset = 0
        for path in paths:
            df = pd.read_parquet(path)
            n = len(df)
            sl = normalized[offset : offset + n]
            offset += n
            df[args.action_key] = list(sl.astype(np.float32))
            rel = path.relative_to(root / "data")
            out_p = out_root / "data" / rel
            out_p.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out_p, index=False)
        print(f"\nWrote dataset with normalized actions to {out_root}")

    if not args.out_npz and not args.write_dataset and args.preview == 0 and not args.summary:
        print("\n(No output requested; use --preview, --summary, --out-npz, or --write-dataset.)")


if __name__ == "__main__":
    main()
