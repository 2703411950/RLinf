#!/usr/bin/env python3
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

"""Analyze collected embodied trajectories (ReplayBuffer demos/ or LeRobot export).

Examples:
  # RLinf collect_real_data.py 写入的 demos/（含 metadata.json + trajectory_*.pt）
  python toolkits/replay_buffer/analyze_collected_trajectories.py \\
      --replay-dir /path/to/logs/.../demos

  # CollectEpisode 导出的 LeRobot 根目录（含 meta/ 与 data/）
  python toolkits/replay_buffer/analyze_collected_trajectories.py \\
      --lerobot-root /path/to/logs/.../collected_data

  # 只扫索引、不全量加载 .pt（更快）
  python toolkits/replay_buffer/analyze_collected_trajectories.py \\
      --replay-dir /path/to/demos --index-only

  # 限制分析条数、避免大库 OOM
  python toolkits/replay_buffer/analyze_collected_trajectories.py \\
      --replay-dir /path/to/demos --max-trajectories 50
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import torch


def _tensor_shape_summary(x: Any) -> str | None:
    if torch.is_tensor(x):
        return str(tuple(x.shape))
    if isinstance(x, dict):
        parts = {k: _tensor_shape_summary(v) for k, v in x.items()}
        return str(parts)
    return None


def _summarize_nested_obs_shapes(obs: Any, max_keys: int = 12) -> dict[str, str]:
    out: dict[str, str] = {}
    if not isinstance(obs, dict):
        return out
    for i, (k, v) in enumerate(obs.items()):
        if i >= max_keys:
            out["..."] = f"(truncated, total_keys={len(obs)})"
            break
        s = _tensor_shape_summary(v)
        if s is not None:
            out[k] = s
    return out


def analyze_replay_buffer_dir(
    replay_dir: Path,
    *,
    index_only: bool,
    max_trajectories: int | None,
) -> None:
    """Analyze TrajectoryReplayBuffer checkpoint directory (e.g. .../demos)."""
    from rlinf.data.replay_buffer import TrajectoryReplayBuffer

    meta_path = replay_dir / "metadata.json"
    idx_path = replay_dir / "trajectory_index.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"缺少 metadata.json: {meta_path}")
    if not idx_path.is_file():
        raise FileNotFoundError(f"缺少 trajectory_index.json: {idx_path}")

    with meta_path.open() as f:
        metadata = json.load(f)
    with idx_path.open() as f:
        index_data = json.load(f)

    traj_index = {int(k): v for k, v in index_data.get("trajectory_index", {}).items()}
    id_list = [int(k) for k in index_data.get("trajectory_id_list", [])]

    print("=== ReplayBuffer 目录（真机采集 demos 等）===")
    print(f"path: {replay_dir.resolve()}")
    print(f"metadata: {metadata}")
    print(f"trajectory_count (index): {len(id_list)}")

    num_samples_list = [traj_index[i]["num_samples"] for i in id_list if i in traj_index]
    if num_samples_list:
        print(
            "num_samples per traj: "
            f"min={min(num_samples_list)}, max={max(num_samples_list)}, "
            f"mean={statistics.mean(num_samples_list):.2f}, "
            f"median={statistics.median(num_samples_list):.2f}"
        )

    if index_only:
        print("\n(--index-only) 跳过逐条加载 trajectory_*.pt")
        return

    buffer = TrajectoryReplayBuffer(
        seed=metadata.get("seed", 1234),
        enable_cache=False,
        auto_save=False,
        trajectory_format=metadata.get("trajectory_format", "pt"),
    )
    buffer.load_checkpoint(str(replay_dir), is_distributed=False)

    to_scan = id_list
    if max_trajectories is not None:
        to_scan = id_list[: max(0, max_trajectories)]

    reward_sums: list[float] = []
    traj_lengths_t: list[int] = []
    action_dims: list[int] = []
    obs_keys_examples: dict[str, str] | None = None

    for tid in to_scan:
        info = traj_index.get(tid)
        if info is None:
            continue
        mid = info["model_weights_id"]
        traj = buffer._load_trajectory(tid, mid)
        if traj.rewards is not None:
            r = traj.rewards
            traj_lengths_t.append(int(r.shape[0]))
            reward_sums.append(float(r.sum().item()))
        if traj.actions is not None:
            a = traj.actions
            if a.dim() >= 2:
                action_dims.append(int(a.shape[-1]))
        if obs_keys_examples is None and traj.curr_obs:
            obs_keys_examples = _summarize_nested_obs_shapes(traj.curr_obs)

    print(f"\n逐条加载统计（共分析 {len(to_scan)} / {len(id_list)} 条）:")
    if traj_lengths_t:
        print(
            "  时间步 T (rewards 第0维): "
            f"min={min(traj_lengths_t)}, max={max(traj_lengths_t)}, "
            f"mean={statistics.mean(traj_lengths_t):.2f}"
        )
    if reward_sums:
        print(
            "  每条轨迹 reward 总和: "
            f"min={min(reward_sums):.4f}, max={max(reward_sums):.4f}, "
            f"mean={statistics.mean(reward_sums):.4f}"
        )
        pos = sum(1 for x in reward_sums if x > 0)
        print(f"  reward 总和 > 0 的轨迹: {pos}/{len(reward_sums)}")
    if action_dims:
        ad_mean = statistics.mean(action_dims)
        if len(set(action_dims)) == 1:
            print(f"  action 维度: {action_dims[0]}")
        else:
            print(f"  action 末维维度: min={min(action_dims)}, max={max(action_dims)}, mean={ad_mean:.2f}")
    if obs_keys_examples:
        print(f"  curr_obs 张量形状示例（首条有效轨迹）: {obs_keys_examples}")

    if max_trajectories is not None and len(id_list) > len(to_scan):
        print(
            f"\n提示: 仅分析了前 {len(to_scan)} 条；去掉 --max-trajectories 可全量扫描。"
        )


def analyze_lerobot_root(root: Path) -> None:
    """Summarize LeRobot-style dataset under root/ (meta/ + data/)."""
    meta_dir = root / "meta"
    data_dir = root / "data"
    if not meta_dir.is_dir():
        raise FileNotFoundError(f"缺少 meta/: {meta_dir}")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"缺少 data/: {data_dir}")

    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise RuntimeError(
            "分析 LeRobot 数据需要 pyarrow：pip install pyarrow"
        ) from e

    print("=== LeRobot 导出目录 ===")
    print(f"path: {root.resolve()}")

    info_path = meta_dir / "info.json"
    if info_path.is_file():
        with info_path.open() as f:
            info = json.load(f)
        print(f"meta/info.json keys: {list(info.keys())[:20]}...")

    episode_files = sorted(data_dir.glob("**/episode_*.parquet"))
    print(f"episode parquet 文件数: {len(episode_files)}")

    lengths: list[int] = []
    success_count = 0
    total_rows = 0
    for ep_path in episode_files:
        table = pq.read_table(ep_path)
        n = table.num_rows
        lengths.append(n)
        total_rows += n
        if "is_success" in table.column_names:
            col = table.column("is_success")
            if n > 0 and bool(col[n - 1].as_py()):
                success_count += 1

    if lengths:
        print(f"总帧数（所有 parquet 行数之和）: {total_rows}")
        print(
            "每 episode 帧数: "
            f"min={min(lengths)}, max={max(lengths)}, "
            f"mean={statistics.mean(lengths):.2f}, "
            f"median={statistics.median(lengths):.2f}"
        )
        t1 = sum(1 for L in lengths if L <= 1)
        if t1:
            print(
                f"警告: 有 {t1} 个 episode 只有 <=1 帧，采集可能异常（一步就 done / 配置问题）。"
            )
    if episode_files and "is_success" in pq.read_table(episode_files[0]).column_names:
        print(f"首帧 is_success=True 的 episode 数（粗略）: {success_count}/{len(episode_files)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="分析真机/仿真采集的轨迹（ReplayBuffer demos 或 LeRobot 目录）"
    )
    parser.add_argument(
        "--replay-dir",
        type=str,
        default=None,
        help="TrajectoryReplayBuffer 目录（含 metadata.json、trajectory_index.json）",
    )
    parser.add_argument(
        "--lerobot-root",
        type=str,
        default=None,
        help="LeRobot 数据集根目录（含 meta/ 与 data/）",
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="仅读 metadata/index，不加载 trajectory_*.pt",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        help="最多加载分析的轨迹条数（省内存）",
    )
    args = parser.parse_args()

    if args.replay_dir is None and args.lerobot_root is None:
        parser.error("请指定 --replay-dir 或 --lerobot-root")

    if args.replay_dir is not None:
        analyze_replay_buffer_dir(
            Path(args.replay_dir).expanduser().resolve(),
            index_only=args.index_only,
            max_trajectories=args.max_trajectories,
        )
        print()

    if args.lerobot_root is not None:
        analyze_lerobot_root(Path(args.lerobot_root).expanduser().resolve())


if __name__ == "__main__":
    main()
