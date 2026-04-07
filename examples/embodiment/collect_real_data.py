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

import copy
import os
import sys
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, open_dict
from tqdm import tqdm

from rlinf.config import SupportedModel
from rlinf.data.embodied_io_struct import (
    ChunkStepResult,
    EmbodiedRolloutResult,
)
from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.envs.action_utils import prepare_actions
from rlinf.envs.realworld.realworld_env import RealWorldEnv
from rlinf.models import get_model
from rlinf.models.embodiment.base_policy import BasePolicy
from rlinf.scheduler import Cluster, ComponentPlacement, Worker


class DataCollector(Worker):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.num_data_episodes = cfg.runner.num_data_episodes
        self.action_dim = int(cfg.runner.get("action_dim", 7))
        self.count_success_episodes_only = bool(
            cfg.runner.get("count_success_episodes_only", True)
        )
        self.pause_for_manual_reset = bool(cfg.runner.get("pause_for_manual_reset", False))
        if os.environ.get("RLINF_COLLECT_SKIP_PAUSE", "").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            self.pause_for_manual_reset = False
        self.pause_for_manual_reset_prompt = str(
            cfg.runner.get(
                "pause_for_manual_reset_prompt",
                "[数据采集] 本段轨迹已保存。请手动复位场景（工件/台面等），完成后按 Enter 继续：将执行机械臂 reset 并开始下一段。",
            )
        )
        self.total_cnt = 0
        self.env = RealWorldEnv(
            cfg.env.eval,
            num_envs=1,
            seed_offset=0,
            total_num_processes=1,
            worker_info=self.worker_info,
        )

        if self.cfg.env.get("data_collection", None) and getattr(
            self.cfg.env.data_collection, "enabled", False
        ):
            from rlinf.envs.wrappers import CollectEpisode

            self.env = CollectEpisode(
                self.env,
                save_dir=self.cfg.env.data_collection.save_dir,
                # rank=self._rank,
                # num_envs=1,
                export_format=getattr(
                    self.cfg.env.data_collection, "export_format", "pickle"
                ),
                robot_type=getattr(self.cfg.env.data_collection, "robot_type", "panda"),
                fps=getattr(self.cfg.env.data_collection, "fps", 10),
                only_success=getattr(
                    self.cfg.env.data_collection, "only_success", False
                ),
                stats_sample_ratio=getattr(
                    self.cfg.env.data_collection, "stats_sample_ratio", 0.1
                ),
                finalize_interval=getattr(
                    self.cfg.env.data_collection, "finalize_interval", 100
                ),
            )

        self.save_demos = bool(getattr(self.cfg.runner, "save_demos", True))
        if self.save_demos:
            buffer_path = os.path.join(self.cfg.runner.logger.log_path, "demos")
            self.log_info(f"Initializing ReplayBuffer at: {buffer_path}")
            self.buffer = TrajectoryReplayBuffer(
                seed=self.cfg.seed if hasattr(self.cfg, "seed") else 1234,
                enable_cache=False,
                auto_save=True,
                auto_save_path=buffer_path,
                trajectory_format="pt",
            )
        else:
            self.buffer = None
            self.log_info("save_demos=false: trajectories will not be written to disk.")

        self.use_policy_inference = bool(
            getattr(self.cfg.runner, "use_policy_inference", False)
        )
        self._policy: BasePolicy | None = None
        self._num_action_chunks = 1
        if self.use_policy_inference:
            self._init_policy_model()

    def _init_policy_model(self) -> None:
        """Load rollout policy the same way as ``MultiStepRolloutWorker.init_worker``."""
        if not hasattr(self.cfg, "actor") or not hasattr(self.cfg, "rollout"):
            raise ValueError(
                "use_policy_inference=True requires ``actor`` and ``rollout`` in the Hydra config "
                "(see examples/embodiment/config/realworld_piper_collect_policy.yaml)."
            )
        rollout_model_config = copy.deepcopy(self.cfg.actor.model)
        with open_dict(rollout_model_config):
            rollout_model_config.precision = self.cfg.rollout.model.precision
            rollout_model_config.model_path = self.cfg.rollout.model.model_path

        self._policy = get_model(rollout_model_config)
        ckpt_path = self.cfg.runner.get("ckpt_path", None)
        if ckpt_path:
            state = torch.load(ckpt_path, map_location="cpu")
            self._policy.load_state_dict(state)
        self._policy.eval()
        self._num_action_chunks = int(self.cfg.actor.model.num_action_chunks)
        self.log_info(
            f"[collect] Policy loaded from {self.cfg.rollout.model.model_path}, "
            f"num_action_chunks={self._num_action_chunks}"
        )

    def _policy_predict_kwargs(self, mode: str) -> dict:
        """Match ``MultiStepRolloutWorker.predict`` kwargs for common embodied models."""
        kwargs: dict = {}
        mt = SupportedModel(self.cfg.actor.model.model_type)
        if mt in (
            SupportedModel.OPENPI,
            SupportedModel.MLP_POLICY,
            SupportedModel.GR00T,
            SupportedModel.CNN_POLICY,
        ):
            alg = getattr(self.cfg, "algorithm", None)
            if (
                alg is not None
                and getattr(alg, "loss_type", None) == "embodied_dagger"
            ):
                kwargs["mode"] = "eval"
            else:
                kwargs["mode"] = mode
        if mt in (
            SupportedModel.CNN_POLICY,
            SupportedModel.FLOW_POLICY,
            SupportedModel.MLP_POLICY,
        ):
            kwargs["return_obs"] = not hasattr(self._policy, "q_head")
        return kwargs

    def _obs_for_policy(self, obs: dict) -> dict:
        """Move tensor observations to the policy device; keep ``task_descriptions`` as-is."""
        device = next(self._policy.parameters()).device
        out: dict = {}
        for key, val in obs.items():
            if key == "task_descriptions":
                out[key] = val
            elif isinstance(val, torch.Tensor):
                out[key] = val.to(device, non_blocking=True)
            elif isinstance(val, np.ndarray):
                out[key] = torch.from_numpy(val).to(device)
            else:
                out[key] = val
        # OpenPI ``obs_processor`` requires these keys (see ``openpi_action_model.obs_processor``).
        if "task_descriptions" not in out:
            default_prompt = self.cfg.runner.get("default_task_prompt", "robot manipulation")
            out["task_descriptions"] = [default_prompt]
        # LiberoInputs always reads ``observation/wrist_image``; ``obs_processor`` only sets it when
        # ``wrist_images`` is not None. Duplicate main or use first extra camera as wrist view.
        if out.get("wrist_images") is None:
            if out.get("extra_view_images") is not None:
                ev = out["extra_view_images"]
                if isinstance(ev, torch.Tensor) and ev.dim() >= 2 and ev.shape[1] > 0:
                    out["wrist_images"] = ev[:, 0].contiguous()
                elif isinstance(ev, np.ndarray) and ev.ndim >= 2 and ev.shape[1] > 0:
                    out["wrist_images"] = torch.from_numpy(ev[:, 0]).to(device)
            if out.get("wrist_images") is None and "main_images" in out:
                mi = out["main_images"]
                out["wrist_images"] = (
                    mi.clone() if isinstance(mi, torch.Tensor) else torch.from_numpy(mi).to(device)
                )
        return out

    def _process_obs(self, obs):
        """
        Process observations to match the format expected by EmbodiedRolloutResult.
        """
        # Shallow copy: when ``record_task_description`` is false we pop task_descriptions; the same
        # ``obs`` dict is still used later for policy inference and must keep task_descriptions.
        obs = dict(obs)
        if not self.cfg.runner.record_task_description:
            obs.pop("task_descriptions", None)

        ret_obs = {}
        for key, val in obs.items():
            if isinstance(val, np.ndarray):
                val = torch.from_numpy(val)

            val = val.cpu()

            # Map keys: 'images' -> 'main_images', others remain
            if "images" == key:
                ret_obs["main_images"] = val.clone()  # Keep uint8
            else:
                ret_obs[key] = val.clone()

        return ret_obs

    def _wait_for_manual_reset(self) -> None:
        """Block until the operator finishes manual scene reset (stdin Enter)."""
        if not self.pause_for_manual_reset:
            return
        msg = self.pause_for_manual_reset_prompt
        self.log_info(msg)
        print(msg, file=sys.stderr, flush=True)
        try:
            input()
        except EOFError:
            self.log_warning(
                "stdin closed (non-interactive); skip manual reset pause."
            )

    def run(self):
        if self.use_policy_inference:
            self._run_with_policy()
        else:
            self._run_manual_only()

    def _run_with_policy(self) -> None:
        """Rollout with the same ``prepare_actions`` + ``chunk_step`` path as async training."""
        predict_mode = str(self.cfg.runner.get("policy_predict_mode", "eval"))
        success_cnt = 0
        progress_bar = tqdm(
            range(self.num_data_episodes), desc="Collecting Data Episodes (policy):"
        )

        while success_cnt < self.num_data_episodes:
            obs, _ = self.env.reset()
            current_rollout = EmbodiedRolloutResult(
                max_episode_length=self.cfg.env.eval.max_episode_steps,
            )
            current_obs_processed = self._process_obs(obs)
            episode_done = False

            while not episode_done:
                env_obs = self._obs_for_policy(obs)
                pred_kw = self._policy_predict_kwargs(predict_mode)
                with torch.no_grad():
                    raw_actions, result = self._policy.predict_action_batch(
                        env_obs=env_obs, **pred_kw
                    )

                chunk_actions = prepare_actions(
                    raw_chunk_actions=raw_actions,
                    env_type=self.cfg.env.eval.env_type,
                    model_type=self.cfg.actor.model.model_type,
                    num_action_chunks=self.cfg.actor.model.num_action_chunks,
                    action_dim=self.cfg.actor.model.action_dim,
                )

                self.log_info(f"chunk_actions: {chunk_actions}")    
                if isinstance(chunk_actions, np.ndarray):
                    chunk_actions_t = torch.from_numpy(chunk_actions).float()
                else:
                    chunk_actions_t = chunk_actions.float()

                (
                    obs_list,
                    chunk_rewards,
                    chunk_terminations,
                    chunk_truncations,
                    _infos_list,
                ) = self.env.chunk_step(chunk_actions_t)
                next_obs = obs_list[-1]
                next_obs_processed = self._process_obs(next_obs)
                chunk_dones = torch.logical_or(chunk_terminations, chunk_truncations)

                actions_stored = result["forward_inputs"].get("action")
                if actions_stored is None:
                    ra = raw_actions
                    if isinstance(ra, torch.Tensor):
                        actions_stored = ra.reshape(ra.shape[0], -1).cpu()
                    else:
                        actions_stored = torch.from_numpy(np.asarray(ra)).reshape(
                            np.asarray(ra).shape[0], -1
                        )

                step_result = ChunkStepResult(
                    actions=actions_stored,
                    prev_logprobs=result.get("prev_logprobs"),
                    prev_values=result.get("prev_values"),
                    rewards=chunk_rewards,
                    dones=chunk_dones,
                    terminations=chunk_terminations,
                    truncations=chunk_truncations,
                    forward_inputs=result["forward_inputs"],
                )
                current_rollout.append_step_result(step_result)
                current_rollout.append_transitions(
                    curr_obs=current_obs_processed, next_obs=next_obs_processed
                )

                obs = next_obs
                current_obs_processed = next_obs_processed
                episode_done = bool(chunk_dones.any())

            if isinstance(chunk_rewards, torch.Tensor):
                r_val = float(chunk_rewards.sum().item())
            else:
                r_val = float(np.sum(chunk_rewards))

            if self.count_success_episodes_only:
                success_cnt += int(r_val)
            else:
                success_cnt += 1
            self.total_cnt += 1
            self.log_info(
                f"Episode reward/signal (chunk sum): {r_val}. Total: {success_cnt}/{self.num_data_episodes}"
            )

            trajectory = current_rollout.to_trajectory()
            trajectory.intervene_flags = torch.ones_like(trajectory.intervene_flags)
            if self.buffer is not None:
                self.buffer.add_trajectories([trajectory])

            self._wait_for_manual_reset()
            progress_bar.update(1)

        if self.buffer is not None:
            self.buffer.close()
        if self.save_demos:
            self.log_info(
                f"Finished. Demos saved in: {os.path.join(self.cfg.runner.logger.log_path, 'demos')}"
            )
        else:
            self.log_info("Finished policy rollout (save_demos=false).")
        self.env.close()

    def _run_manual_only(self) -> None:
        """Original zero-action / SpaceMouse intervention collection."""
        obs, _ = self.env.reset()
        success_cnt = 0
        progress_bar = tqdm(
            range(self.num_data_episodes), desc="Collecting Data Episodes:"
        )

        current_rollout = EmbodiedRolloutResult(
            max_episode_length=self.cfg.env.eval.max_episode_steps,
        )

        current_obs_processed = self._process_obs(obs)

        while success_cnt < self.num_data_episodes:
            action = np.zeros((1, self.action_dim))
            # RealWorldEnv.step returns (obs, reward, terminations, truncations, infos)
            next_obs, reward, terminations, truncations, info = self.env.step(action)
            done = terminations | truncations

            if "intervene_action" in info:
                action = info["intervene_action"]

            next_obs_processed = self._process_obs(next_obs)

            # --- Construct ChunkStepResult ---
            # Prepare action tensor [1, action_dim] (e.g. Piper 7 = 6 joints + gripper)
            if isinstance(action, torch.Tensor):
                action_tensor = action.float().cpu()
            else:
                action_tensor = torch.from_numpy(action).float()

            if action_tensor.ndim == 1:
                action_tensor = action_tensor.unsqueeze(0)

            # Reward and Done [1, 1]
            if isinstance(reward, torch.Tensor):
                reward_tensor = reward.float().cpu()
            else:
                reward_tensor = torch.tensor(reward).float()
            if reward_tensor.ndim == 1:
                reward_tensor = reward_tensor.unsqueeze(1)

            if isinstance(terminations, torch.Tensor):
                term_t = terminations.bool().cpu()
            else:
                term_t = torch.tensor(terminations).bool()
            if term_t.ndim == 1:
                term_t = term_t.unsqueeze(1)

            if isinstance(truncations, torch.Tensor):
                trunc_t = truncations.bool().cpu()
            else:
                trunc_t = torch.tensor(truncations).bool()
            if trunc_t.ndim == 1:
                trunc_t = trunc_t.unsqueeze(1)

            done_tensor = term_t | trunc_t

            step_result = ChunkStepResult(
                actions=action_tensor,
                rewards=reward_tensor,
                dones=done_tensor,
                terminations=term_t,
                truncations=trunc_t,
                forward_inputs={"action": action_tensor},
            )

            current_rollout.append_step_result(step_result)
            current_rollout.append_transitions(
                curr_obs=current_obs_processed, next_obs=next_obs_processed
            )

            obs = next_obs
            current_obs_processed = next_obs_processed

            episode_done = bool(done_tensor.any()) if isinstance(done_tensor, torch.Tensor) else bool(done_tensor)
            if episode_done:
                r_val = (
                    reward[0]
                    if hasattr(reward, "__getitem__") and len(reward) > 0
                    else reward
                )
                if isinstance(r_val, torch.Tensor):
                    r_val = r_val.item()

                if self.count_success_episodes_only:
                    success_cnt += int(r_val)
                else:
                    success_cnt += 1
                self.total_cnt += 1
                self.log_info(
                    f"Episode reward/signal: {r_val}. Total: {success_cnt}/{self.num_data_episodes}"
                )

                trajectory = current_rollout.to_trajectory()
                trajectory.intervene_flags = torch.ones_like(trajectory.intervene_flags)
                if self.buffer is not None:
                    self.buffer.add_trajectories([trajectory])

                self._wait_for_manual_reset()

                # Reset for next episode (robot returns to configured reset pose)
                obs, _ = self.env.reset()
                current_obs_processed = self._process_obs(obs)
                current_rollout = EmbodiedRolloutResult(
                    max_episode_length=self.cfg.env.eval.max_episode_steps,
                )
                progress_bar.update(1)

        if self.buffer is not None:
            self.buffer.close()
        if self.save_demos:
            self.log_info(
                f"Finished. Demos saved in: {os.path.join(self.cfg.runner.logger.log_path, 'demos')}"
            )
        else:
            self.log_info("Finished manual collection (save_demos=false).")
        self.env.close()


@hydra.main(
    version_base="1.1", config_path="config", config_name="realworld_collect_data"
)
def main(cfg):
    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = ComponentPlacement(cfg, cluster)
    env_placement = component_placement.get_strategy("env")
    collector = DataCollector.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )
    collector.run().wait()


if __name__ == "__main__":
    main()
