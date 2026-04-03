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

import os
import time
from collections import defaultdict
from typing import Any

from omegaconf import DictConfig

from rlinf.utils.logging import get_logger
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.runner_utils import check_progress


class OfflineEmbodiedRunner:
    """Offline (data-only) runner for embodied SAC-style training.

    This runner launches only the actor worker group and repeatedly calls
    ``actor.run_training()``. It intentionally does not start env/rollout
    workers, so training can run purely on pre-collected replay data.
    """

    def __init__(self, cfg: DictConfig, actor: Any):
        self.cfg = cfg
        self.actor = actor

        self.logger = get_logger()
        self.metric_logger = MetricLogger(cfg)

        self.global_step = 0
        self.max_steps = self._compute_max_steps()

    def _compute_max_steps(self) -> int:
        # Keep the same semantics as EmbodiedRunner.set_max_steps().
        num_steps_per_epoch = 1
        max_steps = num_steps_per_epoch * int(self.cfg.runner.max_epochs)

        # Optional user override.
        if (configured_max_steps := self.cfg.runner.get("max_steps", -1)) >= 0:
            max_steps = min(max_steps, int(configured_max_steps))
        return int(max_steps)

    @staticmethod
    def _aggregate_numeric_metrics(metrics_list: list[dict]) -> dict[str, float]:
        merged: dict[str, list[float]] = defaultdict(list)
        for metrics in metrics_list:
            if not metrics:
                continue
            for k, v in metrics.items():
                merged[k].append(float(v))
        return {k: sum(vs) / len(vs) for k, vs in merged.items() if vs}

    def init_workers(self) -> None:
        # Initialize only actor workers.
        self.actor.init_worker().wait()

        resume_dir = self.cfg.runner.get("resume_dir", None)
        if resume_dir is None:
            return

        self.logger.info(
            f"Resuming offline training from checkpoint directory: {resume_dir}"
        )
        actor_checkpoint_path = os.path.join(resume_dir, "actor")
        if not os.path.exists(actor_checkpoint_path):
            raise FileNotFoundError(
                f"resume_dir actor checkpoint not found: {actor_checkpoint_path}"
            )

        self.actor.load_checkpoint(actor_checkpoint_path).wait()
        # Expect .../checkpoints/global_step_<N>
        self.global_step = int(resume_dir.split("global_step_")[-1])

    def _save_checkpoint(self) -> None:
        self.logger.info(f"Saving checkpoint at step {self.global_step}.")
        base_output_dir = os.path.join(
            self.cfg.runner.logger.log_path,
            self.cfg.runner.logger.experiment_name,
            f"checkpoints/global_step_{self.global_step}",
        )
        actor_save_path = os.path.join(base_output_dir, "actor")
        os.makedirs(actor_save_path, exist_ok=True)
        self.actor.save_checkpoint(actor_save_path, self.global_step).wait()

    def run(self) -> None:
        start_time = time.time()
        start_step = self.global_step
        self.logger.info(
            f"Running OfflineEmbodiedRunner from step {start_step} to {self.max_steps}"
        )

        for _ in range(start_step, self.max_steps):
            # Keep the same global_step semantics as EmbodiedRunner.
            self.actor.set_global_step(self.global_step).wait()

            actor_training_handle = self.actor.run_training()
            actor_training_results = actor_training_handle.wait()

            # actor_training_results is usually a list[dict] per actor rank.
            if not actor_training_results or not actor_training_results[0]:
                self.global_step += 1
                continue

            aggregated = self._aggregate_numeric_metrics(actor_training_results)
            training_metrics = {f"train/{k}": v for k, v in aggregated.items()}
            self.global_step += 1
            self.metric_logger.log(training_metrics, self.global_step)

            _, save_model, _ = check_progress(
                step=self.global_step,
                max_steps=self.max_steps,
                val_check_interval=int(self.cfg.runner.val_check_interval),
                save_interval=int(self.cfg.runner.save_interval),
                limit_val_batches=1.0,
                run_time_exceeded=False,
            )
            if save_model:
                self._save_checkpoint()

        self.metric_logger.finish()
        self.logger.info(
            f"Offline training finished in {time.time() - start_time:.1f}s "
            f"(global_step={self.global_step})."
        )

