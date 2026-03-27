from rlinf.envs.realworld.piper.piper_dual_env import PiperDualEnv, PiperDualRobotConfig


class PiperDualPegInsertionEnv(PiperDualEnv):
    """Dual-arm Piper peg insertion task environment."""

    def __init__(self, override_cfg, worker_info=None, hardware_info=None, env_idx=0):
        config = PiperDualRobotConfig(**override_cfg)
        super().__init__(config, worker_info, hardware_info, env_idx)

    @property
    def task_description(self):
        return "dual piper peg and insertion"
