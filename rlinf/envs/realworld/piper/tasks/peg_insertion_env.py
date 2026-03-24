from rlinf.envs.realworld.piper.piper_env import PiperEnv, PiperRobotConfig


class PiperPegInsertionEnv(PiperEnv):
    """Piper peg insertion task environment."""

    def __init__(self, override_cfg, worker_info=None, hardware_info=None, env_idx=0):
        config = PiperRobotConfig(**override_cfg)
        super().__init__(config, worker_info, hardware_info, env_idx)

    @property
    def task_description(self):
        return "piper peg and insertion"
