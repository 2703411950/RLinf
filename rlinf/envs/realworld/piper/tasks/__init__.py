from gymnasium.envs.registration import register

from rlinf.envs.realworld.piper.tasks.dual_peg_insertion_env import (
    PiperDualPegInsertionEnv as PiperDualPegInsertionEnv,
)
from rlinf.envs.realworld.piper.tasks.peg_insertion_env import (
    PiperPegInsertionEnv as PiperPegInsertionEnv,
)

register(
    id="PiperPegInsertionEnv-v1",
    entry_point="rlinf.envs.realworld.piper.tasks:PiperPegInsertionEnv",
)

register(
    id="PiperDualPegInsertionEnv-v1",
    entry_point="rlinf.envs.realworld.piper.tasks:PiperDualPegInsertionEnv",
)
