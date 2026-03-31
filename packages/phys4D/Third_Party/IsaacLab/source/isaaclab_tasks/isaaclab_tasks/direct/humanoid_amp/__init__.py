




"""
AMP Humanoid locomotion environment.
"""

import gymnasium as gym

from . import agents





gym.register(
    id="Isaac-Humanoid-AMP-Dance-Direct-v0",
    entry_point=f"{__name__}.humanoid_amp_env:HumanoidAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoid_amp_env_cfg:HumanoidAmpDanceEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_dance_amp_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Humanoid-AMP-Run-Direct-v0",
    entry_point=f"{__name__}.humanoid_amp_env:HumanoidAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoid_amp_env_cfg:HumanoidAmpRunEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_run_amp_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Humanoid-AMP-Walk-Direct-v0",
    entry_point=f"{__name__}.humanoid_amp_env:HumanoidAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoid_amp_env_cfg:HumanoidAmpWalkEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_walk_amp_cfg.yaml",
    },
)
