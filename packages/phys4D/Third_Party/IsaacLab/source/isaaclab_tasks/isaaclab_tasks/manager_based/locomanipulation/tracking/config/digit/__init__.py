




import gymnasium as gym

from . import agents




gym.register(
    id="Isaac-Tracking-LocoManip-Digit-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.loco_manip_env_cfg:DigitLocoManipEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DigitLocoManipPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Tracking-LocoManip-Digit-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.loco_manip_env_cfg:DigitLocoManipEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DigitLocoManipPPORunnerCfg",
    },
)
