




import gymnasium as gym










gym.register(
    id="Isaac-Stack-Cube-UR10-Long-Suction-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stack_ik_rel_env_cfg:UR10LongSuctionCubeStackEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-UR10-Short-Suction-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stack_ik_rel_env_cfg:UR10ShortSuctionCubeStackEnvCfg",
    },
    disable_env_checker=True,
)
