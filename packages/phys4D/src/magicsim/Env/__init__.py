import gymnasium as gym

gym.register(
    id="SyncBaseEnv-V0",
    entry_point="magicsim.Env.Environment.SyncBaseEnv:SyncBaseEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="SyncCameraEnv-V0",
    entry_point="magicsim.Env.Environment.SyncCameraEnv:SyncCameraEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="SyncCollectEnv-V0",
    entry_point="magicsim.Env.Environment.SyncCollectEnv:SyncCollectEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="SyncRobotEnv-V0",
    entry_point="magicsim.Env.Environment.SyncRobotEnv:SyncRobotEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="IsaacRLEnv-V0",
    entry_point="magicsim.Env.Environment.Isaac.IsaacRLEnv:IsaacRLEnv",
    disable_env_checker=True,
    order_enforce=False,
    kwargs={
        "env_cfg_entry_point": "magicsim.Env.Environment.Isaac.IsaacRLEnv:IsaacRLEnvCfg",
    },
)
