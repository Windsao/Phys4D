





import gymnasium as gym









gym.register(
    id="Isaac-Stack-Cube-Galbot-Left-Arm-Gripper-RmpFlow-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stack_rmp_rel_env_cfg:RmpFlowGalbotLeftArmCubeStackEnvCfg",
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Stack-Cube-Galbot-Right-Arm-Suction-RmpFlow-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stack_rmp_rel_env_cfg:RmpFlowGalbotRightArmCubeStackEnvCfg",
    },
    disable_env_checker=True,
)





gym.register(
    id="Isaac-Stack-Cube-Galbot-Left-Arm-Gripper-Visuomotor-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stack_rmp_rel_env_cfg:RmpFlowGalbotLeftArmCubeStackVisuomotorEnvCfg",
    },
    disable_env_checker=True,
)




gym.register(
    id="Isaac-Stack-Cube-Galbot-Left-Arm-Gripper-Visuomotor-Joint-Position-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.stack_rmp_rel_env_cfg:GalbotLeftArmJointPositionCubeStackVisuomotorEnvCfg_PLAY"
        ),
    },
    disable_env_checker=True,
)




gym.register(
    id="Isaac-Stack-Cube-Galbot-Left-Arm-Gripper-Visuomotor-RmpFlow-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stack_rmp_rel_env_cfg:GalbotLeftArmRmpFlowCubeStackVisuomotorEnvCfg_PLAY",
    },
    disable_env_checker=True,
)
