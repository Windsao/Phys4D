




import gymnasium as gym








gym.register(
    id="Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.place_toy2box_rmp_rel_env_cfg:RmpFlowAgibotPlaceToy2BoxEnvCfg",
    },
    disable_env_checker=True,
)




gym.register(
    id="Isaac-Place-Mug-Agibot-Left-Arm-RmpFlow-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.place_upright_mug_rmp_rel_env_cfg:RmpFlowAgibotPlaceUprightMugEnvCfg",
    },
    disable_env_checker=True,
)
