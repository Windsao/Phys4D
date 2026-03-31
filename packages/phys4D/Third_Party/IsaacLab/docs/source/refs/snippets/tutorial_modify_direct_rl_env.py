







from .h1_env import H1Env, H1EnvCfg



gym.register(
    id="Isaac-H1-Direct-v0",
    entry_point="isaaclab_tasks.direct.humanoid:H1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": H1EnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HumanoidPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)



from isaaclab_assets import H1_CFG



action_space = 19
observation_space = 69



robot: ArticulationCfg = H1_CFG.replace(prim_path="/World/envs/env_.*/Robot")
joint_gears: list = [
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
]



