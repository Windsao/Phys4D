




import gymnasium as gym

from . import agents





gym.register(
    id="Isaac-AutoMate-Assembly-Direct-v0",
    entry_point=f"{__name__}.assembly_env:AssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.assembly_env:AssemblyEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)


gym.register(
    id="Isaac-AutoMate-Disassembly-Direct-v0",
    entry_point=f"{__name__}.disassembly_env:DisassemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.disassembly_env:DisassemblyEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
