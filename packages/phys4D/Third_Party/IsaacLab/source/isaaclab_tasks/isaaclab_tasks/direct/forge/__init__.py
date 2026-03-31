




import gymnasium as gym

from . import agents
from .forge_env import ForgeEnv
from .forge_env_cfg import ForgeTaskGearMeshCfg, ForgeTaskNutThreadCfg, ForgeTaskPegInsertCfg





gym.register(
    id="Isaac-Forge-PegInsert-Direct-v0",
    entry_point="isaaclab_tasks.direct.forge:ForgeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ForgeTaskPegInsertCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Forge-GearMesh-Direct-v0",
    entry_point="isaaclab_tasks.direct.forge:ForgeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ForgeTaskGearMeshCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Forge-NutThread-Direct-v0",
    entry_point="isaaclab_tasks.direct.forge:ForgeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ForgeTaskNutThreadCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_nut_thread.yaml",
    },
)
