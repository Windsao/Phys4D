




import gymnasium as gym

from . import agents
from .factory_env import FactoryEnv
from .factory_env_cfg import FactoryTaskGearMeshCfg, FactoryTaskNutThreadCfg, FactoryTaskPegInsertCfg






gym.register(
    id="TacEx-Factory-PegInsert-Direct-v0",
    entry_point=f"{__name__}.factory_env:FactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryTaskPegInsertCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)


gym.register(
    id="TacEx-Factory-GearMesh-Direct-v0",
    entry_point=f"{__name__}.factory_env:FactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryTaskGearMeshCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)


gym.register(
    id="TacEx-Factory-NutThread-Direct-v0",
    entry_point=f"{__name__}.factory_env:FactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryTaskNutThreadCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
