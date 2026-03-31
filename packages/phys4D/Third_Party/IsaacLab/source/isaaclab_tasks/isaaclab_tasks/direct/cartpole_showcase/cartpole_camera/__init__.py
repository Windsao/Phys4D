




"""
Cartpole balancing environment with camera.
"""

import gymnasium as gym

from . import agents









gym.register(
    id="Isaac-Cartpole-Camera-Showcase-Box-Box-Direct-v0",
    entry_point=f"{__name__}.cartpole_camera_env:CartpoleCameraShowcaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_camera_env_cfg:BoxBoxEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_box_box_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-Camera-Showcase-Box-Discrete-Direct-v0",
    entry_point=f"{__name__}.cartpole_camera_env:CartpoleCameraShowcaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_camera_env_cfg:BoxDiscreteEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_box_discrete_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-Camera-Showcase-Box-MultiDiscrete-Direct-v0",
    entry_point=f"{__name__}.cartpole_camera_env:CartpoleCameraShowcaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_camera_env_cfg:BoxMultiDiscreteEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_box_multidiscrete_ppo_cfg.yaml",
    },
)





gym.register(
    id="Isaac-Cartpole-Camera-Showcase-Dict-Box-Direct-v0",
    entry_point=f"{__name__}.cartpole_camera_env:CartpoleCameraShowcaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_camera_env_cfg:DictBoxEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_dict_box_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-Camera-Showcase-Dict-Discrete-Direct-v0",
    entry_point=f"{__name__}.cartpole_camera_env:CartpoleCameraShowcaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_camera_env_cfg:DictDiscreteEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_dict_discrete_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-Camera-Showcase-Dict-MultiDiscrete-Direct-v0",
    entry_point=f"{__name__}.cartpole_camera_env:CartpoleCameraShowcaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_camera_env_cfg:DictMultiDiscreteEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_dict_multidiscrete_ppo_cfg.yaml",
    },
)





gym.register(
    id="Isaac-Cartpole-Camera-Showcase-Tuple-Box-Direct-v0",
    entry_point=f"{__name__}.cartpole_camera_env:CartpoleCameraShowcaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_camera_env_cfg:TupleBoxEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_tuple_box_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-Camera-Showcase-Tuple-Discrete-Direct-v0",
    entry_point=f"{__name__}.cartpole_camera_env:CartpoleCameraShowcaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_camera_env_cfg:TupleDiscreteEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_tuple_discrete_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-Camera-Showcase-Tuple-MultiDiscrete-Direct-v0",
    entry_point=f"{__name__}.cartpole_camera_env:CartpoleCameraShowcaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_camera_env_cfg:TupleMultiDiscreteEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_tuple_multidiscrete_ppo_cfg.yaml",
    },
)
