




"""Sub-package with environment wrappers for Locomanipulation SDG."""

import gymnasium as gym

gym.register(
    id="Isaac-G1-SteeringWheel-Locomanipulation",
    entry_point=f"{__name__}.g1_locomanipulation_sdg_env:G1LocomanipulationSDGEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_locomanipulation_sdg_env:G1LocomanipulationSDGEnvCfg",
    },
    disable_env_checker=True,
)
