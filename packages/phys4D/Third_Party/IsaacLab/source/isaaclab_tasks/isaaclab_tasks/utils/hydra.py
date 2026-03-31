




"""Sub-module with utilities for the hydra configuration system."""


import functools
from collections.abc import Callable

try:
    import hydra
    from hydra.core.config_store import ConfigStore
    from omegaconf import DictConfig, OmegaConf
except ImportError:
    raise ImportError("Hydra is not installed. Please install it by running 'pip install hydra-core'.")

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.envs.utils.spaces import replace_env_cfg_spaces_with_strings, replace_strings_with_env_cfg_spaces
from isaaclab.utils import replace_slices_with_strings, replace_strings_with_slices

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


def register_task_to_hydra(
    task_name: str, agent_cfg_entry_point: str
) -> tuple[ManagerBasedRLEnvCfg | DirectRLEnvCfg, dict]:
    """Register the task configuration to the Hydra configuration store.

    This function resolves the configuration file for the environment and agent based on the task's name.
    It then registers the configurations to the Hydra configuration store.

    Args:
        task_name: The name of the task.
        agent_cfg_entry_point: The entry point key to resolve the agent's configuration file.

    Returns:
        A tuple containing the parsed environment and agent configuration objects.
    """

    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    agent_cfg = None
    if agent_cfg_entry_point:
        agent_cfg = load_cfg_from_registry(task_name, agent_cfg_entry_point)


    env_cfg = replace_env_cfg_spaces_with_strings(env_cfg)

    env_cfg_dict = env_cfg.to_dict()
    if isinstance(agent_cfg, dict) or agent_cfg is None:
        agent_cfg_dict = agent_cfg
    else:
        agent_cfg_dict = agent_cfg.to_dict()
    cfg_dict = {"env": env_cfg_dict, "agent": agent_cfg_dict}

    cfg_dict = replace_slices_with_strings(cfg_dict)

    ConfigStore.instance().store(name=task_name, node=cfg_dict)
    return env_cfg, agent_cfg


def hydra_task_config(task_name: str, agent_cfg_entry_point: str) -> Callable:
    """Decorator to handle the Hydra configuration for a task.

    This decorator registers the task to Hydra and updates the environment and agent configurations from Hydra parsed
    command line arguments.

    Args:
        task_name: The name of the task.
        agent_cfg_entry_point: The entry point key to resolve the agent's configuration file.

    Returns:
        The decorated function with the envrionment's and agent's configurations updated from command line arguments.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            env_cfg, agent_cfg = register_task_to_hydra(task_name.split(":")[-1], agent_cfg_entry_point)


            @hydra.main(config_path=None, config_name=task_name.split(":")[-1], version_base="1.3")
            def hydra_main(hydra_env_cfg: DictConfig, env_cfg=env_cfg, agent_cfg=agent_cfg):

                hydra_env_cfg = OmegaConf.to_container(hydra_env_cfg, resolve=True)

                hydra_env_cfg = replace_strings_with_slices(hydra_env_cfg)

                env_cfg.from_dict(hydra_env_cfg["env"])


                env_cfg = replace_strings_with_env_cfg_spaces(env_cfg)

                if isinstance(agent_cfg, dict) or agent_cfg is None:
                    agent_cfg = hydra_env_cfg["agent"]
                else:
                    agent_cfg.from_dict(hydra_env_cfg["agent"])

                func(env_cfg, agent_cfg, *args, **kwargs)


            hydra_main()

        return wrapper

    return decorator
