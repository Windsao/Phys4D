




"""Launch Isaac Sim Simulator first."""

import sys

from isaaclab.app import AppLauncher


app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import functools
from collections.abc import Callable

import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf

from isaaclab.utils import replace_strings_with_slices

import isaaclab_tasks
from isaaclab_tasks.utils.hydra import register_task_to_hydra


def hydra_task_config_test(task_name: str, agent_cfg_entry_point: str) -> Callable:
    """Copied from hydra.py hydra_task_config, since hydra.main requires a single point of entry,
    which will not work with multiple tests. Here, we replace hydra.main with hydra initialize
    and compose."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            env_cfg, agent_cfg = register_task_to_hydra(task_name, agent_cfg_entry_point)


            with initialize(config_path=None, version_base="1.3"):
                hydra_env_cfg = compose(config_name=task_name, overrides=sys.argv[1:])

                hydra_env_cfg = OmegaConf.to_container(hydra_env_cfg, resolve=True)

                hydra_env_cfg = replace_strings_with_slices(hydra_env_cfg)

                env_cfg.from_dict(hydra_env_cfg["env"])
                if isinstance(agent_cfg, dict):
                    agent_cfg = hydra_env_cfg["agent"]
                else:
                    agent_cfg.from_dict(hydra_env_cfg["agent"])

                func(env_cfg, agent_cfg, *args, **kwargs)

        return wrapper

    return decorator


def test_hydra():
    """Test the hydra configuration system."""


    sys.argv = [
        sys.argv[0],
        "env.decimation=42",
        "env.events.physics_material.params.asset_cfg.joint_ids='slice(0 ,1, 2)'",
        "env.scene.robot.init_state.joint_vel={.*: 4.0}",
        "env.rewards.feet_air_time=null",
        "agent.max_iterations=3",
    ]

    @hydra_task_config_test("Isaac-Velocity-Flat-H1-v0", "rsl_rl_cfg_entry_point")
    def main(env_cfg, agent_cfg):

        assert env_cfg.decimation == 42
        assert env_cfg.events.physics_material.params["asset_cfg"].joint_ids == slice(0, 1, 2)
        assert env_cfg.scene.robot.init_state.joint_vel == {".*": 4.0}
        assert env_cfg.rewards.feet_air_time is None

        assert agent_cfg.max_iterations == 3

    main()

    sys.argv = [sys.argv[0]]
    hydra.core.global_hydra.GlobalHydra.instance().clear()


def test_nested_iterable_dict():
    """Test the hydra configuration system when dict is nested in an Iterable."""

    @hydra_task_config_test("Isaac-Lift-Cube-Franka-v0", "rsl_rl_cfg_entry_point")
    def main(env_cfg, agent_cfg):

        assert env_cfg.scene.ee_frame.target_frames[0].name == "end_effector"
        assert env_cfg.scene.ee_frame.target_frames[0].offset.pos[2] == 0.1034

    main()

    sys.argv = [sys.argv[0]]
    hydra.core.global_hydra.GlobalHydra.instance().clear()
