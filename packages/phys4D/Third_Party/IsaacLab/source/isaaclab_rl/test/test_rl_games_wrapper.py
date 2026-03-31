




"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher


app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
import os
import torch

import carb
import omni.usd
import pytest

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent

from isaaclab_rl.rl_games import RlGamesVecEnvWrapper

import isaaclab_tasks
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


@pytest.fixture(scope="module")
def registered_tasks():

    os.environ["WANDB_DISABLED"] = "true"

    registered_tasks = list()
    for task_spec in gym.registry.values():
        if "Isaac" in task_spec.id:
            cfg_entry_point = gym.spec(task_spec.id).kwargs.get("rl_games_cfg_entry_point")
            if cfg_entry_point is not None:

                if "assembly" in task_spec.id.lower():
                    continue
                registered_tasks.append(task_spec.id)

    registered_tasks.sort()
    registered_tasks = registered_tasks[:5]



    carb_settings_iface = carb.settings.get_settings()
    carb_settings_iface.set_bool("/physics/cooking/ujitsoCollisionCooking", False)


    print(">>> All registered environments:", registered_tasks)
    return registered_tasks


def test_random_actions(registered_tasks):
    """Run random actions and check environments return valid signals."""

    num_envs = 64
    device = "cuda"
    for task_name in registered_tasks:

        print(f">>> Running test for environment: {task_name}")

        omni.usd.get_context().new_stage()

        carb.settings.get_settings().set_bool("/isaaclab/render/rtx_sensors", False)
        try:

            env_cfg = parse_env_cfg(task_name, device=device, num_envs=num_envs)

            env = gym.make(task_name, cfg=env_cfg)

            if isinstance(env.unwrapped, DirectMARLEnv):
                env = multi_agent_to_single_agent(env)

            env = RlGamesVecEnvWrapper(env, "cuda:0", 100, 100)
        except Exception as e:
            if "env" in locals() and hasattr(env, "_is_closed"):
                env.close()
            else:
                if hasattr(e, "obj") and hasattr(e.obj, "_is_closed"):
                    e.obj.close()
            pytest.fail(f"Failed to set-up the environment for task {task_name}. Error: {e}")


        env.unwrapped.sim._app_control_on_stop_handle = None


        obs = env.reset()

        assert _check_valid_tensor(obs)


        with torch.inference_mode():
            for _ in range(100):

                actions = 2 * torch.rand(env.num_envs, *env.action_space.shape, device=env.device) - 1

                transition = env.step(actions)

                for data in transition:
                    assert _check_valid_tensor(data), f"Invalid data: {data}"


        print(f">>> Closing environment: {task_name}")
        env.close()


"""
Helper functions.
"""


@staticmethod
def _check_valid_tensor(data: torch.Tensor | dict) -> bool:
    """Checks if given data does not have corrupted values.

    Args:
        data: Data buffer.

    Returns:
        True if the data is valid.
    """
    if isinstance(data, torch.Tensor):
        return not torch.any(torch.isnan(data))
    elif isinstance(data, dict):
        valid_tensor = True
        for value in data.values():
            if isinstance(value, dict):
                valid_tensor &= _check_valid_tensor(value)
            elif isinstance(value, torch.Tensor):
                valid_tensor &= not torch.any(torch.isnan(value))
        return valid_tensor
    else:
        raise ValueError(f"Input data of invalid type: {type(data)}.")
