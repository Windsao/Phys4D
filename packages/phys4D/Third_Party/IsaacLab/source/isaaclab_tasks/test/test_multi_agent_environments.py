




"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher


app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import pytest
from env_test_utils import _check_random_actions, setup_environment

import isaaclab_tasks


@pytest.mark.parametrize("num_envs, device", [(32, "cuda"), (1, "cuda")])
@pytest.mark.parametrize("task_name", setup_environment(multi_agent=True))
def test_environments(task_name, num_envs, device):
    """Run all environments with given parameters and check environments return valid signals."""
    print(f">>> Running test for environment: {task_name} with num_envs={num_envs} and device={device}")

    _check_random_actions(task_name, device, num_envs, multi_agent=True)

    print(f">>> Closing environment: {task_name}")
    print("-" * 80)
