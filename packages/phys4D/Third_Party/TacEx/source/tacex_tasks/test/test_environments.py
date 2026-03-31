






"""Launch Isaac Sim Simulator first."""

import sys



if sys.platform != "win32":
    import pinocchio

from isaaclab.app import AppLauncher


app_launcher = AppLauncher(headless=True, enable_cameras=True)

simulation_app = app_launcher.app


"""Rest everything follows."""

import pytest
from env_test_utils import _run_environments, setup_environment


import tacex_tasks


@pytest.mark.parametrize("num_envs, device", [(32, "cuda"), (1, "cuda")])
@pytest.mark.parametrize("task_name", setup_environment(include_play=False, factory_envs=False, multi_agent=False))
def test_environments(task_name, num_envs, device):

    _run_environments(task_name, device, num_envs, create_stage_in_memory=False)
