




"""Launch Isaac Sim Simulator first."""

import sys



if sys.platform != "win32":
    import pinocchio

from isaaclab.app import AppLauncher


app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

from isaacsim.core.version import get_version

"""Rest everything follows."""

import pytest
from env_test_utils import _run_environments, setup_environment

import isaaclab_tasks


















@pytest.mark.parametrize("num_envs, device", [(2, "cuda")])
@pytest.mark.parametrize("task_name", setup_environment(include_play=False, factory_envs=False, multi_agent=False))
def test_environments_with_stage_in_memory_and_clone_in_fabric_disabled(task_name, num_envs, device):

    isaac_sim_version = float(".".join(get_version()[2]))
    if isaac_sim_version < 5:
        pytest.skip("Stage in memory is not supported in this version of Isaac Sim")


    _run_environments(task_name, device, num_envs, create_stage_in_memory=True, disable_clone_in_fabric=True)
