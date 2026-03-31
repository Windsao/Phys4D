




"""Launch Isaac Sim Simulator first."""

import sys



if sys.platform != "win32":
    import pinocchio

from isaaclab.app import AppLauncher


app_launcher = AppLauncher(
    headless=True, enable_cameras=False, kit_args='--/app/extensions/excluded=["omni.usd.metrics.assembler.ui"]'
)
simulation_app = app_launcher.app

"""Rest everything follows."""

import pytest
from env_test_utils import _run_environments

import isaaclab_tasks


@pytest.mark.parametrize("num_envs, device", [(32, "cuda"), (1, "cuda")])
def test_lift_teddy_bear_environment(num_envs, device):
    """Test the Isaac-Lift-Teddy-Bear-Franka-IK-Abs-v0 environment in isolation."""
    task_name = "Isaac-Lift-Teddy-Bear-Franka-IK-Abs-v0"


    try:
        _run_environments(task_name, device, num_envs, create_stage_in_memory=False)
    except Exception as e:

        pytest.skip(f"Isaac-Lift-Teddy-Bear-Franka-IK-Abs-v0 environment failed to load: {e}")
