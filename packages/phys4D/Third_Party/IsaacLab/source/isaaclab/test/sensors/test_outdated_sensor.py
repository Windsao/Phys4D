



"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher


simulation_app = AppLauncher(headless=True, enable_cameras=True).app


"""Rest everything follows."""

import gymnasium as gym
import shutil
import tempfile
import torch

import carb
import omni.usd
import pytest

import isaaclab_tasks
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


@pytest.fixture()
def temp_dir():
    """Fixture to create and clean up a temporary directory for test datasets."""


    carb_settings_iface = carb.settings.get_settings()
    carb_settings_iface.set_bool("/physics/cooking/ujitsoCollisionCooking", False)

    temp_dir = tempfile.mkdtemp()
    yield temp_dir

    shutil.rmtree(temp_dir)


@pytest.mark.parametrize("task_name", ["Isaac-Stack-Cube-Franka-IK-Rel-v0"])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 2])
@pytest.mark.isaacsim_ci
def test_action_state_recorder_terms(temp_dir, task_name, device, num_envs):
    """Check FrameTransformer values after reset."""
    omni.usd.get_context().new_stage()


    env_cfg = parse_env_cfg(task_name, device=device, num_envs=num_envs)
    env_cfg.wait_for_textures = False


    env = gym.make(task_name, cfg=env_cfg)


    env.unwrapped.sim._app_control_on_stop_handle = None


    obs = env.reset()[0]


    pre_reset_eef_pos = obs["policy"]["eef_pos"].clone()
    print(pre_reset_eef_pos)


    idle_actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    obs = env.step(idle_actions)[0]


    post_reset_eef_pos = obs["policy"]["eef_pos"]
    print(post_reset_eef_pos)


    torch.testing.assert_close(pre_reset_eef_pos, post_reset_eef_pos, atol=1e-5, rtol=1e-3)


    env.close()
