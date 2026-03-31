




"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher


app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
import torch

import carb
import omni.usd
import pytest

import isaaclab_tasks
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


@pytest.fixture(scope="module", autouse=True)
def setup_environment():


    carb_settings_iface = carb.settings.get_settings()
    carb_settings_iface.set_bool("/physics/cooking/ujitsoCollisionCooking", False)


@pytest.mark.parametrize(
    "task_name",
    [
        "Isaac-Open-Drawer-Franka-v0",
        "Isaac-Lift-Cube-Franka-v0",
    ],
)
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_manipulation_env_determinism(task_name, device):
    """Check deterministic environment creation for manipulation."""
    _test_environment_determinism(task_name, device)


@pytest.mark.parametrize(
    "task_name",
    [
        "Isaac-Velocity-Flat-Anymal-C-v0",
        "Isaac-Velocity-Rough-Anymal-C-v0",
        "Isaac-Velocity-Rough-Anymal-C-Direct-v0",
    ],
)
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_locomotion_env_determinism(task_name, device):
    """Check deterministic environment creation for locomotion."""
    _test_environment_determinism(task_name, device)


@pytest.mark.parametrize(
    "task_name",
    [
        "Isaac-Repose-Cube-Allegro-v0",

    ],
)
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_dextrous_env_determinism(task_name, device):
    """Check deterministic environment creation for dextrous manipulation."""
    _test_environment_determinism(task_name, device)


def _test_environment_determinism(task_name: str, device: str):
    """Check deterministic environment creation."""

    num_envs = 32
    num_steps = 100

    obs_1, rew_1 = _obtain_transition_tuples(task_name, num_envs, device, num_steps)
    obs_2, rew_2 = _obtain_transition_tuples(task_name, num_envs, device, num_steps)



    torch.testing.assert_close(rew_1, rew_2)

    for key in obs_1.keys():
        torch.testing.assert_close(obs_1[key], obs_2[key])


def _obtain_transition_tuples(task_name: str, num_envs: int, device: str, num_steps: int) -> tuple[dict, torch.Tensor]:
    """Run random actions and obtain transition tuples after fixed number of steps."""

    omni.usd.get_context().new_stage()
    try:

        env_cfg = parse_env_cfg(task_name, device=device, num_envs=num_envs)

        env_cfg.seed = 42

        env = gym.make(task_name, cfg=env_cfg)
    except Exception as e:
        if "env" in locals() and hasattr(env, "_is_closed"):
            env.close()
        else:
            if hasattr(e, "obj") and hasattr(e.obj, "_is_closed"):
                e.obj.close()
        pytest.fail(f"Failed to set-up the environment for task {task_name}. Error: {e}")


    env.unwrapped.sim._app_control_on_stop_handle = None


    obs, _ = env.reset()

    with torch.inference_mode():
        for _ in range(num_steps):

            actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1

            obs, rewards = env.step(actions)[:2]


    env.close()

    return obs, rewards
