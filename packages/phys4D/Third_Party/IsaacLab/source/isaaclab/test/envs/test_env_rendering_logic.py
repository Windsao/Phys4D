




"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher



simulation_app = AppLauncher(headless=True, enable_cameras=True).app

"""Rest everything follows."""

import torch

import omni.usd
import pytest

from isaaclab.envs import (
    DirectRLEnv,
    DirectRLEnvCfg,
    ManagerBasedEnv,
    ManagerBasedEnvCfg,
    ManagerBasedRLEnv,
    ManagerBasedRLEnvCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass


@configclass
class EmptyManagerCfg:
    """Empty specifications for the environment."""

    pass


def create_manager_based_env(render_interval: int):
    """Create a manager based environment."""

    @configclass
    class EnvCfg(ManagerBasedEnvCfg):
        """Configuration for the test environment."""

        decimation: int = 4
        episode_length_s: float = 100.0
        sim: SimulationCfg = SimulationCfg(dt=0.005, render_interval=render_interval)
        scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=1.0)
        actions: EmptyManagerCfg = EmptyManagerCfg()
        observations: EmptyManagerCfg = EmptyManagerCfg()

    return ManagerBasedEnv(cfg=EnvCfg())


def create_manager_based_rl_env(render_interval: int):
    """Create a manager based RL environment."""

    @configclass
    class EnvCfg(ManagerBasedRLEnvCfg):
        """Configuration for the test environment."""

        decimation: int = 4
        episode_length_s: float = 100.0
        sim: SimulationCfg = SimulationCfg(dt=0.005, render_interval=render_interval)
        scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=1.0)
        actions: EmptyManagerCfg = EmptyManagerCfg()
        observations: EmptyManagerCfg = EmptyManagerCfg()
        rewards: EmptyManagerCfg = EmptyManagerCfg()
        terminations: EmptyManagerCfg = EmptyManagerCfg()

    return ManagerBasedRLEnv(cfg=EnvCfg())


def create_direct_rl_env(render_interval: int):
    """Create a direct RL environment."""

    @configclass
    class EnvCfg(DirectRLEnvCfg):
        """Configuration for the test environment."""

        decimation: int = 4
        action_space: int = 0
        observation_space: int = 0
        episode_length_s: float = 100.0
        sim: SimulationCfg = SimulationCfg(dt=0.005, render_interval=render_interval)
        scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=1.0)

    class Env(DirectRLEnv):
        """Test environment."""

        def _pre_physics_step(self, actions):
            pass

        def _apply_action(self):
            pass

        def _get_observations(self):
            return {}

        def _get_rewards(self):
            return {}

        def _get_dones(self):
            return torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool)

    return Env(cfg=EnvCfg())


@pytest.fixture
def physics_callback():
    """Create a physics callback for tracking physics steps."""
    physics_time = 0.0
    num_physics_steps = 0

    def callback(dt):
        nonlocal physics_time, num_physics_steps
        physics_time += dt
        num_physics_steps += 1

    return callback, lambda: (physics_time, num_physics_steps)


@pytest.fixture
def render_callback():
    """Create a render callback for tracking render steps."""
    render_time = 0.0
    num_render_steps = 0

    def callback(event):
        nonlocal render_time, num_render_steps
        render_time += event.payload["dt"]
        num_render_steps += 1

    return callback, lambda: (render_time, num_render_steps)


@pytest.mark.parametrize("env_type", ["manager_based_env", "manager_based_rl_env", "direct_rl_env"])
@pytest.mark.parametrize("render_interval", [1, 2, 4, 8, 10])
def test_env_rendering_logic(env_type, render_interval, physics_callback, render_callback):
    """Test the rendering logic of the different environment workflows."""
    physics_cb, get_physics_stats = physics_callback
    render_cb, get_render_stats = render_callback


    omni.usd.get_context().new_stage()
    try:

        if env_type == "manager_based_env":
            env = create_manager_based_env(render_interval)
        elif env_type == "manager_based_rl_env":
            env = create_manager_based_rl_env(render_interval)
        else:
            env = create_direct_rl_env(render_interval)
    except Exception as e:
        if "env" in locals() and hasattr(env, "_is_closed"):
            env.close()
        else:
            if hasattr(e, "obj") and hasattr(e.obj, "_is_closed"):
                e.obj.close()
        pytest.fail(f"Failed to set-up the environment {env_type}. Error: {e}")




    env.sim.set_setting("/isaaclab/render/rtx_sensors", True)




    env.sim._app_control_on_stop_handle = None



    assert env.sim.render_mode == SimulationContext.RenderMode.PARTIAL_RENDERING


    env.sim.add_physics_callback("physics_step", physics_cb)
    env.sim.add_render_callback("render_step", render_cb)


    actions = torch.zeros((env.num_envs, 0), device=env.device)


    for i in range(50):

        env.step(action=actions)


        _, num_physics_steps = get_physics_stats()
        assert num_physics_steps == (i + 1) * env.cfg.decimation, "Physics steps mismatch"

        physics_time, _ = get_physics_stats()
        assert abs(physics_time - num_physics_steps * env.cfg.sim.dt) < 1e-6, "Physics time mismatch"


        _, num_render_steps = get_render_stats()
        assert num_render_steps == (i + 1) * env.cfg.decimation // env.cfg.sim.render_interval, "Render steps mismatch"

        render_time, _ = get_render_stats()
        assert (
            abs(render_time - num_render_steps * env.cfg.sim.dt * env.cfg.sim.render_interval) < 1e-6
        ), "Render time mismatch"


    env.close()
