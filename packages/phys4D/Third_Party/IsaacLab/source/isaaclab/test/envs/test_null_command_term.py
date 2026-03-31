




"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher


simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

from collections import namedtuple

import pytest

from isaaclab.envs.mdp import NullCommandCfg


@pytest.fixture
def env():
    """Create a dummy environment."""
    return namedtuple("ManagerBasedRLEnv", ["num_envs", "dt", "device"])(20, 0.1, "cpu")


def test_str(env):
    """Test the string representation of the command manager."""
    cfg = NullCommandCfg()
    command_term = cfg.class_type(cfg, env)

    print()
    print(command_term)


def test_compute(env):
    """Test the compute function. For null command generator, it does nothing."""
    cfg = NullCommandCfg()
    command_term = cfg.class_type(cfg, env)


    command_term.reset()

    command_term.compute(dt=env.dt)

    with pytest.raises(RuntimeError):
        command_term.command
