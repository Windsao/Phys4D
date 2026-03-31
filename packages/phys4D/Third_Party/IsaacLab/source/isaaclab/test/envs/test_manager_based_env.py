







from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher


simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import torch

import omni.usd
import pytest

from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass


@configclass
class EmptyManagerCfg:
    """Empty manager specifications for the environment."""

    pass


@configclass
class EmptyObservationWithHistoryCfg:
    """Empty observation with history specifications for the environment."""

    @configclass
    class EmptyObservationGroupWithHistoryCfg(ObsGroup):
        """Empty observation with history specifications for the environment."""

        dummy_term: ObsTerm = ObsTerm(func=lambda env: torch.randn(env.num_envs, 1, device=env.device))

        def __post_init__(self):
            self.history_length = 5

    empty_observation: EmptyObservationGroupWithHistoryCfg = EmptyObservationGroupWithHistoryCfg()


@configclass
class EmptySceneCfg(InteractiveSceneCfg):
    """Configuration for an empty scene."""

    pass


def get_empty_base_env_cfg(device: str = "cuda:0", num_envs: int = 1, env_spacing: float = 1.0):
    """Generate base environment config based on device"""

    @configclass
    class EmptyEnvCfg(ManagerBasedEnvCfg):
        """Configuration for the empty test environment."""


        scene: EmptySceneCfg = EmptySceneCfg(num_envs=num_envs, env_spacing=env_spacing)

        actions: EmptyManagerCfg = EmptyManagerCfg()
        observations: EmptyManagerCfg = EmptyManagerCfg()

        def __post_init__(self):
            """Post initialization."""

            self.decimation = 4

            self.sim.dt = 0.005
            self.sim.render_interval = self.decimation

            self.sim.device = device

    return EmptyEnvCfg()


def get_empty_base_env_cfg_with_history(device: str = "cuda:0", num_envs: int = 1, env_spacing: float = 1.0):
    """Generate base environment config based on device"""

    @configclass
    class EmptyEnvWithHistoryCfg(ManagerBasedEnvCfg):
        """Configuration for the empty test environment."""


        scene: EmptySceneCfg = EmptySceneCfg(num_envs=num_envs, env_spacing=env_spacing)

        actions: EmptyManagerCfg = EmptyManagerCfg()
        observations: EmptyObservationWithHistoryCfg = EmptyObservationWithHistoryCfg()

        def __post_init__(self):
            """Post initialization."""

            self.decimation = 4

            self.sim.dt = 0.005
            self.sim.render_interval = self.decimation

            self.sim.device = device

    return EmptyEnvWithHistoryCfg()


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_initialization(device):
    """Test initialization of ManagerBasedEnv."""

    omni.usd.get_context().new_stage()

    env = ManagerBasedEnv(cfg=get_empty_base_env_cfg(device=device))

    assert env.action_manager.total_action_dim == 0
    assert len(env.action_manager.active_terms) == 0
    assert len(env.action_manager.action_term_dim) == 0

    assert len(env.observation_manager.active_terms) == 0
    assert len(env.observation_manager.group_obs_dim) == 0
    assert len(env.observation_manager.group_obs_term_dim) == 0
    assert len(env.observation_manager.group_obs_concatenate) == 0

    act = torch.randn_like(env.action_manager.action)

    for _ in range(2):
        obs, ext = env.step(action=act)

    env.close()


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_observation_history_changes_only_after_step(device):
    """Test observation history of ManagerBasedEnv.

    The history buffer should only change after a step is taken.
    """

    omni.usd.get_context().new_stage()

    env = ManagerBasedEnv(cfg=get_empty_base_env_cfg_with_history(device=device))


    for group_name in env.observation_manager._group_obs_term_names:
        group_term_names = env.observation_manager._group_obs_term_names[group_name]
        for term_name in group_term_names:
            torch.testing.assert_close(
                env.observation_manager._group_obs_term_history_buffer[group_name][term_name].current_length,
                torch.zeros((env.num_envs,), device=device, dtype=torch.int64),
            )


    env.observation_manager.compute()
    for group_name in env.observation_manager._group_obs_term_names:
        group_term_names = env.observation_manager._group_obs_term_names[group_name]
        for term_name in group_term_names:
            torch.testing.assert_close(
                env.observation_manager._group_obs_term_history_buffer[group_name][term_name].current_length,
                torch.zeros((env.num_envs,), device=device, dtype=torch.int64),
            )


    act = torch.randn_like(env.action_manager.action)
    env.step(act)
    group_obs = dict()
    for group_name in env.observation_manager._group_obs_term_names:
        group_term_names = env.observation_manager._group_obs_term_names[group_name]
        group_obs[group_name] = dict()
        for term_name in group_term_names:
            torch.testing.assert_close(
                env.observation_manager._group_obs_term_history_buffer[group_name][term_name].current_length,
                torch.ones((env.num_envs,), device=device, dtype=torch.int64),
            )
            group_obs[group_name][term_name] = env.observation_manager._group_obs_term_history_buffer[group_name][
                term_name
            ].buffer


    env.observation_manager.compute()
    for group_name in env.observation_manager._group_obs_term_names:
        group_term_names = env.observation_manager._group_obs_term_names[group_name]
        for term_name in group_term_names:
            torch.testing.assert_close(
                env.observation_manager._group_obs_term_history_buffer[group_name][term_name].current_length,
                torch.ones((env.num_envs,), device=device, dtype=torch.int64),
            )
            assert torch.allclose(
                group_obs[group_name][term_name],
                env.observation_manager._group_obs_term_history_buffer[group_name][term_name].buffer,
            )


    env.close()
