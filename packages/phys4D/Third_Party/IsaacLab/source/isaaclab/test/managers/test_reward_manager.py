




"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher


simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import torch
from collections import namedtuple

import pytest

from isaaclab.managers import RewardManager, RewardTermCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass


def grilled_chicken(env):
    return 1


def grilled_chicken_with_bbq(env, bbq: bool):
    return 0


def grilled_chicken_with_curry(env, hot: bool):
    return 0


def grilled_chicken_with_yoghurt(env, hot: bool, bland: float):
    return 0


@pytest.fixture
def env():
    sim = SimulationContext()
    return namedtuple("ManagerBasedRLEnv", ["num_envs", "dt", "device", "sim"])(20, 0.1, "cpu", sim)


def test_str(env):
    """Test the string representation of the reward manager."""
    cfg = {
        "term_1": RewardTermCfg(func=grilled_chicken, weight=10),
        "term_2": RewardTermCfg(func=grilled_chicken_with_bbq, weight=5, params={"bbq": True}),
        "term_3": RewardTermCfg(
            func=grilled_chicken_with_yoghurt,
            weight=1.0,
            params={"hot": False, "bland": 2.0},
        ),
    }
    rew_man = RewardManager(cfg, env)
    assert len(rew_man.active_terms) == 3

    print()
    print(rew_man)


def test_config_equivalence(env):
    """Test the equivalence of reward manager created from different config types."""

    cfg = {
        "my_term": RewardTermCfg(func=grilled_chicken, weight=10),
        "your_term": RewardTermCfg(func=grilled_chicken_with_bbq, weight=2.0, params={"bbq": True}),
        "his_term": RewardTermCfg(
            func=grilled_chicken_with_yoghurt,
            weight=1.0,
            params={"hot": False, "bland": 2.0},
        ),
    }
    rew_man_from_dict = RewardManager(cfg, env)


    @configclass
    class MyRewardManagerCfg:
        """Reward manager config with no type annotations."""

        my_term = RewardTermCfg(func=grilled_chicken, weight=10.0)
        your_term = RewardTermCfg(func=grilled_chicken_with_bbq, weight=2.0, params={"bbq": True})
        his_term = RewardTermCfg(func=grilled_chicken_with_yoghurt, weight=1.0, params={"hot": False, "bland": 2.0})

    cfg = MyRewardManagerCfg()
    rew_man_from_cfg = RewardManager(cfg, env)


    @configclass
    class MyRewardManagerAnnotatedCfg:
        """Reward manager config with type annotations."""

        my_term: RewardTermCfg = RewardTermCfg(func=grilled_chicken, weight=10.0)
        your_term: RewardTermCfg = RewardTermCfg(func=grilled_chicken_with_bbq, weight=2.0, params={"bbq": True})
        his_term: RewardTermCfg = RewardTermCfg(
            func=grilled_chicken_with_yoghurt, weight=1.0, params={"hot": False, "bland": 2.0}
        )

    cfg = MyRewardManagerAnnotatedCfg()
    rew_man_from_annotated_cfg = RewardManager(cfg, env)



    assert rew_man_from_dict.active_terms == rew_man_from_annotated_cfg.active_terms
    assert rew_man_from_cfg.active_terms == rew_man_from_annotated_cfg.active_terms
    assert rew_man_from_dict.active_terms == rew_man_from_cfg.active_terms

    assert rew_man_from_dict._term_cfgs == rew_man_from_annotated_cfg._term_cfgs
    assert rew_man_from_cfg._term_cfgs == rew_man_from_annotated_cfg._term_cfgs
    assert rew_man_from_dict._term_cfgs == rew_man_from_cfg._term_cfgs


def test_compute(env):
    """Test the computation of reward."""
    cfg = {
        "term_1": RewardTermCfg(func=grilled_chicken, weight=10),
        "term_2": RewardTermCfg(func=grilled_chicken_with_curry, weight=0.0, params={"hot": False}),
    }
    rew_man = RewardManager(cfg, env)

    expected_reward = cfg["term_1"].weight * env.dt

    rewards = rew_man.compute(dt=env.dt)

    assert float(rewards[0]) == expected_reward
    assert tuple(rewards.shape) == (env.num_envs,)


def test_config_empty(env):
    """Test the creation of reward manager with empty config."""
    rew_man = RewardManager(None, env)
    assert len(rew_man.active_terms) == 0


    print()
    print(rew_man)


    rewards = rew_man.compute(dt=env.dt)


    torch.testing.assert_close(rewards, torch.zeros_like(rewards))


def test_active_terms(env):
    """Test the correct reading of active terms."""
    cfg = {
        "term_1": RewardTermCfg(func=grilled_chicken, weight=10),
        "term_2": RewardTermCfg(func=grilled_chicken_with_bbq, weight=5, params={"bbq": True}),
        "term_3": RewardTermCfg(func=grilled_chicken_with_curry, weight=0.0, params={"hot": False}),
    }
    rew_man = RewardManager(cfg, env)

    assert len(rew_man.active_terms) == 3


def test_missing_weight(env):
    """Test the missing of weight in the config."""

    cfg = {
        "term_1": RewardTermCfg(func=grilled_chicken, weight=10),
        "term_2": RewardTermCfg(func=grilled_chicken_with_bbq, params={"bbq": True}),
    }
    with pytest.raises(TypeError):
        RewardManager(cfg, env)


def test_invalid_reward_func_module(env):
    """Test the handling of invalid reward function's module in string representation."""
    cfg = {
        "term_1": RewardTermCfg(func=grilled_chicken, weight=10),
        "term_2": RewardTermCfg(func=grilled_chicken_with_bbq, weight=5, params={"bbq": True}),
        "term_3": RewardTermCfg(func="a:grilled_chicken_with_no_bbq", weight=0.1, params={"hot": False}),
    }
    with pytest.raises(ValueError):
        RewardManager(cfg, env)


def test_invalid_reward_config(env):
    """Test the handling of invalid reward function's config parameters."""
    cfg = {
        "term_1": RewardTermCfg(func=grilled_chicken_with_bbq, weight=0.1, params={"hot": False}),
        "term_2": RewardTermCfg(func=grilled_chicken_with_yoghurt, weight=2.0, params={"hot": False}),
    }
    with pytest.raises(ValueError):
        RewardManager(cfg, env)
