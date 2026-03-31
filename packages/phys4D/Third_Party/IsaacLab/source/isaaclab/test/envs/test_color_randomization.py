




"""
This script tests the functionality of texture randomization applied to the cartpole scene.
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher


app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import torch

import omni.usd
import pytest
from isaacsim.core.version import get_version

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import CartpoleSceneCfg


@configclass
class ActionsCfg:
    """Action specifications for the environment."""

    joint_efforts = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=5.0)


@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""


        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True


    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""




    cart_texture_randomizer = EventTerm(
        func=mdp.randomize_visual_color,
        mode="prestartup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["cart"]),
            "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
            "event_name": "cart_color_randomizer",
        },
    )


    pole_texture_randomizer = EventTerm(
        func=mdp.randomize_visual_color,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["pole"]),
            "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
            "event_name": "pole_color_randomizer",
        },
    )

    reset_cart_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.1, 0.1),
        },
    )

    reset_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "position_range": (-0.125 * math.pi, 0.125 * math.pi),
            "velocity_range": (-0.01 * math.pi, 0.01 * math.pi),
        },
    )


@configclass
class CartpoleEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the cartpole environment."""


    scene = CartpoleSceneCfg(env_spacing=2.5)


    actions = ActionsCfg()
    observations = ObservationsCfg()
    events = EventCfg()

    def __post_init__(self):
        """Post initialization."""

        self.viewer.eye = [4.5, 0.0, 6.0]
        self.viewer.lookat = [0.0, 0.0, 2.0]

        self.decimation = 4

        self.sim.dt = 0.005


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_color_randomization(device):
    """Test color randomization for cartpole environment."""

    isaac_sim_version = float(".".join(get_version()[2]))
    if isaac_sim_version < 5:
        pytest.skip("Color randomization test hangs in this version of Isaac Sim")


    omni.usd.get_context().new_stage()

    try:

        env_cfg = CartpoleEnvCfg()
        env_cfg.scene.num_envs = 16
        env_cfg.scene.replicate_physics = False
        env_cfg.sim.device = device


        env = ManagerBasedEnv(cfg=env_cfg)

        try:

            with torch.inference_mode():
                for count in range(50):

                    if count % 10 == 0:
                        env.reset()

                    joint_efforts = torch.randn_like(env.action_manager.action)

                    env.step(joint_efforts)
        finally:
            env.close()
    finally:

        omni.usd.get_context().close_stage()
