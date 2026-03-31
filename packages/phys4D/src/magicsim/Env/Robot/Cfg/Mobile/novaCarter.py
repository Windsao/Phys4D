from typing import Dict
import torch
from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.managers import ActionTermCfg as ActionTerm

from magicsim.Env.Robot.Cfg.Mobile.Mobile import (
    MobileActionsCfg,
    MobileCfg,
    MobileObsCfg,
)
import magicsim.Env.Robot.mdp as mdp

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.actuators import ImplicitActuatorCfg

NOVA_CARTER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/NovaCarter/nova_carter.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "joint_wheel_left": 0.0,
            "joint_wheel_right": 0.0,
        },
    ),
    actuators={
        "wheel_drive": ImplicitActuatorCfg(
            joint_names_expr=["joint_wheel_.*"],
            effort_limit_sim=100.0,
            velocity_limit_sim=20.0,
            stiffness=0.0,
            damping=10.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of the Nova Carter mobile robot."""


@configclass
class NovaCarterActionsCfg(MobileActionsCfg):
    """Action specifications for the MDP."""

    available_action: Dict[str, Dict[str, ActionTerm]] = {
        "base_action": {
            "differential_drive": mdp.DifferentialActionCfg(
                action_space=torch.tensor(
                    [
                        [-10, -10],
                        [10, 10],
                    ]
                ),
                joint_names=["joint_wheel_left", "joint_wheel_right"],
            ),
        },
    }

    def __post_init__(self):
        super().__post_init__()


@configclass
class NovaCarterCfg(MobileCfg):
    """Configuration for a mobile manipulator: Ridgeback + Franka"""

    prim_path: str = MISSING
    asset_name: str = "robot"
    base_action_name: str = MISSING
    robot: ArticulationCfg = MISSING

    action: NovaCarterActionsCfg = MISSING
    obs: MobileObsCfg = MISSING

    def __post_init__(self):
        self.robot: ArticulationCfg = NOVA_CARTER_CFG

        self.robot.prim_path = self.prim_path

        self.action = NovaCarterActionsCfg(
            asset_name=self.asset_name,
            base_action_name=self.base_action_name,
        )

        self.obs = MobileObsCfg(asset_name=self.asset_name)
