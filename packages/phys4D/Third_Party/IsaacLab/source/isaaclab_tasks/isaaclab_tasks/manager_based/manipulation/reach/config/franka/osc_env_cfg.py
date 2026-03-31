




from isaaclab.controllers.operational_space_cfg import OperationalSpaceControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import OperationalSpaceControllerActionCfg
from isaaclab.utils import configclass

from . import joint_pos_env_cfg




from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG


@configclass
class FrankaReachEnvCfg(joint_pos_env_cfg.FrankaReachEnvCfg):
    def __post_init__(self):

        super().__post_init__()



        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.actuators["panda_shoulder"].stiffness = 0.0
        self.scene.robot.actuators["panda_shoulder"].damping = 0.0
        self.scene.robot.actuators["panda_forearm"].stiffness = 0.0
        self.scene.robot.actuators["panda_forearm"].damping = 0.0
        self.scene.robot.spawn.rigid_props.disable_gravity = True




        self.actions.arm_action = OperationalSpaceControllerActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",



            controller_cfg=OperationalSpaceControllerCfg(
                target_types=["pose_abs"],
                impedance_mode="variable_kp",
                inertial_dynamics_decoupling=True,
                partial_inertial_dynamics_decoupling=False,
                gravity_compensation=False,
                motion_stiffness_task=100.0,
                motion_damping_ratio_task=1.0,
                motion_stiffness_limits_task=(50.0, 200.0),
                nullspace_control="position",
            ),
            nullspace_joint_pos_target="center",
            position_scale=1.0,
            orientation_scale=1.0,
            stiffness_scale=100.0,
        )

        self.observations.policy.joint_pos = None
        self.observations.policy.joint_vel = None


@configclass
class FrankaReachEnvCfg_PLAY(FrankaReachEnvCfg):
    def __post_init__(self):

        super().__post_init__()

        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5

        self.observations.policy.enable_corruption = False
