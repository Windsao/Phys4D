




from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from . import joint_pos_env_cfg




from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG


@configclass
class FrankaReachEnvCfg(joint_pos_env_cfg.FrankaReachEnvCfg):
    def __post_init__(self):

        super().__post_init__()



        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )


@configclass
class FrankaReachEnvCfg_PLAY(FrankaReachEnvCfg):
    def __post_init__(self):

        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        self.observations.policy.enable_corruption = False
