




from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from . import stack_joint_pos_instance_randomize_env_cfg




from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG


@configclass
class FrankaCubeStackInstanceRandomizeEnvCfg(
    stack_joint_pos_instance_randomize_env_cfg.FrankaCubeStackInstanceRandomizeEnvCfg
):
    def __post_init__(self):

        super().__post_init__()



        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


        self.scene.num_envs = 2


        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )
