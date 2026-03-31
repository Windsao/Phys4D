




import math

from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg




from isaaclab_assets import FRANKA_PANDA_CFG







@configclass
class FrankaReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):

        super().__post_init__()


        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["panda_hand"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["panda_hand"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["panda_hand"]


        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )


        self.commands.ee_pose.body_name = "panda_hand"
        self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)


@configclass
class FrankaReachEnvCfg_PLAY(FrankaReachEnvCfg):
    def __post_init__(self):

        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        self.observations.policy.enable_corruption = False
