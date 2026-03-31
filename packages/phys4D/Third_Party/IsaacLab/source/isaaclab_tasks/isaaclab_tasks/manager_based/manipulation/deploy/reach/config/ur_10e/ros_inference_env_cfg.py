




from isaaclab.utils import configclass

from .joint_pos_env_cfg import UR10eReachEnvCfg


@configclass
class UR10eReachROSInferenceEnvCfg(UR10eReachEnvCfg):
    """Exposing variables for ROS inferences"""

    def __post_init__(self):

        super().__post_init__()



        self.obs_order = ["arm_dof_pos", "arm_dof_vel", "target_pos", "target_quat"]
        self.policy_action_space = "joint"
        self.arm_joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        self.policy_action_space = "joint"
        self.action_space = 6
        self.state_space = 0
        self.observation_space = 19


        self.joint_action_scale = self.actions.arm_action.scale

        self.action_scale_joint_space = [
            self.joint_action_scale,
            self.joint_action_scale,
            self.joint_action_scale,
            self.joint_action_scale,
            self.joint_action_scale,
            self.joint_action_scale,
        ]
