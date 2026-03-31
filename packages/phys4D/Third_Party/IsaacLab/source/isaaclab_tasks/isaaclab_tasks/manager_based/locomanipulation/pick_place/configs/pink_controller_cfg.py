




"""Configuration for pink controller.

This module provides configurations for humanoid robot pink IK controllers,
including both fixed base and mobile configurations for upper body manipulation.
"""

from isaaclab.controllers.pink_ik.local_frame_task import LocalFrameTask
from isaaclab.controllers.pink_ik.null_space_posture_task import NullSpacePostureTask
from isaaclab.controllers.pink_ik.pink_ik_cfg import PinkIKControllerCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg





G1_UPPER_BODY_IK_CONTROLLER_CFG = PinkIKControllerCfg(
    articulation_name="robot",
    base_link_name="pelvis",
    num_hand_joints=14,
    show_ik_warnings=True,
    fail_on_joint_limit_violation=False,
    variable_input_tasks=[
        LocalFrameTask(
            "g1_29dof_with_hand_rev_1_0_left_wrist_yaw_link",
            base_link_frame_name="g1_29dof_with_hand_rev_1_0_pelvis",
            position_cost=8.0,
            orientation_cost=2.0,
            lm_damping=10,
            gain=0.5,
        ),
        LocalFrameTask(
            "g1_29dof_with_hand_rev_1_0_right_wrist_yaw_link",
            base_link_frame_name="g1_29dof_with_hand_rev_1_0_pelvis",
            position_cost=8.0,
            orientation_cost=2.0,
            lm_damping=10,
            gain=0.5,
        ),
        NullSpacePostureTask(
            cost=0.5,
            lm_damping=1,
            controlled_frames=[
                "g1_29dof_with_hand_rev_1_0_left_wrist_yaw_link",
                "g1_29dof_with_hand_rev_1_0_right_wrist_yaw_link",
            ],
            controlled_joints=[
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "waist_yaw_joint",
                "waist_pitch_joint",
                "waist_roll_joint",
            ],
            gain=0.3,
        ),
    ],
    fixed_input_tasks=[],
)
"""Base configuration for the G1 pink IK controller.

This configuration sets up the pink IK controller for the G1 humanoid robot with
left and right wrist control tasks. The controller is designed for upper body
manipulation tasks.
"""






G1_UPPER_BODY_IK_ACTION_CFG = PinkInverseKinematicsActionCfg(
    pink_controlled_joint_names=[
        ".*_shoulder_pitch_joint",
        ".*_shoulder_roll_joint",
        ".*_shoulder_yaw_joint",
        ".*_elbow_joint",
        ".*_wrist_pitch_joint",
        ".*_wrist_roll_joint",
        ".*_wrist_yaw_joint",
        "waist_.*_joint",
    ],
    hand_joint_names=[
        "left_hand_index_0_joint",
        "left_hand_middle_0_joint",
        "left_hand_thumb_0_joint",
        "right_hand_index_0_joint",
        "right_hand_middle_0_joint",
        "right_hand_thumb_0_joint",
        "left_hand_index_1_joint",
        "left_hand_middle_1_joint",
        "left_hand_thumb_1_joint",
        "right_hand_index_1_joint",
        "right_hand_middle_1_joint",
        "right_hand_thumb_1_joint",
        "left_hand_thumb_2_joint",
        "right_hand_thumb_2_joint",
    ],
    target_eef_link_names={
        "left_wrist": "left_wrist_yaw_link",
        "right_wrist": "right_wrist_yaw_link",
    },

    asset_name="robot",



    controller=G1_UPPER_BODY_IK_CONTROLLER_CFG,
)
"""Base configuration for the G1 pink IK action.

This configuration sets up the pink IK action for the G1 humanoid robot,
defining which joints are controlled by the IK solver and which are fixed.
The configuration includes:
- Upper body joints controlled by IK (shoulders, elbows, wrists)
- Fixed joints (pelvis, legs, hands)
- Hand joint names for additional control
- Reference to the pink IK controller configuration
"""
