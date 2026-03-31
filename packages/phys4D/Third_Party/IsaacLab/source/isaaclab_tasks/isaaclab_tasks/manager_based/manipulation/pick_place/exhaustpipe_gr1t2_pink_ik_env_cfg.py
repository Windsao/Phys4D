




import carb
from pink.tasks import DampingTask, FrameTask

import isaaclab.controllers.utils as ControllerUtils
from isaaclab.controllers.pink_ik import NullSpacePostureTask, PinkIKControllerCfg
from isaaclab.devices import DevicesCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg
from isaaclab.devices.openxr.retargeters import GR1T2RetargeterCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.pick_place.exhaustpipe_gr1t2_base_env_cfg import (
    ExhaustPipeGR1T2BaseEnvCfg,
)


@configclass
class ExhaustPipeGR1T2PinkIKEnvCfg(ExhaustPipeGR1T2BaseEnvCfg):
    def __post_init__(self):

        super().__post_init__()

        self.actions.gr1_action = PinkInverseKinematicsActionCfg(
            pink_controlled_joint_names=[
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_pitch_joint",
                "left_wrist_yaw_joint",
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_pitch_joint",
                "right_wrist_yaw_joint",
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
            ],
            hand_joint_names=[
                "L_index_proximal_joint",
                "L_middle_proximal_joint",
                "L_pinky_proximal_joint",
                "L_ring_proximal_joint",
                "L_thumb_proximal_yaw_joint",
                "R_index_proximal_joint",
                "R_middle_proximal_joint",
                "R_pinky_proximal_joint",
                "R_ring_proximal_joint",
                "R_thumb_proximal_yaw_joint",
                "L_index_intermediate_joint",
                "L_middle_intermediate_joint",
                "L_pinky_intermediate_joint",
                "L_ring_intermediate_joint",
                "L_thumb_proximal_pitch_joint",
                "R_index_intermediate_joint",
                "R_middle_intermediate_joint",
                "R_pinky_intermediate_joint",
                "R_ring_intermediate_joint",
                "R_thumb_proximal_pitch_joint",
                "L_thumb_distal_joint",
                "R_thumb_distal_joint",
            ],
            target_eef_link_names={
                "left_wrist": "left_hand_pitch_link",
                "right_wrist": "right_hand_pitch_link",
            },

            asset_name="robot",



            controller=PinkIKControllerCfg(
                articulation_name="robot",
                base_link_name="base_link",
                num_hand_joints=22,
                show_ik_warnings=False,
                fail_on_joint_limit_violation=False,
                variable_input_tasks=[
                    FrameTask(
                        "GR1T2_fourier_hand_6dof_left_hand_pitch_link",
                        position_cost=8.0,
                        orientation_cost=1.0,
                        lm_damping=10,
                        gain=0.5,
                    ),
                    FrameTask(
                        "GR1T2_fourier_hand_6dof_right_hand_pitch_link",
                        position_cost=8.0,
                        orientation_cost=1.0,
                        lm_damping=10,
                        gain=0.5,
                    ),
                    DampingTask(
                        cost=0.5,
                    ),
                    NullSpacePostureTask(
                        cost=0.2,
                        lm_damping=1,
                        controlled_frames=[
                            "GR1T2_fourier_hand_6dof_left_hand_pitch_link",
                            "GR1T2_fourier_hand_6dof_right_hand_pitch_link",
                        ],
                        controlled_joints=[
                            "left_shoulder_pitch_joint",
                            "left_shoulder_roll_joint",
                            "left_shoulder_yaw_joint",
                            "left_elbow_pitch_joint",
                            "right_shoulder_pitch_joint",
                            "right_shoulder_roll_joint",
                            "right_shoulder_yaw_joint",
                            "right_elbow_pitch_joint",
                            "waist_yaw_joint",
                            "waist_pitch_joint",
                            "waist_roll_joint",
                        ],
                    ),
                ],
                fixed_input_tasks=[],
                xr_enabled=bool(carb.settings.get_settings().get("/app/xr/enabled")),
            ),
        )

        temp_urdf_output_path, temp_urdf_meshes_output_path = ControllerUtils.convert_usd_to_urdf(
            self.scene.robot.spawn.usd_path, self.temp_urdf_dir, force_conversion=True
        )


        self.actions.gr1_action.controller.urdf_path = temp_urdf_output_path
        self.actions.gr1_action.controller.mesh_path = temp_urdf_meshes_output_path

        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        GR1T2RetargeterCfg(
                            enable_visualization=True,

                            num_open_xr_hand_joints=2 * self.NUM_OPENXR_HAND_JOINTS,
                            sim_device=self.sim.device,
                            hand_joint_names=self.actions.gr1_action.hand_joint_names,
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
            }
        )
