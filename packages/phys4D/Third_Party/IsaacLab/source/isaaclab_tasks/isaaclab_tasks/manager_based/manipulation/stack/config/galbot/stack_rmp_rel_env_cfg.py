





import os

import isaaclab.sim as sim_utils
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.openxr import OpenXRDevice, OpenXRDeviceCfg
from isaaclab.devices.openxr.retargeters import GripperRetargeterCfg, Se3RelRetargeterCfg
from isaaclab.devices.spacemouse import Se3SpaceMouseCfg
from isaaclab.envs.mdp.actions.rmpflow_actions_cfg import RMPFlowActionCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.stack import mdp

from . import stack_joint_pos_env_cfg




from isaaclab.controllers.config.rmp_flow import (
    GALBOT_LEFT_ARM_RMPFLOW_CFG,
    GALBOT_RIGHT_ARM_RMPFLOW_CFG,
)
from isaaclab.markers.config import FRAME_MARKER_CFG





@configclass
class RmpFlowGalbotLeftArmCubeStackEnvCfg(stack_joint_pos_env_cfg.GalbotLeftArmCubeStackEnvCfg):

    def __post_init__(self):

        super().__post_init__()



        use_relative_mode_env = os.getenv("USE_RELATIVE_MODE", "True")
        self.use_relative_mode = use_relative_mode_env.lower() in ["true", "1", "t"]


        self.actions.arm_action = RMPFlowActionCfg(
            asset_name="robot",
            joint_names=["left_arm_joint.*"],
            body_name="left_gripper_tcp_link",
            controller=GALBOT_LEFT_ARM_RMPFLOW_CFG,
            scale=1.0,
            body_offset=RMPFlowActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
            articulation_prim_expr="/World/envs/env_.*/Robot",
            use_relative_mode=self.use_relative_mode,
        )


        self.sim.dt = 1 / 60
        self.sim.render_interval = 6

        self.decimation = 3
        self.episode_length_s = 30.0

        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
                "spacemouse": Se3SpaceMouseCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        Se3RelRetargeterCfg(
                            bound_hand=OpenXRDevice.TrackingTarget.HAND_LEFT,
                            zero_out_xy_rotation=True,
                            use_wrist_rotation=False,
                            use_wrist_position=True,
                            delta_pos_scale_factor=10.0,
                            delta_rot_scale_factor=10.0,
                            sim_device=self.sim.device,
                        ),
                        GripperRetargeterCfg(
                            bound_hand=OpenXRDevice.TrackingTarget.HAND_LEFT, sim_device=self.sim.device
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
            }
        )





@configclass
class RmpFlowGalbotRightArmCubeStackEnvCfg(stack_joint_pos_env_cfg.GalbotRightArmCubeStackEnvCfg):

    def __post_init__(self):

        super().__post_init__()



        use_relative_mode_env = os.getenv("USE_RELATIVE_MODE", "True")
        self.use_relative_mode = use_relative_mode_env.lower() in ["true", "1", "t"]


        self.actions.arm_action = RMPFlowActionCfg(
            asset_name="robot",
            joint_names=["right_arm_joint.*"],
            body_name="right_suction_cup_tcp_link",
            controller=GALBOT_RIGHT_ARM_RMPFLOW_CFG,
            scale=1.0,
            body_offset=RMPFlowActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
            articulation_prim_expr="/World/envs/env_.*/Robot",
            use_relative_mode=self.use_relative_mode,
        )

        self.sim.dt = 1 / 120
        self.sim.render_interval = 6

        self.decimation = 6
        self.episode_length_s = 30.0


        self.sim.physx.enable_ccd = True

        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
                "spacemouse": Se3SpaceMouseCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        Se3RelRetargeterCfg(
                            bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT,
                            zero_out_xy_rotation=True,
                            use_wrist_rotation=False,
                            use_wrist_position=True,
                            delta_pos_scale_factor=10.0,
                            delta_rot_scale_factor=10.0,
                            sim_device=self.sim.device,
                        ),
                        GripperRetargeterCfg(
                            bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT, sim_device=self.sim.device
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
            }
        )





@configclass
class RmpFlowGalbotLeftArmCubeStackVisuomotorEnvCfg(RmpFlowGalbotLeftArmCubeStackEnvCfg):

    def __post_init__(self):

        super().__post_init__()


        self.scene.right_wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_arm_camera_sim_view_frame/right_camera",
            update_period=0.0333,
            height=256,
            width=256,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
            offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
        )

        self.scene.left_wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/left_arm_camera_sim_view_frame/left_camera",
            update_period=0.0333,
            height=256,
            width=256,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
            offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
        )


        self.scene.ego_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/head_camera_sim_view_frame/head_camera",
            update_period=0.0333,
            height=256,
            width=256,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
            offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
        )


        self.scene.front_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/front_camera",
            update_period=0.0333,
            height=256,
            width=256,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
            offset=CameraCfg.OffsetCfg(pos=(1.0, 0.0, 0.6), rot=(-0.3799, 0.5963, 0.5963, -0.3799), convention="ros"),
        )

        marker_right_camera_cfg = FRAME_MARKER_CFG.copy()
        marker_right_camera_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_right_camera_cfg.prim_path = "/Visuals/FrameTransformerRightCamera"

        self.scene.right_arm_camera_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            visualizer_cfg=marker_right_camera_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/right_arm_camera_sim_view_frame",
                    name="right_camera",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.0],
                        rot=(0.5, -0.5, 0.5, -0.5),
                    ),
                ),
            ],
        )

        marker_left_camera_cfg = FRAME_MARKER_CFG.copy()
        marker_left_camera_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_left_camera_cfg.prim_path = "/Visuals/FrameTransformerLeftCamera"

        self.scene.left_arm_camera_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            visualizer_cfg=marker_left_camera_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/left_arm_camera_sim_view_frame",
                    name="left_camera",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.0],
                        rot=(0.5, -0.5, 0.5, -0.5),
                    ),
                ),
            ],
        )


        self.rerender_on_reset = True
        self.sim.render.antialiasing_mode = "OFF"


        self.image_obs_list = ["ego_cam", "left_wrist_cam", "right_wrist_cam"]







@configclass
class GalbotLeftArmJointPositionCubeStackVisuomotorEnvCfg_PLAY(RmpFlowGalbotLeftArmCubeStackVisuomotorEnvCfg):
    def __post_init__(self):

        super().__post_init__()

        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["left_arm_joint.*"], scale=1.0, use_default_offset=False
        )

        self.actions.gripper_action = mdp.AbsBinaryJointPositionActionCfg(
            asset_name="robot",
            threshold=0.030,
            joint_names=["left_gripper_.*_joint"],
            open_command_expr={"left_gripper_.*_joint": 0.035},
            close_command_expr={"left_gripper_.*_joint": 0.023},


        )





@configclass
class GalbotLeftArmRmpFlowCubeStackVisuomotorEnvCfg_PLAY(RmpFlowGalbotLeftArmCubeStackVisuomotorEnvCfg):
    def __post_init__(self):

        super().__post_init__()


        self.actions.gripper_action = mdp.AbsBinaryJointPositionActionCfg(
            asset_name="robot",
            threshold=0.030,
            joint_names=["left_gripper_.*_joint"],
            open_command_expr={"left_gripper_.*_joint": 0.035},
            close_command_expr={"left_gripper_.*_joint": 0.023},
        )
