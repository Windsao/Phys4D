




import tempfile

import isaaclab.controllers.utils as ControllerUtils
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg, XrCfg
from isaaclab.devices.openxr.retargeters.humanoid.fourier.gr1t2_retargeter import GR1T2RetargeterCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from .pickplace_gr1t2_env_cfg import ActionsCfg, EventCfg, ObjectTableSceneCfg, ObservationsCfg, TerminationsCfg


@configclass
class PickPlaceGR1T2WaistEnabledEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the GR1T2 environment."""


    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()

    terminations: TerminationsCfg = TerminationsCfg()
    events = EventCfg()


    commands = None
    rewards = None
    curriculum = None


    xr: XrCfg = XrCfg(
        anchor_pos=(0.0, 0.0, 0.0),
        anchor_rot=(1.0, 0.0, 0.0, 0.0),
    )


    NUM_OPENXR_HAND_JOINTS = 26


    temp_urdf_dir = tempfile.gettempdir()

    def __post_init__(self):
        """Post initialization."""

        self.decimation = 6
        self.episode_length_s = 20.0

        self.sim.dt = 1 / 120
        self.sim.render_interval = 2


        waist_joint_names = ["waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"]
        for joint_name in waist_joint_names:
            self.actions.upper_body_ik.pink_controlled_joint_names.append(joint_name)


        temp_urdf_output_path, temp_urdf_meshes_output_path = ControllerUtils.convert_usd_to_urdf(
            self.scene.robot.spawn.usd_path, self.temp_urdf_dir, force_conversion=True
        )


        self.actions.upper_body_ik.controller.urdf_path = temp_urdf_output_path
        self.actions.upper_body_ik.controller.mesh_path = temp_urdf_meshes_output_path

        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        GR1T2RetargeterCfg(
                            enable_visualization=True,

                            num_open_xr_hand_joints=2 * self.NUM_OPENXR_HAND_JOINTS,
                            sim_device=self.sim.device,
                            hand_joint_names=self.actions.upper_body_ik.hand_joint_names,
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
            }
        )
