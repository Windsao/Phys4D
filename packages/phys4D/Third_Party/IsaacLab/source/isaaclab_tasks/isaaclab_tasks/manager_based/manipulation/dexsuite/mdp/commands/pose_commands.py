





"""Sub-module containing command generators for pose tracking."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import pose_commands_cfg as dex_cmd_cfgs


class ObjectUniformPoseCommand(CommandTerm):
    """Uniform pose command generator for an object (in the robot base frame).

    This command term samples target object poses by:
      • Drawing (x, y, z) uniformly within configured Cartesian bounds, and
      • Drawing roll-pitch-yaw uniformly within configured ranges, then converting
        to a quaternion (w, x, y, z). Optionally makes quaternions unique by enforcing
        a positive real part.

    Frames:
        Targets are defined in the robot's *base frame*. For metrics/visualization,
        targets are transformed into the *world frame* using the robot root pose.

    Outputs:
        The command buffer has shape (num_envs, 7): `(x, y, z, qw, qx, qy, qz)`.

    Metrics:
        `position_error` and `orientation_error` are computed between the commanded
        world-frame pose and the object's current world-frame pose.

    Config:
        `cfg` must provide the sampling ranges, whether to enforce quaternion uniqueness,
        and optional visualization settings.
    """

    cfg: dex_cmd_cfgs.ObjectUniformPoseCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: dex_cmd_cfgs.ObjectUniformPoseCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """

        super().__init__(cfg, env)


        self.robot: Articulation = env.scene[cfg.asset_name]
        self.object: RigidObject = env.scene[cfg.object_name]
        self.success_vis_asset: RigidObject = env.scene[cfg.success_vis_asset_name]



        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0
        self.pose_command_w = torch.zeros_like(self.pose_command_b)

        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)

        self.success_visualizer = VisualizationMarkers(self.cfg.success_visualizer_cfg)
        self.success_visualizer.set_visibility(True)

    def __str__(self) -> str:
        msg = "UniformPoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired pose command. Shape is (num_envs, 7).

        The first three elements correspond to the position, followed by the quaternion orientation in (w, x, y, z).
        """
        return self.pose_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):

        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )

        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.object.data.root_state_w[:, :3],
            self.object.data.root_state_w[:, 3:7],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

        success_id = self.metrics["position_error"] < 0.05
        if not self.cfg.position_only:
            success_id &= self.metrics["orientation_error"] < 0.5
        self.success_visualizer.visualize(self.success_vis_asset.data.root_pos_w, marker_indices=success_id.int())

    def _resample_command(self, env_ids: Sequence[int]):


        r = torch.empty(len(env_ids), device=self.device)
        self.pose_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
        self.pose_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)
        self.pose_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.pos_z)

        euler_angles = torch.zeros_like(self.pose_command_b[env_ids, :3])
        euler_angles[:, 0].uniform_(*self.cfg.ranges.roll)
        euler_angles[:, 1].uniform_(*self.cfg.ranges.pitch)
        euler_angles[:, 2].uniform_(*self.cfg.ranges.yaw)
        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])

        self.pose_command_b[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):

        if debug_vis:
            if not hasattr(self, "goal_visualizer"):

                self.goal_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)

                self.curr_visualizer = VisualizationMarkers(self.cfg.curr_pose_visualizer_cfg)

            self.goal_visualizer.set_visibility(True)
            self.curr_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_visualizer"):
                self.goal_visualizer.set_visibility(False)
                self.curr_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):


        if not self.robot.is_initialized:
            return

        if not self.cfg.position_only:

            self.goal_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])

            self.curr_visualizer.visualize(self.object.data.root_pos_w, self.object.data.root_quat_w)
        else:
            distance = torch.norm(self.pose_command_w[:, :3] - self.object.data.root_pos_w[:, :3], dim=1)
            success_id = (distance < 0.05).int()


            self.goal_visualizer.visualize(self.pose_command_w[:, :3], marker_indices=success_id + 1)

            self.curr_visualizer.visualize(self.object.data.root_pos_w, marker_indices=success_id + 1)
