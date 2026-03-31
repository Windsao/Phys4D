





from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectMARLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate

from .shadow_hand_over_env_cfg import ShadowHandOverEnvCfg


class ShadowHandOverEnv(DirectMARLEnv):
    cfg: ShadowHandOverEnvCfg

    def __init__(self, cfg: ShadowHandOverEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.num_hand_dofs = self.right_hand.num_joints


        self.right_hand_dof_targets = torch.zeros(
            (self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device
        )
        self.right_hand_prev_targets = torch.zeros(
            (self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device
        )
        self.right_hand_curr_targets = torch.zeros(
            (self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device
        )
        self.left_hand_dof_targets = torch.zeros(
            (self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device
        )
        self.left_hand_prev_targets = torch.zeros(
            (self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device
        )
        self.left_hand_curr_targets = torch.zeros(
            (self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device
        )


        self.actuated_dof_indices = list()
        for joint_name in cfg.actuated_joint_names:
            self.actuated_dof_indices.append(self.right_hand.joint_names.index(joint_name))
        self.actuated_dof_indices.sort()


        self.finger_bodies = list()
        for body_name in self.cfg.fingertip_body_names:
            self.finger_bodies.append(self.right_hand.body_names.index(body_name))
        self.finger_bodies.sort()
        self.num_fingertips = len(self.finger_bodies)


        joint_pos_limits = self.right_hand.root_physx_view.get_dof_limits().to(self.device)
        self.hand_dof_lower_limits = joint_pos_limits[..., 0]
        self.hand_dof_upper_limits = joint_pos_limits[..., 1]


        self.in_hand_pos = self.object.data.default_root_state[:, 0:3].clone()
        self.in_hand_pos[:, 2] -= 0.04

        self.goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_rot[:, 0] = 1.0
        self.goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.goal_pos[:, :] = torch.tensor([0.0, -0.64, 0.54], device=self.device)

        self.goal_markers = VisualizationMarkers(self.cfg.goal_object_cfg)


        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

    def _setup_scene(self):

        self.right_hand = Articulation(self.cfg.right_robot_cfg)
        self.left_hand = Articulation(self.cfg.left_robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        self.scene.clone_environments(copy_from_source=False)

        self.scene.articulations["right_robot"] = self.right_hand
        self.scene.articulations["left_robot"] = self.left_hand
        self.scene.rigid_objects["object"] = self.object

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        self.actions = actions

    def _apply_action(self) -> None:

        self.right_hand_curr_targets[:, self.actuated_dof_indices] = scale(
            self.actions["right_hand"],
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )
        self.right_hand_curr_targets[:, self.actuated_dof_indices] = (
            self.cfg.act_moving_average * self.right_hand_curr_targets[:, self.actuated_dof_indices]
            + (1.0 - self.cfg.act_moving_average) * self.right_hand_prev_targets[:, self.actuated_dof_indices]
        )
        self.right_hand_curr_targets[:, self.actuated_dof_indices] = saturate(
            self.right_hand_curr_targets[:, self.actuated_dof_indices],
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )


        self.left_hand_curr_targets[:, self.actuated_dof_indices] = scale(
            self.actions["left_hand"],
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )
        self.left_hand_curr_targets[:, self.actuated_dof_indices] = (
            self.cfg.act_moving_average * self.left_hand_curr_targets[:, self.actuated_dof_indices]
            + (1.0 - self.cfg.act_moving_average) * self.left_hand_prev_targets[:, self.actuated_dof_indices]
        )
        self.left_hand_curr_targets[:, self.actuated_dof_indices] = saturate(
            self.left_hand_curr_targets[:, self.actuated_dof_indices],
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )


        self.right_hand_prev_targets[:, self.actuated_dof_indices] = self.right_hand_curr_targets[
            :, self.actuated_dof_indices
        ]
        self.left_hand_prev_targets[:, self.actuated_dof_indices] = self.left_hand_curr_targets[
            :, self.actuated_dof_indices
        ]


        self.right_hand.set_joint_position_target(
            self.right_hand_curr_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
        )
        self.left_hand.set_joint_position_target(
            self.left_hand_curr_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        observations = {
            "right_hand": torch.cat(
                (


                    unscale(self.right_hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),

                    self.cfg.vel_obs_scale * self.right_hand_dof_vel,

                    self.right_fingertip_pos.view(self.num_envs, self.num_fingertips * 3),

                    self.right_fingertip_rot.view(self.num_envs, self.num_fingertips * 4),

                    self.right_fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),

                    self.actions["right_hand"],


                    self.object_pos,

                    self.object_rot,

                    self.object_linvel,

                    self.cfg.vel_obs_scale * self.object_angvel,


                    self.goal_pos,

                    self.goal_rot,

                    quat_mul(self.object_rot, quat_conjugate(self.goal_rot)),
                ),
                dim=-1,
            ),
            "left_hand": torch.cat(
                (


                    unscale(self.left_hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),

                    self.cfg.vel_obs_scale * self.left_hand_dof_vel,

                    self.left_fingertip_pos.view(self.num_envs, self.num_fingertips * 3),

                    self.left_fingertip_rot.view(self.num_envs, self.num_fingertips * 4),

                    self.left_fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),

                    self.actions["left_hand"],


                    self.object_pos,

                    self.object_rot,

                    self.object_linvel,

                    self.cfg.vel_obs_scale * self.object_angvel,


                    self.goal_pos,

                    self.goal_rot,

                    quat_mul(self.object_rot, quat_conjugate(self.goal_rot)),
                ),
                dim=-1,
            ),
        }
        return observations

    def _get_states(self) -> torch.Tensor:
        states = torch.cat(
            (


                unscale(self.right_hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),

                self.cfg.vel_obs_scale * self.right_hand_dof_vel,

                self.right_fingertip_pos.view(self.num_envs, self.num_fingertips * 3),

                self.right_fingertip_rot.view(self.num_envs, self.num_fingertips * 4),

                self.right_fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),

                self.actions["right_hand"],


                unscale(self.left_hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),

                self.cfg.vel_obs_scale * self.left_hand_dof_vel,

                self.left_fingertip_pos.view(self.num_envs, self.num_fingertips * 3),

                self.left_fingertip_rot.view(self.num_envs, self.num_fingertips * 4),

                self.left_fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),

                self.actions["left_hand"],


                self.object_pos,

                self.object_rot,

                self.object_linvel,

                self.cfg.vel_obs_scale * self.object_angvel,


                self.goal_pos,

                self.goal_rot,

                quat_mul(self.object_rot, quat_conjugate(self.goal_rot)),
            ),
            dim=-1,
        )
        return states

    def _get_rewards(self) -> dict[str, torch.Tensor]:

        goal_dist = torch.norm(self.object_pos - self.goal_pos, p=2, dim=-1)
        rew_dist = 2 * torch.exp(-self.cfg.dist_reward_scale * goal_dist)


        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["dist_reward"] = rew_dist.mean()
        self.extras["log"]["dist_goal"] = goal_dist.mean()

        return {"right_hand": rew_dist, "left_hand": rew_dist}

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        self._compute_intermediate_values()


        out_of_reach = self.object_pos[:, 2] <= self.cfg.fall_dist

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        terminated = {agent: out_of_reach for agent in self.cfg.possible_agents}
        time_outs = {agent: time_out for agent in self.cfg.possible_agents}
        return terminated, time_outs

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = self.right_hand._ALL_INDICES

        super()._reset_idx(env_ids)


        self._reset_target_pose(env_ids)


        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)

        object_default_state[:, 0:3] = (
            object_default_state[:, 0:3] + self.cfg.reset_position_noise * pos_noise + self.scene.env_origins[env_ids]
        )

        rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)
        object_default_state[:, 3:7] = randomize_rotation(
            rot_noise[:, 0], rot_noise[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        )

        object_default_state[:, 7:] = torch.zeros_like(self.object.data.default_root_state[env_ids, 7:])
        self.object.write_root_pose_to_sim(object_default_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(object_default_state[:, 7:], env_ids)


        delta_max = self.hand_dof_upper_limits[env_ids] - self.right_hand.data.default_joint_pos[env_ids]
        delta_min = self.hand_dof_lower_limits[env_ids] - self.right_hand.data.default_joint_pos[env_ids]

        dof_pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * dof_pos_noise
        dof_pos = self.right_hand.data.default_joint_pos[env_ids] + self.cfg.reset_dof_pos_noise * rand_delta

        dof_vel_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        dof_vel = self.right_hand.data.default_joint_vel[env_ids] + self.cfg.reset_dof_vel_noise * dof_vel_noise

        self.right_hand_prev_targets[env_ids] = dof_pos
        self.right_hand_curr_targets[env_ids] = dof_pos
        self.right_hand_dof_targets[env_ids] = dof_pos

        self.right_hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.right_hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)


        delta_max = self.hand_dof_upper_limits[env_ids] - self.left_hand.data.default_joint_pos[env_ids]
        delta_min = self.hand_dof_lower_limits[env_ids] - self.left_hand.data.default_joint_pos[env_ids]

        dof_pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * dof_pos_noise
        dof_pos = self.left_hand.data.default_joint_pos[env_ids] + self.cfg.reset_dof_pos_noise * rand_delta

        dof_vel_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        dof_vel = self.left_hand.data.default_joint_vel[env_ids] + self.cfg.reset_dof_vel_noise * dof_vel_noise

        self.left_hand_prev_targets[env_ids] = dof_pos
        self.left_hand_curr_targets[env_ids] = dof_pos
        self.left_hand_dof_targets[env_ids] = dof_pos

        self.left_hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.left_hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

        self._compute_intermediate_values()

    def _reset_target_pose(self, env_ids):

        rand_floats = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)
        new_rot = randomize_rotation(
            rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        )


        self.goal_rot[env_ids] = new_rot
        goal_pos = self.goal_pos + self.scene.env_origins
        self.goal_markers.visualize(goal_pos, self.goal_rot)

    def _compute_intermediate_values(self):

        self.right_fingertip_pos = self.right_hand.data.body_pos_w[:, self.finger_bodies]
        self.right_fingertip_rot = self.right_hand.data.body_quat_w[:, self.finger_bodies]
        self.right_fingertip_pos -= self.scene.env_origins.repeat((1, self.num_fingertips)).reshape(
            self.num_envs, self.num_fingertips, 3
        )
        self.right_fingertip_velocities = self.right_hand.data.body_vel_w[:, self.finger_bodies]

        self.right_hand_dof_pos = self.right_hand.data.joint_pos
        self.right_hand_dof_vel = self.right_hand.data.joint_vel


        self.left_fingertip_pos = self.left_hand.data.body_pos_w[:, self.finger_bodies]
        self.left_fingertip_rot = self.left_hand.data.body_quat_w[:, self.finger_bodies]
        self.left_fingertip_pos -= self.scene.env_origins.repeat((1, self.num_fingertips)).reshape(
            self.num_envs, self.num_fingertips, 3
        )
        self.left_fingertip_velocities = self.left_hand.data.body_vel_w[:, self.finger_bodies]

        self.left_hand_dof_pos = self.left_hand.data.joint_pos
        self.left_hand_dof_vel = self.left_hand.data.joint_vel


        self.object_pos = self.object.data.root_pos_w - self.scene.env_origins
        self.object_rot = self.object.data.root_quat_w
        self.object_velocities = self.object.data.root_vel_w
        self.object_linvel = self.object.data.root_lin_vel_w
        self.object_angvel = self.object.data.root_ang_vel_w


@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )
