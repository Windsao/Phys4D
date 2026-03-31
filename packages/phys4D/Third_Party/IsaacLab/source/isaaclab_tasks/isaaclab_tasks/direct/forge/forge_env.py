



import numpy as np
import torch

import isaacsim.core.utils.torch as torch_utils

from isaaclab.utils.math import axis_angle_from_quat

from isaaclab_tasks.direct.factory import factory_utils
from isaaclab_tasks.direct.factory.factory_env import FactoryEnv

from . import forge_utils
from .forge_env_cfg import ForgeEnvCfg


class ForgeEnv(FactoryEnv):
    cfg: ForgeEnvCfg

    def __init__(self, cfg: ForgeEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize additional randomization and logging tensors."""
        super().__init__(cfg, render_mode, **kwargs)


        self.success_pred_scale = 0.0
        self.first_pred_success_tx = {}
        for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
            self.first_pred_success_tx[thresh] = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)


        self.flip_quats = torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)


        self.force_sensor_body_idx = self._robot.body_names.index("force_sensor")
        self.force_sensor_smooth = torch.zeros((self.num_envs, 6), device=self.device)
        self.force_sensor_world_smooth = torch.zeros((self.num_envs, 6), device=self.device)


        self.default_gains = torch.tensor(self.cfg.ctrl.default_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.default_pos_threshold = torch.tensor(self.cfg.ctrl.pos_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.default_rot_threshold = torch.tensor(self.cfg.ctrl.rot_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.default_dead_zone = torch.tensor(self.cfg.ctrl.default_dead_zone, device=self.device).repeat(
            (self.num_envs, 1)
        )

        self.pos_threshold = self.default_pos_threshold.clone()
        self.rot_threshold = self.default_rot_threshold.clone()

    def _compute_intermediate_values(self, dt):
        """Add noise to observations for force sensing."""
        super()._compute_intermediate_values(dt)


        pos_noise_level, rot_noise_level_deg = self.cfg.obs_rand.fingertip_pos, self.cfg.obs_rand.fingertip_rot_deg
        fingertip_pos_noise = torch.randn((self.num_envs, 3), dtype=torch.float32, device=self.device)
        fingertip_pos_noise = fingertip_pos_noise @ torch.diag(
            torch.tensor([pos_noise_level, pos_noise_level, pos_noise_level], dtype=torch.float32, device=self.device)
        )
        self.noisy_fingertip_pos = self.fingertip_midpoint_pos + fingertip_pos_noise

        rot_noise_axis = torch.randn((self.num_envs, 3), dtype=torch.float32, device=self.device)
        rot_noise_axis /= torch.linalg.norm(rot_noise_axis, dim=1, keepdim=True)
        rot_noise_angle = torch.randn((self.num_envs,), dtype=torch.float32, device=self.device) * np.deg2rad(
            rot_noise_level_deg
        )
        self.noisy_fingertip_quat = torch_utils.quat_mul(
            self.fingertip_midpoint_quat, torch_utils.quat_from_angle_axis(rot_noise_angle, rot_noise_axis)
        )
        self.noisy_fingertip_quat[:, [0, 3]] = 0.0
        self.noisy_fingertip_quat = self.noisy_fingertip_quat * self.flip_quats.unsqueeze(-1)


        self.ee_linvel_fd = (self.noisy_fingertip_pos - self.prev_fingertip_pos) / dt
        self.prev_fingertip_pos = self.noisy_fingertip_pos.clone()


        rot_diff_quat = torch_utils.quat_mul(
            self.noisy_fingertip_quat, torch_utils.quat_conjugate(self.prev_fingertip_quat)
        )
        rot_diff_quat *= torch.sign(rot_diff_quat[:, 0]).unsqueeze(-1)
        rot_diff_aa = axis_angle_from_quat(rot_diff_quat)
        self.ee_angvel_fd = rot_diff_aa / dt
        self.ee_angvel_fd[:, 0:2] = 0.0
        self.prev_fingertip_quat = self.noisy_fingertip_quat.clone()


        self.force_sensor_world = self._robot.root_physx_view.get_link_incoming_joint_force()[
            :, self.force_sensor_body_idx
        ]

        alpha = self.cfg.ft_smoothing_factor
        self.force_sensor_world_smooth = alpha * self.force_sensor_world + (1 - alpha) * self.force_sensor_world_smooth

        self.force_sensor_smooth = torch.zeros_like(self.force_sensor_world)
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.force_sensor_smooth[:, :3], self.force_sensor_smooth[:, 3:6] = forge_utils.change_FT_frame(
            self.force_sensor_world_smooth[:, 0:3],
            self.force_sensor_world_smooth[:, 3:6],
            (identity_quat, torch.zeros((self.num_envs, 3), device=self.device)),
            (identity_quat, self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise),
        )


        force_noise = torch.randn((self.num_envs, 3), dtype=torch.float32, device=self.device)
        force_noise *= self.cfg.obs_rand.ft_force
        self.noisy_force = self.force_sensor_smooth[:, 0:3] + force_noise

    def _get_observations(self):
        """Add additional FORGE observations."""
        obs_dict, state_dict = self._get_factory_obs_state_dict()

        noisy_fixed_pos = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        prev_actions = self.actions.clone()
        prev_actions[:, 3:5] = 0.0

        obs_dict.update({
            "fingertip_pos": self.noisy_fingertip_pos,
            "fingertip_pos_rel_fixed": self.noisy_fingertip_pos - noisy_fixed_pos,
            "fingertip_quat": self.noisy_fingertip_quat,
            "force_threshold": self.contact_penalty_thresholds[:, None],
            "ft_force": self.noisy_force,
            "prev_actions": prev_actions,
        })

        state_dict.update({
            "ema_factor": self.ema_factor,
            "ft_force": self.force_sensor_smooth[:, 0:3],
            "force_threshold": self.contact_penalty_thresholds[:, None],
            "prev_actions": prev_actions,
        })

        obs_tensors = factory_utils.collapse_obs_dict(obs_dict, self.cfg.obs_order + ["prev_actions"])
        state_tensors = factory_utils.collapse_obs_dict(state_dict, self.cfg.state_order + ["prev_actions"])
        return {"policy": obs_tensors, "critic": state_tensors}

    def _apply_action(self):
        """FORGE actions are defined as targets relative to the fixed asset."""
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)


        pos_actions = self.actions[:, 0:3]
        pos_actions = pos_actions @ torch.diag(torch.tensor(self.cfg.ctrl.pos_action_bounds, device=self.device))

        rot_actions = self.actions[:, 3:6]
        rot_actions = rot_actions @ torch.diag(torch.tensor(self.cfg.ctrl.rot_action_bounds, device=self.device))



        fixed_pos_action_frame = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        ctrl_target_fingertip_preclipped_pos = fixed_pos_action_frame + pos_actions

        rot_actions[:, 0:2] = 0.0


        rot_actions[:, 2] = np.deg2rad(-180.0) + np.deg2rad(270.0) * (rot_actions[:, 2] + 1.0) / 2.0

        bolt_frame_quat = torch_utils.quat_from_euler_xyz(
            roll=rot_actions[:, 0], pitch=rot_actions[:, 1], yaw=rot_actions[:, 2]
        )

        rot_180_euler = torch.tensor([np.pi, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        quat_bolt_to_ee = torch_utils.quat_from_euler_xyz(
            roll=rot_180_euler[:, 0], pitch=rot_180_euler[:, 1], yaw=rot_180_euler[:, 2]
        )

        ctrl_target_fingertip_preclipped_quat = torch_utils.quat_mul(quat_bolt_to_ee, bolt_frame_quat)



        self.delta_pos = ctrl_target_fingertip_preclipped_pos - self.fingertip_midpoint_pos
        pos_error_clipped = torch.clip(self.delta_pos, -self.pos_threshold, self.pos_threshold)
        ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_error_clipped






        curr_roll, curr_pitch, curr_yaw = torch_utils.get_euler_xyz(self.fingertip_midpoint_quat)
        desired_roll, desired_pitch, desired_yaw = torch_utils.get_euler_xyz(ctrl_target_fingertip_preclipped_quat)
        desired_xyz = torch.stack([desired_roll, desired_pitch, desired_yaw], dim=1)



        curr_yaw = factory_utils.wrap_yaw(curr_yaw)
        desired_yaw = factory_utils.wrap_yaw(desired_yaw)


        self.delta_yaw = desired_yaw - curr_yaw
        clipped_yaw = torch.clip(self.delta_yaw, -self.rot_threshold[:, 2], self.rot_threshold[:, 2])
        desired_xyz[:, 2] = curr_yaw + clipped_yaw


        desired_roll = torch.where(desired_roll < 0.0, desired_roll + 2 * torch.pi, desired_roll)
        desired_pitch = torch.where(desired_pitch < 0.0, desired_pitch + 2 * torch.pi, desired_pitch)

        delta_roll = desired_roll - curr_roll
        clipped_roll = torch.clip(delta_roll, -self.rot_threshold[:, 0], self.rot_threshold[:, 0])
        desired_xyz[:, 0] = curr_roll + clipped_roll

        curr_pitch = torch.where(curr_pitch > torch.pi, curr_pitch - 2 * torch.pi, curr_pitch)
        desired_pitch = torch.where(desired_pitch > torch.pi, desired_pitch - 2 * torch.pi, desired_pitch)

        delta_pitch = desired_pitch - curr_pitch
        clipped_pitch = torch.clip(delta_pitch, -self.rot_threshold[:, 1], self.rot_threshold[:, 1])
        desired_xyz[:, 1] = curr_pitch + clipped_pitch

        ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            roll=desired_xyz[:, 0], pitch=desired_xyz[:, 1], yaw=desired_xyz[:, 2]
        )

        self.generate_ctrl_signals(
            ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
            ctrl_target_gripper_dof_pos=0.0,
        )

    def _get_rewards(self):
        """FORGE reward includes a contact penalty and success prediction error."""

        rew_buf = super()._get_rewards()

        rew_dict, rew_scales = {}, {}

        pos_error = torch.norm(self.delta_pos, p=2, dim=-1) / self.cfg.ctrl.pos_action_threshold[0]
        rot_error = torch.abs(self.delta_yaw) / self.cfg.ctrl.rot_action_threshold[0]

        contact_force = torch.norm(self.force_sensor_smooth[:, 0:3], p=2, dim=-1, keepdim=False)
        contact_penalty = torch.nn.functional.relu(contact_force - self.contact_penalty_thresholds)

        check_rot = self.cfg_task.name == "nut_thread"
        true_successes = self._get_curr_successes(
            success_threshold=self.cfg_task.success_threshold, check_rot=check_rot
        )
        policy_success_pred = (self.actions[:, 6] + 1) / 2
        success_pred_error = (true_successes.float() - policy_success_pred).abs()

        if true_successes.float().mean() >= self.cfg_task.delay_until_ratio:
            self.success_pred_scale = 1.0


        rew_dict = {
            "action_penalty_asset": pos_error + rot_error,
            "contact_penalty": contact_penalty,
            "success_pred_error": success_pred_error,
        }
        rew_scales = {
            "action_penalty_asset": -self.cfg_task.action_penalty_asset_scale,
            "contact_penalty": -self.cfg_task.contact_penalty_scale,
            "success_pred_error": -self.success_pred_scale,
        }
        for rew_name, rew in rew_dict.items():
            rew_buf += rew_dict[rew_name] * rew_scales[rew_name]

        self._log_forge_metrics(rew_dict, policy_success_pred)
        return rew_buf

    def _reset_idx(self, env_ids):
        """Perform additional randomizations."""
        super()._reset_idx(env_ids)


        fixed_pos_action_frame = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        pos_actions = self.fingertip_midpoint_pos - fixed_pos_action_frame
        pos_action_bounds = torch.tensor(self.cfg.ctrl.pos_action_bounds, device=self.device)
        pos_actions = pos_actions @ torch.diag(1.0 / pos_action_bounds)
        self.actions[:, 0:3] = self.prev_actions[:, 0:3] = pos_actions


        unrot_180_euler = torch.tensor([-np.pi, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        unrot_quat = torch_utils.quat_from_euler_xyz(
            roll=unrot_180_euler[:, 0], pitch=unrot_180_euler[:, 1], yaw=unrot_180_euler[:, 2]
        )

        fingertip_quat_rel_bolt = torch_utils.quat_mul(unrot_quat, self.fingertip_midpoint_quat)
        fingertip_yaw_bolt = torch_utils.get_euler_xyz(fingertip_quat_rel_bolt)[-1]
        fingertip_yaw_bolt = torch.where(
            fingertip_yaw_bolt > torch.pi / 2, fingertip_yaw_bolt - 2 * torch.pi, fingertip_yaw_bolt
        )
        fingertip_yaw_bolt = torch.where(
            fingertip_yaw_bolt < -torch.pi, fingertip_yaw_bolt + 2 * torch.pi, fingertip_yaw_bolt
        )

        yaw_action = (fingertip_yaw_bolt + np.deg2rad(180.0)) / np.deg2rad(270.0) * 2.0 - 1.0
        self.actions[:, 5] = self.prev_actions[:, 5] = yaw_action
        self.actions[:, 6] = self.prev_actions[:, 6] = -1.0


        ema_rand = torch.rand((self.num_envs, 1), dtype=torch.float32, device=self.device)
        ema_lower, ema_upper = self.cfg.ctrl.ema_factor_range
        self.ema_factor = ema_lower + ema_rand * (ema_upper - ema_lower)


        prop_gains = self.default_gains.clone()
        self.pos_threshold = self.default_pos_threshold.clone()
        self.rot_threshold = self.default_rot_threshold.clone()
        prop_gains = forge_utils.get_random_prop_gains(
            prop_gains, self.cfg.ctrl.task_prop_gains_noise_level, self.num_envs, self.device
        )
        self.pos_threshold = forge_utils.get_random_prop_gains(
            self.pos_threshold, self.cfg.ctrl.pos_threshold_noise_level, self.num_envs, self.device
        )
        self.rot_threshold = forge_utils.get_random_prop_gains(
            self.rot_threshold, self.cfg.ctrl.rot_threshold_noise_level, self.num_envs, self.device
        )
        self.task_prop_gains = prop_gains
        self.task_deriv_gains = factory_utils.get_deriv_gains(prop_gains)

        contact_rand = torch.rand((self.num_envs,), dtype=torch.float32, device=self.device)
        contact_lower, contact_upper = self.cfg.task.contact_penalty_threshold_range
        self.contact_penalty_thresholds = contact_lower + contact_rand * (contact_upper - contact_lower)

        self.dead_zone_thresholds = (
            torch.rand((self.num_envs, 6), dtype=torch.float32, device=self.device) * self.default_dead_zone
        )

        self.force_sensor_world_smooth[:, :] = 0.0

        self.flip_quats = torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)
        rand_flips = torch.rand(self.num_envs) > 0.5
        self.flip_quats[rand_flips] = -1.0

    def _reset_buffers(self, env_ids):
        """Reset additional logging metrics."""
        super()._reset_buffers(env_ids)

        for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
            self.first_pred_success_tx[thresh][env_ids] = 0

    def _log_forge_metrics(self, rew_dict, policy_success_pred):
        """Log metrics to evaluate success prediction performance."""
        for rew_name, rew in rew_dict.items():
            self.extras[f"logs_rew_{rew_name}"] = rew.mean()

        for thresh, first_success_tx in self.first_pred_success_tx.items():
            curr_predicted_success = policy_success_pred > thresh
            first_success_idxs = torch.logical_and(curr_predicted_success, first_success_tx == 0)

            first_success_tx[:] = torch.where(first_success_idxs, self.episode_length_buf, first_success_tx)


            if torch.any(self.reset_buf):

                delay_ids = torch.logical_and(self.ep_success_times != 0, first_success_tx != 0)
                delay_times = (first_success_tx[delay_ids] - self.ep_success_times[delay_ids]).sum() / delay_ids.sum()
                if delay_ids.sum().item() > 0:
                    self.extras[f"early_term_delay_all/{thresh}"] = delay_times

                correct_delay_ids = torch.logical_and(delay_ids, first_success_tx > self.ep_success_times)
                correct_delay_times = (
                    first_success_tx[correct_delay_ids] - self.ep_success_times[correct_delay_ids]
                ).sum() / correct_delay_ids.sum()
                if correct_delay_ids.sum().item() > 0:
                    self.extras[f"early_term_delay_correct/{thresh}"] = correct_delay_times.item()


                pred_success_idxs = first_success_tx != 0

                true_success_preds = torch.logical_and(
                    self.ep_success_times[pred_success_idxs] > 0,
                    self.ep_success_times[pred_success_idxs]
                    < first_success_tx[pred_success_idxs],
                )

                num_pred_success = pred_success_idxs.sum().item()
                et_prec = true_success_preds.sum() / num_pred_success
                if num_pred_success > 0:
                    self.extras[f"early_term_precision/{thresh}"] = et_prec

                true_success_idxs = self.ep_success_times > 0
                num_true_success = true_success_idxs.sum().item()
                et_recall = true_success_preds.sum() / num_true_success
                if num_true_success > 0:
                    self.extras[f"early_term_recall/{thresh}"] = et_recall
