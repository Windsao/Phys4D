




from __future__ import annotations

import math
import torch

import pytorch_kinematics as pk

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    euler_xyz_from_quat,
    sample_uniform,
    subtract_frame_transforms,
    wrap_to_pi,
)


from tacex_assets import TACEX_ASSETS_DATA_DIR

from .reset_with_IK_solver import BallRollingIKResetEnvCfg





from isaaclab.markers import CUBOID_MARKER_CFG


@configclass
class BallRollingWithoutReachingEnvCfg(BallRollingIKResetEnvCfg):

    reaching_penalty = {"weight": -0.2}
    reaching_reward_tanh = {"std": 0.2, "weight": 0.4}
    at_obj_reward = {"weight": 1, "minimal_distance": 0.01}
    tracking_reward = {"weight": 0.3, "w": 1, "v": 1, "alpha": 1e-5, "minimal_distance": 0.01}

    success_reward = {
        "weight": 10,
        "threshold": 0.005,
    }
    height_penalty = {
        "weight": -0.1,
        "min_height": 0.008,
    }
    orient_penalty = {"weight": -0.1}


    action_rate_penalty_scale = [
        -1e-4,
        -1e-2,
    ]
    joint_vel_penalty_scale = [-1e-4, -1e-2]


    curriculum_steps = [1e6]
    obj_pos_randomization_range = [[-0.1, 0.1], [-0.25, 0.25]]


    episode_length_s = 8.3333
    action_space = 5
    observation_space = (
        14
    )
    state_space = 0

    ball_radius = 0.01
    x_bounds = (0.2, 0.75)
    y_bounds = (-0.375, 0.375)
    too_far_away_threshold = 0.35


class BallRollingWithoutReachingEnv(DirectRLEnv):
    """RL env in which the robot has to push/roll a ball to a goal position.

    This base env uses (absolute) joint positions.
    Absolute joint pos and vel are used for the observations.
    """










    cfg: BallRollingWithoutReachingEnvCfg

    def __init__(self, cfg: BallRollingWithoutReachingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation


        self.curriculum_phase_id = 0

        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)


        self.init_goal_distances = torch.zeros(self.num_envs, device=self.device)

        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        self._desired_pos_w[:, 2] = 0.00125


        with open(self.cfg.ik_solver_cfg["urdf_path"], mode="rb") as urdf_file:
            ik_chain = pk.build_chain_from_urdf(urdf_file.read())


        ik_chain = pk.SerialChain(ik_chain, self.cfg.ik_solver_cfg["ee_link_name"])

        ik_chain = ik_chain.to(dtype=torch.float32, device=self.device)


        ik_chain_lim = torch.tensor(ik_chain.get_joint_limits(), device=self.device)



        self.ik_solver = pk.PseudoInverseIK(
            ik_chain,
            max_iterations=self.cfg.ik_solver_cfg["max_iterations"],
            num_retries=self.cfg.ik_solver_cfg["num_retries"],
            joint_limits=ik_chain_lim.T,
            early_stopping_any_converged=True,
            early_stopping_no_improvement=None,
            debug=False,
            lr=self.cfg.ik_solver_cfg["learning_rate"],
        )
        self.des_reset_ee_pos = torch.zeros((self.num_envs, 3), device=self.device)

        self.des_reset_ee_rot = (
            torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], device=self.device)
            .unsqueeze(0)
            .repeat(self.num_envs, 1, 1)
        )



        self._ik_controller = DifferentialIKController(
            cfg=self.cfg.ik_controller_cfg, num_envs=self.num_envs, device=self.device
        )

        body_ids, body_names = self._robot.find_bodies("panda_hand")

        self._body_idx = body_ids[0]
        self._body_name = body_names[0]



        self._jacobi_body_idx = self._body_idx - 1



        self._offset_pos = torch.tensor([0.0, 0.0, 0.131], device=self.device).repeat(self.num_envs, 1)
        self._offset_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)



        self.processed_actions = torch.zeros((self.num_envs, self._ik_controller.action_dim), device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)


        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.object = RigidObject(self.cfg.ball)
        self.scene.rigid_objects["object"] = self.object


        self.scene.clone_environments(copy_from_source=False)

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        ee_frame_cfg = FrameTransformerCfg(
            prim_path="/World/envs/env_.*/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="/World/envs/env_.*/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.131),
                    ),
                ),
            ],
        )


        self._ee_frame = FrameTransformer(ee_frame_cfg)
        self.scene.sensors["ee_frame"] = self._ee_frame





        ground = AssetBaseCfg(
            prim_path="/World/defaultGroundPlane",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, 0)),
            spawn=sim_utils.GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
            ),
        )
        ground.spawn.func(
            ground.prim_path, ground.spawn, translation=ground.init_state.pos, orientation=ground.init_state.rot
        )


        plate = RigidObjectCfg(
            prim_path="/World/envs/env_.*/plate",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0, 0)),
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{TACEX_ASSETS_DATA_DIR}/Props/plate.usd",
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    kinematic_enabled=True,
                ),
            ),
        )
        plate.spawn.func(
            plate.prim_path, plate.spawn, translation=plate.init_state.pos, orientation=ground.init_state.rot
        )


        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)



    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions[:] = self.actions
        self.actions[:] = actions.clamp(-1, 1)

        self.processed_actions[:, :5] = self.actions

        self.processed_actions[:, 5] = 0


        self.ee_pos_curr_b, self.ee_quat_curr_b = self._compute_frame_pose()

        self._ik_controller.set_command(self.processed_actions, self.ee_pos_curr_b, self.ee_quat_curr_b)

    def _apply_action(self):

        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()
        joint_pos = self._robot.data.joint_pos[:, :]

        if self.ee_pos_curr_b.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            joint_pos_des = self._ik_controller.compute(ee_pos_curr_b, ee_quat_curr_b, jacobian, joint_pos)
        else:
            joint_pos_des = joint_pos.clone()

        self._robot.set_joint_position_target(joint_pos_des)




    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        obj_pos = self.object.data.root_link_pos_w - self.scene.env_origins
        out_of_bounds_x = (obj_pos[:, 0] < self.cfg.x_bounds[0]) | (obj_pos[:, 0] > self.cfg.x_bounds[1])
        out_of_bounds_y = (obj_pos[:, 1] < self.cfg.y_bounds[0]) | (obj_pos[:, 1] > self.cfg.y_bounds[1])

        obj_goal_distance = torch.norm(
            self._desired_pos_w[:, :2] - self.scene.env_origins[:, :2] - obj_pos[:, :2], dim=1
        )
        obj_too_far_away = obj_goal_distance > 1

        ee_frame_pos = (
            self._ee_frame.data.target_pos_w[..., 0, :] - self.scene.env_origins
        )
        ee_too_far_away = torch.norm(obj_pos - ee_frame_pos, dim=1) > self.cfg.too_far_away_threshold

        reset_cond = out_of_bounds_x | out_of_bounds_y | obj_too_far_away | ee_too_far_away

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return reset_cond, time_out


    def _get_rewards(self) -> torch.Tensor:

        obj_pos = self.object.data.root_link_state_w[:, :3]

        obj_pos[:, 2] += 0.005
        ee_frame_pos = self._ee_frame.data.target_pos_w[
            ..., 0, :
        ]


        object_ee_distance = torch.norm(obj_pos - ee_frame_pos, dim=1)
        reaching_penalty = self.cfg.reaching_penalty["weight"] * torch.square(object_ee_distance)

        object_ee_distance_tanh = 1 - torch.tanh(object_ee_distance / self.cfg.reaching_reward_tanh["std"])

        at_obj_reward = (object_ee_distance < self.cfg.at_obj_reward["minimal_distance"]) * self.cfg.at_obj_reward[
            "weight"
        ]


        obj_goal_distance = torch.norm(self._desired_pos_w[:, :2] - self.object.data.root_link_state_w[:, :2], dim=1)
        tracking_goal = -(
            self.cfg.tracking_reward["w"] * obj_goal_distance
            + self.cfg.tracking_reward["v"] * torch.log(obj_goal_distance + self.cfg.tracking_reward["alpha"])
        )

        tracking_goal = (object_ee_distance < self.cfg.tracking_reward["minimal_distance"]) * tracking_goal
        tracking_goal *= self.cfg.tracking_reward["weight"]







        height_penalty = (ee_frame_pos[:, 2] < self.cfg.height_penalty["min_height"]) * self.cfg.height_penalty[
            "weight"
        ]


        ee_frame_orient = euler_xyz_from_quat(self._ee_frame.data.target_quat_source[..., 0, :])
        x = wrap_to_pi(
            ee_frame_orient[0] - math.pi
        )
        y = wrap_to_pi(ee_frame_orient[1])
        orient_penalty = ((torch.abs(x) > math.pi / 8) | (torch.abs(y) > math.pi / 8)) * self.cfg.orient_penalty[
            "weight"
        ]

        success_reward = (obj_goal_distance < self.cfg.success_reward["threshold"]) * self.cfg.success_reward["weight"]


        action_rate_penalty = torch.sum(torch.square(self.actions - self.prev_actions), dim=1)

        joint_vel_penalty = torch.sum(torch.square(self._robot.data.joint_vel[:, :]), dim=1)



        if self.common_step_counter > self.cfg.curriculum_steps[self.curriculum_phase_id - 1]:
            self.curriculum_phase_id = 1

        rewards = (
            +reaching_penalty
            + self.cfg.reaching_reward_tanh["weight"] * object_ee_distance_tanh
            + at_obj_reward
            + tracking_goal

            + success_reward
            + orient_penalty
            + height_penalty
            + self.cfg.action_rate_penalty_scale[self.curriculum_phase_id] * action_rate_penalty
            + self.cfg.joint_vel_penalty_scale[self.curriculum_phase_id] * joint_vel_penalty
        )

        self.extras["log"] = {
            "reaching_penalty": reaching_penalty.float().mean(),
            "reaching_reward_tanh": (self.cfg.reaching_reward_tanh["weight"] * object_ee_distance_tanh).mean(),
            "at_obj_reward": at_obj_reward.float().mean(),
            "tracking_goal": tracking_goal.float().mean(),

            "success_reward": success_reward.float().mean(),

            "orientation_penalty": orient_penalty.float().mean(),
            "height_penalty": height_penalty.mean(),
            "action_rate_penalty": (
                (self.cfg.action_rate_penalty_scale[self.curriculum_phase_id] * action_rate_penalty).mean()
            ),
            "joint_vel_penalty": (
                (self.cfg.joint_vel_penalty_scale[self.curriculum_phase_id] * joint_vel_penalty).mean()
            ),

            "Metric/num_ee_at_obj": torch.sum(object_ee_distance < self.cfg.tracking_reward["minimal_distance"]),
            "Metric/ee_obj_error": object_ee_distance.mean(),
            "Metric/obj_goal_error": obj_goal_distance.mean(),
        }
        return rewards

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        obj_pos = self.object.data.default_root_state[env_ids]
        obj_pos[:, :3] += self.scene.env_origins[env_ids]
        obj_pos[:, :2] += sample_uniform(
            self.cfg.obj_pos_randomization_range[self.curriculum_phase_id][0],
            self.cfg.obj_pos_randomization_range[self.curriculum_phase_id][1],
            (len(env_ids), 2),
            self.device,
        )
        self.object.write_root_state_to_sim(obj_pos, env_ids=env_ids)



        self.des_reset_ee_pos[env_ids, :] = obj_pos[:, :3].clone() - self.scene.env_origins[env_ids]
        self.des_reset_ee_pos[env_ids, 2] += 2 * self.cfg.ball_radius + 0.01


        goal_poses = pk.Transform3d(
            pos=self.des_reset_ee_pos[env_ids], rot=self.des_reset_ee_rot[env_ids], device=self.device
        )

        sol = self.ik_solver.solve(goal_poses)







        indices = torch.argmin(sol.err_pos, dim=1)
        best_sol_currently = sol.solutions[torch.arange(indices.size(0)), indices]

        joint_pos = torch.clamp(best_sol_currently, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


        self._desired_pos_w[env_ids, :2] = (
            self.object.data.default_root_state[env_ids][:, :2] + self.scene.env_origins[env_ids][:, :2]
        )
        self._desired_pos_w[env_ids, :2] += sample_uniform(
            self.cfg.obj_pos_randomization_range[self.curriculum_phase_id][0],
            self.cfg.obj_pos_randomization_range[self.curriculum_phase_id][1],
            (len(env_ids), 2),
            self.device,
        )


        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self._ik_controller.reset(env_ids)


    def _get_observations(self) -> dict:
        """The position of the object in the robot's root frame."""

        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()
        ee_frame_orient = euler_xyz_from_quat(ee_quat_curr_b)
        x = wrap_to_pi(ee_frame_orient[0]).unsqueeze(1)
        y = wrap_to_pi(ee_frame_orient[1]).unsqueeze(1)


        object_pos_w = self.object.data.root_link_pos_w[:, :3]
        object_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_link_state_w[:, :3], self._robot.data.root_link_state_w[:, 3:7], object_pos_w
        )

        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_link_state_w[:, :3], self._robot.data.root_link_state_w[:, 3:7], self._desired_pos_w
        )

        obs = torch.cat(
            (
                ee_pos_curr_b,
                x,
                y,
                object_pos_b[:, :2],
                desired_pos_b[:, :2],
                self.actions,
            ),
            dim=-1,
        )


        return {"policy": obs}

    """
    Helper Functions for IK control (from task_space_actions.py of IsaacLab).
    """

    @property
    def jacobian_w(self) -> torch.Tensor:
        return self._robot.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, :]

    @property
    def jacobian_b(self) -> torch.Tensor:
        jacobian = self.jacobian_w
        base_rot = self._robot.data.root_link_quat_w
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot))
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
        return jacobian

    def _compute_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the pose of the target frame in the root frame.

        Returns:
            A tuple of the body's position and orientation in the root frame.
        """

        ee_pos_w = self._robot.data.body_link_pos_w[:, self._body_idx]
        ee_quat_w = self._robot.data.body_link_quat_w[:, self._body_idx]
        root_pos_w = self._robot.data.root_link_pos_w
        root_quat_w = self._robot.data.root_link_quat_w

        ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)


        ee_pose_b, ee_quat_b = math_utils.combine_frame_transforms(
            ee_pose_b, ee_quat_b, self._offset_pos, self._offset_rot
        )

        return ee_pose_b, ee_quat_b

    def _compute_frame_jacobian(self):
        """Computes the geometric Jacobian of the target frame in the root frame.

        This function accounts for the target frame offset and applies the necessary transformations to obtain
        the right Jacobian from the parent body Jacobian.
        """

        jacobian = self.jacobian_b








        jacobian[:, 0:3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(self._offset_pos), jacobian[:, 3:, :])


        jacobian[:, 3:, :] = torch.bmm(math_utils.matrix_from_quat(self._offset_rot), jacobian[:, 3:, :])

        return jacobian



    def _set_debug_vis_impl(self, debug_vis: bool):

        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()

                marker_cfg.markers["cuboid"].size = (
                    2 * self.cfg.success_reward["threshold"],
                    2 * self.cfg.success_reward["threshold"],
                    0.01,
                )

                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)

            self.goal_pos_visualizer.set_visibility(True)









        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)



    def _debug_vis_callback(self, event):

        self.goal_pos_visualizer.visualize(self._desired_pos_w)






