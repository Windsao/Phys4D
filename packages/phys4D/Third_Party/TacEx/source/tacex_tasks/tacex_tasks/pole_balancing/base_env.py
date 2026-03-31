




from __future__ import annotations

import math
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    euler_xyz_from_quat,
    sample_uniform,
    subtract_frame_transforms,
    wrap_to_pi,
)
from isaaclab.utils.noise import NoiseModelCfg, UniformNoiseCfg

from tacex import GelSightSensor


from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets.robots.franka.franka_gsmini_single_rigid import (
    FRANKA_PANDA_ARM_SINGLE_GSMINI_HIGH_PD_RIGID_CFG,
)
from tacex_assets.sensors.gelsight_mini.gsmini_cfg import GelSightMiniCfg

from tacex_tasks.utils import DirectLiveVisualizer





class CustomEnvWindow(BaseEnvWindow):
    """Window manager for the RL environment."""

    def __init__(self, env: DirectRLEnvCfg, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """

        super().__init__(env, window_name)

        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:

                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class PoleBalancingEnvCfg(DirectRLEnvCfg):

    viewer: ViewerCfg = ViewerCfg()
    viewer.eye = (1, -0.5, 0.1)
    viewer.lookat = (-19.4, 18.2, -1.1)

    viewer.origin_type = "env"
    viewer.env_idx = 0
    viewer.resolution = (1280, 720)

    debug_vis = True

    ui_window_class_type = CustomEnvWindow

    decimation = 1

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,

        physx=PhysxCfg(
            enable_ccd=True,

        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=5.0,
            dynamic_friction=5.0,
            restitution=0.0,
        ),
    )


    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=1, replicate_physics=True)


    robot: ArticulationCfg = FRANKA_PANDA_ARM_SINGLE_GSMINI_HIGH_PD_RIGID_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(









            joint_pos={
                "panda_joint1": 1.5,
                "panda_joint2": -1.76,
                "panda_joint3": -1.84,
                "panda_joint4": -2.52,
                "panda_joint5": 1.25,
                "panda_joint6": 1.58,
                "panda_joint7": -1.72,
            },
        ),
    )

    ik_controller_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls")

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
                    rot=(1.0, 0.0, 0.0, 0.0),

                ),
            ),
        ],
    )


    pole: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/pole",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{TACEX_ASSETS_DATA_DIR}/Props/pole.usd",

            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=120,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.41336, 0.01123, 0.4637)),
    )


    gsmini = GelSightMiniCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case",
        sensor_camera_cfg=GelSightMiniCfg.SensorCameraCfg(
            prim_path_appendix="/Camera",
            update_period=0,
            resolution=(32, 32),
            data_types=["depth"],
            clipping_range=(0.024, 0.034),
        ),
        device="cuda",
        debug_vis=True,

        optical_sim_cfg=None,

        marker_motion_sim_cfg=None,























        data_types=["camera_depth"],
    )


    action_noise_model = NoiseModelCfg(noise_cfg=UniformNoiseCfg(n_min=-0.001, n_max=0.001, operation="add"))



    reward_terms = {
        "at_obj_reward": {"weight": 0.75, "minimal_distance": 0.005},
        "height_reward": {"weight": 0.25, "w": 10.0, "v": 0.3, "alpha": 0.00067, "target_height_cm": 50},
        "orient_reward": {"weight": 0.25},
        "staying_alive_rew": {"weight": 0.5},
        "termination_penalty": {"weight": -10.0},
        "ee_goal_tracking_penalty": {"weight": -0.001},
        "ee_goal_fine_tracking_reward": {"weight": 0.75, "std": 0.0380},
        "action_rate_penalty": {"weight": -1e-4},
        "joint_vel_penalty": {"weight": -1e-4},
    }


    num_levels = 10

    obj_pos_randomization_range = [-0.05, 0.05]


    episode_length_s = 8.3333 / 2
    action_space = 6
    observation_space = {
        "proprio_obs": 14,
        "vision_obs": [32, 32, 1],
    }
    state_space = 0
    action_scale = 0.05

    x_bounds = (0.15, 0.75)
    y_bounds = (-0.75, 0.75)
    too_far_away_threshold = 0.05
    min_height_threshold = 0.3


class PoleBalancingEnv(DirectRLEnv):
    """RL env in which the robot has to balance a pole towards a goal position."""










    cfg: PoleBalancingEnvCfg

    def __init__(self, cfg: PoleBalancingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation


        self.current_curriculum_level = 0
        self.curriculum_weights = torch.linspace(0, 1, self.cfg.num_levels, device=self.device)

        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)




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

        self._goal_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._goal_pos_w[:, 2] = self.cfg.reward_terms["height_reward"]["target_height_cm"] * 0.01

        self.reward_terms = {}
        for rew_terms in self.cfg.reward_terms:
            self.reward_terms[rew_terms] = torch.zeros((self.num_envs), device=self.device)

        if self.cfg.debug_vis:

            self.visualizers = {
                "Actions": DirectLiveVisualizer(
                    self.cfg.debug_vis, self.num_envs, self._window, visualizer_name="Actions"
                ),
                "Observations": DirectLiveVisualizer(
                    self.cfg.debug_vis, self.num_envs, self._window, visualizer_name="Observations"
                ),
                "Rewards": DirectLiveVisualizer(
                    self.cfg.debug_vis, self.num_envs, self._window, visualizer_name="Rewards"
                ),
                "Metrics": DirectLiveVisualizer(
                    self.cfg.debug_vis, self.num_envs, self._window, visualizer_name="Metrics"
                ),
            }
            self.visualizers["Actions"].terms["actions"] = self.actions

            self.visualizers["Observations"].terms["ee_pos"] = torch.zeros((self.num_envs, 3))
            self.visualizers["Observations"].terms["ee_rot"] = torch.zeros((self.num_envs, 3))
            self.visualizers["Observations"].terms["goal"] = torch.zeros((self.num_envs, 2))
            self.visualizers["Observations"].terms["sensor_output"] = self._get_observations()["policy"]["vision_obs"]

            self.visualizers["Rewards"].terms["rewards"] = torch.zeros((self.num_envs, 10))
            self.visualizers["Rewards"].terms_names["rewards"] = [
                "at_obj_reward",
                "height_reward",
                "orient_reward",
                "staying_alive_rew",
                "termination_penalty",
                "ee_goal_tracking",
                "ee_goal_fine_tracking_reward",
                "action_rate_penalty",
                "joint_vel_penalty",
                "full",
            ]

            self.visualizers["Metrics"].terms["ee_height"] = torch.zeros((self.num_envs, 1))
            self.visualizers["Metrics"].terms["pole_orient_x"] = torch.zeros((self.num_envs, 1))
            self.visualizers["Metrics"].terms["pole_orient_y"] = torch.zeros((self.num_envs, 1))
            self.visualizers["Metrics"].terms["obj_ee_distance"] = torch.zeros((self.num_envs, 1))

            for vis in self.visualizers.values():
                vis.create_visualizer()


        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.object = RigidObject(self.cfg.pole)
        self.scene.rigid_objects["object"] = self.object


        self.scene.clone_environments(copy_from_source=False)


        self._ee_frame = FrameTransformer(self.cfg.ee_frame_cfg)
        self.scene.sensors["ee_frame"] = self._ee_frame

        self.gsmini = GelSightSensor(self.cfg.gsmini)
        self.scene.sensors["gsmini"] = self.gsmini


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


        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)



    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions[:] = self.actions.clone()
        self.actions[:] = actions

        self.processed_actions[:, :] = self.actions * self.cfg.action_scale


        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()

        self._ik_controller.set_command(self.processed_actions, ee_pos_curr_b, ee_quat_curr_b)

    def _apply_action(self):

        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()
        joint_pos = self._robot.data.joint_pos[:, :]

        if ee_pos_curr_b.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            joint_pos_des = self._ik_controller.compute(ee_pos_curr_b, ee_quat_curr_b, jacobian, joint_pos)
        else:
            joint_pos_des = joint_pos.clone()
        self._robot.set_joint_position_target(joint_pos_des)






    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        obj_pos = self.object.data.root_link_pos_w - self.scene.env_origins
        out_of_bounds_x = (obj_pos[:, 0] < self.cfg.x_bounds[0]) | (obj_pos[:, 0] > self.cfg.x_bounds[1])
        out_of_bounds_y = (obj_pos[:, 1] < self.cfg.y_bounds[0]) | (obj_pos[:, 1] > self.cfg.y_bounds[1])

        obj_goal_distance = torch.norm(self._goal_pos_w[:, :2] - self.scene.env_origins[:, :2] - obj_pos[:, :2], dim=1)
        obj_too_far_away = obj_goal_distance > 1.0

        ee_frame_pos = (
            self._ee_frame.data.target_pos_w[..., 0, :] - self.scene.env_origins
        )
        ee_too_far_away = torch.norm(obj_pos - ee_frame_pos, dim=1) > self.cfg.too_far_away_threshold


        pole_orient = euler_xyz_from_quat(self.object.data.root_link_quat_w)
        x = wrap_to_pi(pole_orient[0])
        y = wrap_to_pi(pole_orient[1])
        orient_cond = (torch.abs(x) > math.pi / 4) | (torch.abs(y) > math.pi / 4)

        ee_min_height = ee_frame_pos[:, 2] < self.cfg.min_height_threshold
        obj_min_height = obj_pos[:, 2] < self.cfg.min_height_threshold

        reset_cond = (
            out_of_bounds_x
            | out_of_bounds_y
            | obj_too_far_away
            | ee_too_far_away
            | orient_cond
            | ee_min_height
            | obj_min_height
        )

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return reset_cond, time_out


    def _get_rewards(self) -> torch.Tensor:

        obj_pos = self.object.data.root_link_pos_w
        ee_frame_pos = self._ee_frame.data.target_pos_w[
            ..., 0, :
        ]


        object_ee_distance = torch.norm(obj_pos - ee_frame_pos, dim=1)

        self.reward_terms["at_obj_reward"][:] = torch.where(
            object_ee_distance <= self.cfg.reward_terms["at_obj_reward"]["minimal_distance"],
            self.cfg.reward_terms["at_obj_reward"]["weight"],
            0.0,
        )

        height_diff = (
            self.cfg.reward_terms["height_reward"]["target_height_cm"] - ee_frame_pos[:, 2] * 100
        ) * 0.1
        height_reward = -(
            self.cfg.reward_terms["height_reward"]["w"] * height_diff**2
            + self.cfg.reward_terms["height_reward"]["v"]
            * torch.log(height_diff**2 + self.cfg.reward_terms["height_reward"]["alpha"])
        ).clamp(-1, 1)

        height_reward = torch.where(
            (ee_frame_pos[:, 2] <= self.cfg.min_height_threshold), height_reward - 10, height_reward
        )
        self.reward_terms["height_reward"][:] = height_reward * self.cfg.reward_terms["height_reward"]["weight"]


        pole_orient = euler_xyz_from_quat(self.object.data.root_link_quat_w)
        x = wrap_to_pi(pole_orient[0])
        y = wrap_to_pi(pole_orient[1])
        orient_reward = torch.where(
            (torch.abs(x) < math.pi / 8) | (torch.abs(y) < math.pi / 8),
            1.0 * self.cfg.reward_terms["orient_reward"]["weight"],
            0.0,
        )
        self.reward_terms["orient_reward"][:] = orient_reward

        ee_goal_distance = torch.norm(ee_frame_pos - self._goal_pos_w, dim=1)
        self.reward_terms["ee_goal_tracking_penalty"][:] = (
            torch.square(ee_goal_distance * 100) * self.cfg.reward_terms["ee_goal_tracking_penalty"]["weight"]
        )
        self.reward_terms["ee_goal_fine_tracking_reward"][:] = (
            1 - torch.tanh(ee_goal_distance / self.cfg.reward_terms["ee_goal_fine_tracking_reward"]["std"]) ** 2
        )

        self.reward_terms["staying_alive_rew"][:] = (
            self.cfg.reward_terms["staying_alive_rew"]["weight"] * (1.0 - self.reset_terminated.float())
        )[:]

        self.reward_terms["termination_penalty"][:] = (
            self.cfg.reward_terms["termination_penalty"]["weight"] * self.reset_terminated.float()
        )



        self.reward_terms["action_rate_penalty"][:] = self.cfg.reward_terms["action_rate_penalty"][
            "weight"
        ] * torch.sum(torch.square(self.actions - self.prev_actions), dim=1)

        self.reward_terms["joint_vel_penalty"][:] = self.cfg.reward_terms["joint_vel_penalty"]["weight"] * torch.sum(
            torch.square(self._robot.data.joint_vel[:, :]), dim=1
        )

        rewards = (
            +self.reward_terms["at_obj_reward"]
            + self.reward_terms["height_reward"]
            + self.reward_terms["orient_reward"]

            + self.reward_terms["ee_goal_fine_tracking_reward"]
            + self.reward_terms["staying_alive_rew"]
            + self.reward_terms["termination_penalty"]
            + self.reward_terms["action_rate_penalty"]
            + self.reward_terms["joint_vel_penalty"]
        )

        self.extras["log"] = {}
        for rew_name, rew in self.reward_terms.items():
            self.extras["log"][f"rew_{rew_name}"] = rew.mean()

        if self.cfg.debug_vis:
            for i, name in enumerate(self.reward_terms.keys()):
                self.visualizers["Rewards"].terms["rewards"][:, i] = self.reward_terms[name]
            self.visualizers["Rewards"].terms["rewards"][:, -1] = rewards

            self.visualizers["Metrics"].terms["ee_height"] = ee_frame_pos[:, 2].reshape(-1, 1)
            self.visualizers["Metrics"].terms["pole_orient_x"] = torch.rad2deg(x).reshape(-1, 1)
            self.visualizers["Metrics"].terms["pole_orient_y"] = torch.rad2deg(y).reshape(-1, 1)
            self.visualizers["Metrics"].terms["obj_ee_distance"] = object_ee_distance.reshape(-1, 1)

        return rewards


    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)


        obj_pos = self.object.data.default_root_state[env_ids]
        obj_pos[:, :3] += self.scene.env_origins[env_ids]
        self.object.write_root_state_to_sim(obj_pos, env_ids=env_ids)


        joint_pos = self._robot.data.default_joint_pos[env_ids]



        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


        self._goal_pos_w[env_ids, :2] = (
            self.object.data.default_root_state[env_ids, :2]
            + self.scene.env_origins[env_ids, :2]
            + sample_uniform(
                self.cfg.obj_pos_randomization_range[0],
                self.cfg.obj_pos_randomization_range[1],
                (len(env_ids), 2),
                self.device,
            )
        )

        self.prev_actions[env_ids] = 0.0


        self.gsmini.reset(env_ids=env_ids)


    def _get_observations(self) -> dict:
        """The position of the object in the robot's root frame."""

        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()
        ee_frame_orient = euler_xyz_from_quat(ee_quat_curr_b)
        x = wrap_to_pi(ee_frame_orient[0]).unsqueeze(1)
        y = wrap_to_pi(ee_frame_orient[1]).unsqueeze(1)
        z = wrap_to_pi(ee_frame_orient[2]).unsqueeze(1)

        goal_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_link_state_w[:, :3], self._robot.data.root_link_state_w[:, 3:7], self._goal_pos_w
        )
        proprio_obs = torch.cat(
            (ee_pos_curr_b, x, y, z, goal_pos_b[:, :2], self.actions),
            dim=-1,
        )
        vision_obs = self.gsmini._data.output["camera_depth"]

        obs = {"proprio_obs": proprio_obs, "vision_obs": vision_obs}


        if self.cfg.debug_vis:
            self.visualizers["Observations"].terms["ee_pos"] = ee_pos_curr_b[:, :3]
            self.visualizers["Observations"].terms["ee_rot"][:, :1] = x
            self.visualizers["Observations"].terms["ee_rot"][:, 1:2] = y
            self.visualizers["Observations"].terms["ee_rot"][:, 2:3] = z
            self.visualizers["Observations"].terms["sensor_output"] = vision_obs.clone()
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
                marker_cfg = VisualizationMarkersCfg(
                    markers={
                        "sphere": sim_utils.SphereCfg(
                            radius=0.005,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), opacity=0.5),
                        ),
                    }
                )

                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)

            self.goal_pos_visualizer.set_visibility(True)

        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)



    def _debug_vis_callback(self, event):

        translations = self._goal_pos_w.clone()
        self.goal_pos_visualizer.visualize(translations=translations)
