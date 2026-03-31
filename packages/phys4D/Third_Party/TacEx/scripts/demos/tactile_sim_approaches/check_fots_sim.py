from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Ball rolling experiment with a Franka, which is equipped with one GelSight Mini Sensor."
)
parser.add_argument("--num_envs", type=int, default=10, help="Number of environments to spawn.")
parser.add_argument("--sys", type=bool, default=True, help="Whether to track system utilization.")
parser.add_argument(
    "--debug_vis",
    default=True,
    action="store_true",
    help="Whether to render tactile images in the# append AppLauncher cli args",
)
AppLauncher.add_app_launcher_args(parser)


args_cli = parser.parse_args()
args_cli.enable_cameras = True


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import datetime
import json
import numpy as np
import psutil
import time
import torch
import traceback
from pathlib import Path

import carb
import pynvml

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg, OffsetCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    sample_uniform,
)

from tacex import GelSightSensor
from tacex.simulation_approaches.fots import FOTSMarkerSimulatorCfg

from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets.robots.franka.franka_gsmini_single_rigid import (
    FRANKA_PANDA_ARM_SINGLE_GSMINI_HIGH_PD_RIGID_CFG,
)
from tacex_assets.sensors.gelsight_mini.gsmini_cfg import GelSightMiniCfg





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
class BallRollingEnvCfg(DirectRLEnvCfg):

    viewer: ViewerCfg = ViewerCfg()
    viewer.eye = (1.9, 1.4, 0.3)
    viewer.lookat = (-1.5, -1.9, -1.1)




    debug_vis = True

    ui_window_class_type = CustomEnvWindow

    decimation = 1

    sim: SimulationCfg = SimulationCfg(
        dt=0.01,
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


    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=20,
        env_spacing=1.5,
        replicate_physics=True,
        lazy_sensor_update=True,
    )


    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0]),
        spawn=sim_utils.GroundPlaneCfg(),
    )


    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


    plate = AssetBaseCfg(
        prim_path="/World/envs/env_.*/plate",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0, 0)),
        spawn=sim_utils.UsdFileCfg(usd_path=f"{TACEX_ASSETS_DATA_DIR}/Props/plate.usd"),
    )

    ball = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ball",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.015)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{TACEX_ASSETS_DATA_DIR}/Props/ball_wood.usd",

            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                kinematic_enabled=False,
                disable_gravity=False,
            ),
        ),
    )

    robot: ArticulationCfg = FRANKA_PANDA_ARM_SINGLE_GSMINI_HIGH_PD_RIGID_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": 0.0,
                "panda_joint3": 0.0,
                "panda_joint4": -2.46,
                "panda_joint5": 0.0,
                "panda_joint6": 2.5,
                "panda_joint7": 0.741,
            },
        ),
    )


    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
    marker_cfg.prim_path = "/Visuals/FrameTransformer"

    gsmini = GelSightMiniCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case",
        sensor_camera_cfg=GelSightMiniCfg.SensorCameraCfg(
            prim_path_appendix="/Camera",
            update_period=0,
            resolution=(32, 24),
            data_types=["depth"],
            clipping_range=(0.024, 0.034),
        ),
        device="cuda",
        debug_vis=True,

        marker_motion_sim_cfg=FOTSMarkerSimulatorCfg(
            lamb=[0.00125, 0.00021, 0.00038],
            pyramid_kernel_size=[51, 21, 11, 5],
            kernel_size=5,
            marker_params=FOTSMarkerSimulatorCfg.MarkerParams(
                num_markers_col=11, num_markers_row=9, num_markers=99, x0=15, y0=26, dx=26, dy=29
            ),
            tactile_img_res=(320, 240),
            device="cuda",
            frame_transformer_cfg=FrameTransformerCfg(
                prim_path="/World/envs/env_.*/Robot/gelsight_mini_gelpad",

                source_frame_offset=OffsetCfg(

                ),
                target_frames=[FrameTransformerCfg.FrameCfg(prim_path="/World/envs/env_.*/ball")],
                debug_vis=True,
                visualizer_cfg=marker_cfg,
            ),
        ),
        data_types=["marker_motion", "tactile_rgb"],
    )

    gsmini = gsmini.replace(
        optical_sim_cfg=gsmini.optical_sim_cfg.replace(
            with_shadow=False,
            device="cuda:0",
        )
    )

    ik_controller_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")

    ball_radius = 0.005

    obj_pos_randomization_range = [-0.15, 0.15]


    episode_length_s = 0
    action_space = 0
    observation_space = 0
    state_space = 0


class BallRollingEnv(DirectRLEnv):
    cfg: BallRollingEnvCfg

    def __init__(self, cfg: BallRollingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)



        self._ik_controller = DifferentialIKController(
            cfg=self.cfg.ik_controller_cfg, num_envs=self.num_envs, device=self.device
        )

        body_ids, body_names = self._robot.find_bodies("panda_hand")

        self._body_idx = body_ids[0]
        self._body_name = body_names[0]



        self._jacobi_body_idx = self._body_idx - 1



        self._offset_pos = torch.tensor([0.0, 0.0, 0.131], device=self.device).repeat(self.num_envs, 1)
        self._offset_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)



        self.ik_commands = torch.zeros((self.num_envs, self._ik_controller.action_dim), device=self.device)

        self.ik_commands[:, 3:] += torch.tensor([0, 1, 0, 0], device=self.device)



        ball_radius = self.cfg.ball_radius
        gel_length = self.cfg.gsmini.gelpad_dimensions.length
        gel_width = self.cfg.gsmini.gelpad_dimensions.width
        gel_height = self.cfg.gsmini.gelpad_dimensions.height
        z_offset = ball_radius - gel_height / 2

        above_ball = torch.tensor([0, 0, ball_radius * 2], device=self.device)
        center = torch.tensor([0, 0, z_offset], device=self.device)
        backward = torch.tensor([-gel_length / 2, 0, z_offset], device=self.device)
        forward = torch.tensor([gel_length, 0, z_offset], device=self.device)
        left = torch.tensor([0, gel_width / 2, z_offset], device=self.device)
        right = torch.tensor([0, -gel_width, z_offset], device=self.device)

        self.pattern_offsets = [
            above_ball,
            center,
            backward,
            forward,
            center,
            left,
            right,

            center,
            backward,
            forward,
            center,
            left,
            right,
            center,
        ]

        self.current_goal_idx = 0
        self.num_step_goal_change = 50
        self.step_count = 0


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

        ball_pos = self.object.data.root_link_pos_w - self.scene.env_origins

        if self.step_count % self.num_step_goal_change == 0:
            self.current_goal_idx = (self.current_goal_idx + 1) % len(self.pattern_offsets)
        self.ik_commands[:, :3] = ball_pos + self.pattern_offsets[self.current_goal_idx]


        self.ik_commands[:, :2] += sample_uniform(-0.002, 0.002, (self.num_envs, 2), self.device)
        self._ik_controller.set_command(self.ik_commands)

    def _apply_action(self):

        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()
        joint_pos = self._robot.data.joint_pos[:, :]

        if ee_pos_curr_b.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            joint_pos_des = self._ik_controller.compute(ee_pos_curr_b, ee_quat_curr_b, jacobian, joint_pos)
        else:
            joint_pos_des = joint_pos.clone()


        if self.current_goal_idx == 3 or self.current_goal_idx == 4:
            joint_pos_des[:, 6] = -joint_pos_des[:, 6]

        self._robot.set_joint_position_target(joint_pos_des)

        self.step_count += 1




    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        pass


    def _get_rewards(self) -> torch.Tensor:
        pass

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        obj_pos = self.object.data.default_root_state[env_ids]
        obj_pos[:, :3] += self.scene.env_origins[env_ids]






        self.object.write_root_state_to_sim(obj_pos, env_ids=env_ids)


        joint_pos = (
            self._robot.data.default_joint_pos[env_ids]






        )
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


        self.actions[env_ids] = 0.0
        self._ik_controller.reset(env_ids)


        self.current_goal_idx = 0


    def _get_observations(self) -> dict:
        pass

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


"""
System diagnosis
-> adapted from benchmark_cameras.py script of IsaacLab
"""


def _get_utilization_percentages(reset: bool = False, max_values: list[float] = [0.0, 0.0, 0.0, 0.0]) -> list[float]:
    """Get the maximum CPU, RAM, GPU utilization (processing), and
    GPU memory usage percentages since the last time reset was true."""
    if reset:
        max_values[:] = [0, 0, 0, 0]



    cpu_usage = psutil.cpu_percent(interval=None)
    max_values[0] = max(max_values[0], cpu_usage)


    memory_info = psutil.virtual_memory()
    ram_usage = memory_info.percent
    max_values[1] = max(max_values[1], ram_usage)


    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)


            gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_processing_utilization_percent = gpu_utilization.gpu
            max_values[2] = max(max_values[2], gpu_processing_utilization_percent)


            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_memory_total = memory_info.total
            gpu_memory_used = memory_info.used
            gpu_memory_utilization_percent = (gpu_memory_used / gpu_memory_total) * 100
            max_values[3] = max(max_values[3], gpu_memory_utilization_percent)
    else:
        gpu_processing_utilization_percent = None
        gpu_memory_utilization_percent = None

    return max_values


def run_simulator(env: BallRollingEnv):
    """Runs the simulation loop."""


    if env.cfg.gsmini.debug_vis:
        for data_type in env.cfg.gsmini.data_types:
            env.gsmini._prim_view.prims[0].GetAttribute(f"debug_{data_type}").Set(True)


    timestamp = f"{datetime.datetime.now():%Y-%m-%d-%H_%M_%S}"
    output_dir = Path(__file__).parent.resolve()
    file_name = str(output_dir) + f"/envs_{env.num_envs}_{timestamp}.txt"

    frame_times_physics = []
    frame_times_tactile = []

    save_after_num_resets = 10
    current_num_resets = 0

    if torch.cuda.is_available():
        pynvml.nvmlInit()
    system_utilization_analytics = _get_utilization_percentages(reset=False)

    print(f"Starting simulation with {env.num_envs} envs")
    print("Number of steps till reset: ", len(env.pattern_offsets) * env.num_step_goal_change)

    total_sim_time = time.time()

    while simulation_app.is_running():


        if (
            env.step_count % (len(env.pattern_offsets) * env.num_step_goal_change) == 0
        ):
            print(f"[INFO]: Env reset num {current_num_resets}")
            env.reset()
            system_utilization_analytics = _get_utilization_percentages(reset=True)

            if len(frame_times_physics) != 0:
                print("Current total amount of 'in-contact' frames per env: ", len(frame_times_physics))
                print(f"Total sim time currently: {time.time() - total_sim_time:8.4f}ms")
                print(
                    f"Avg physics_sim time per env:    {np.mean(np.array(frame_times_physics) / env.num_envs):8.4f}ms"
                )
                print(
                    f"Avg tactile_sim time per env:    {np.mean(np.array(frame_times_tactile) / env.num_envs):8.4f}ms"
                )
                print(
                    f"| CPU:{system_utilization_analytics[0]}% | "
                    f"RAM:{system_utilization_analytics[1]}% | "
                    f"GPU Compute:{system_utilization_analytics[2]}% | "
                    f"GPU Memory: {system_utilization_analytics[3]:.2f}% |"
                )
                print("")


                if current_num_resets == save_after_num_resets:
                    print("*" * 15)
                    print("Writing performance data into ", file_name)
                    print("*" * 15)
                    with open(file_name, "a+") as f:
                        f.write("Sensor Config: \n")
                        f.write(json.dumps(env.cfg.gsmini.to_dict(), indent=2))
                        f.write("\n")
                        f.write("GPU Info: \n")
                        f.write(f"Name: {pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(0))} \n")
                        f.write(f"Driver: {pynvml.nvmlSystemGetDriverVersion()} \n")
                        f.write("\n")
                        f.write("Performance data: \n")
                        f.write(f"num_envs: {env.num_envs} \n")
                        f.write(
                            f"Total amount of 'in-contact' frames per env (ran pattern {current_num_resets} times):"
                            f" {len(frame_times_physics)}\n"
                        )
                        f.write(f"Total sim time: {time.time() - total_sim_time:8.4f}ms \n")
                        f.write(
                            "Avg physics_sim time for one frame per env:   "
                            f" {np.mean(np.array(frame_times_physics) / env.num_envs):8.4f}ms \n"
                        )
                        f.write(
                            "Avg tactile_sim time for one frame per env:   "
                            f" {np.mean(np.array(frame_times_tactile) / env.num_envs):8.4f}ms \n"
                        )
                        f.write("\n")
                        f.write(
                            f"| CPU:{system_utilization_analytics[0]}% | "
                            f"RAM:{system_utilization_analytics[1]}% | "
                            f"GPU Compute:{system_utilization_analytics[2]}% | "
                            f"GPU Memory: {system_utilization_analytics[3]:.2f}% |"
                        )
                        f.write("\n")
                    frame_times_physics = []
                    frame_times_tactile = []
                    break
            current_num_resets += 1


        physics_start = time.time()
        env._pre_physics_step(None)
        env._apply_action()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)

        env.sim.render()

        env.scene.update(dt=env.physics_dt)
        physics_end = time.time()



        tactile_sim_start = time.time()
        env.gsmini.update(dt=env.physics_dt, force_recompute=True)

        tactile_sim_end = time.time()




        (contact_idx,) = torch.where(env.gsmini._indentation_depth > 0)
        if contact_idx.shape[0] != 0:
            frame_times_physics.append(1000 * (physics_end - physics_start))
            frame_times_tactile.append(1000 * (tactile_sim_end - tactile_sim_start))
    env.close()

    pynvml.nvmlShutdown()


def main():
    """Main function."""

    env_cfg = BallRollingEnvCfg()

    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.gsmini.debug_vis = args_cli.debug_vis

    experiment = BallRollingEnv(env_cfg)


    print("[INFO]: Setup complete...")

    run_simulator(env=experiment)


if __name__ == "__main__":
    try:

        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:

        simulation_app.close()
