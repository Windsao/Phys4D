"""Collecting tactile data of the shapes from https://danfergo.github.io/gelsight-simulation/.

Use
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Control Franka, which is equipped with one GelSight Mini Sensor, by moving the Frame in the GUI"
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
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

import traceback
from contextlib import suppress
from pathlib import Path

import omni.ui

with suppress(ImportError):

    import isaacsim.gui.components.ui_utils as ui_utils

import numpy as np
import torch

import carb
import pynvml
from isaacsim.core.api.objects import VisualCuboid
from isaacsim.core.prims import XFormPrim

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import (
    Articulation,
    ArticulationCfg,
    AssetBaseCfg,
    RigidObject,
    RigidObjectCfg,
)
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import PhysxCfg, RenderCfg, SimulationCfg
from isaaclab.utils import configclass

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

        self.object_names = list(create_shapes_cfg().keys())
        self.current_object_name = self.object_names[0]


        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:

                    self._create_debug_vis_ui_element("targets", self.env)

        with self.ui_window_elements["main_vstack"]:
            self._build_control_frame()

            self.ui_window_elements["debug_frame"].collapsed = True
            self.ui_window_elements["sim_frame"].collapsed = True

    def _build_control_frame(self):
        self.ui_window_elements["action_frame"] = omni.ui.CollapsableFrame(
            title="Shape Touching Demo Script",
            width=omni.ui.Fraction(1),
            height=0,
            collapsed=False,
            style=ui_utils.get_style(),
            horizontal_scrollbar_policy=omni.ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            vertical_scrollbar_policy=omni.ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
        )
        with self.ui_window_elements["action_frame"]:
            self.ui_window_elements["action_vstack"] = omni.ui.VStack(spacing=5, height=50)
            with self.ui_window_elements["action_vstack"]:
                objects_dropdown_cfg = {
                    "label": "Objects",
                    "type": "dropdown",
                    "default_val": 0,
                    "items": self.object_names,
                    "tooltip": "Select an action for the gripper",
                    "on_clicked_fn": self._set_main_obj,
                }
                self.ui_window_elements["object_dropdown"] = ui_utils.dropdown_builder(**objects_dropdown_cfg)

                self.ui_window_elements["reset_button"] = ui_utils.btn_builder(
                    type="button",
                    text="Reset Env",
                    tooltip="Resets the environment, i.e. the objects are spawned back at their initial position.",
                    on_clicked_fn=self._reset_env,
                )

    def get_current_object_name(self):
        current_obj_idx = (
            self.ui_window_elements["object_dropdown"].get_item_value_model().get_value_as_int()
        )
        self.current_object_name = self.object_names[current_obj_idx]
        return self.current_object_name

    def _set_main_obj(self, value):
        print("Set new main obj ", value)

        old_obj: RigidObject = self.env.scene.rigid_objects[self.current_object_name]

        obj_name = self.get_current_object_name()
        new_obj: RigidObject = self.env.scene.rigid_objects[obj_name]
        new_pose = new_obj.data.root_pose_w


        old_obj.write_root_pose_to_sim(new_pose)
        new_obj.write_root_pose_to_sim(self.env.main_pose)

    def _reset_env(self):
        self.env.reset()


def create_shapes_cfg() -> dict[str, RigidObjectCfg]:
    """Creates RigidObjectCfg's for each usd file in the `{TACEX_ASSETS_DATA_DIR}/Props/tactile_test_shapes/` directory.

    The objects are spawned on a line along the y axis.
    Returns:
        shapes: dict of names and corresponding RigidObjectCfg's
    """
    shapes = {}
    usd_files_path = list(Path(f"{TACEX_ASSETS_DATA_DIR}/Props/tactile_test_shapes/").glob("*.usd"))

    for i, file_path in enumerate(usd_files_path):
        file_name = file_path.stem

        shapes[file_name] = RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/{file_name}",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.025 * i, 0.02]),
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(file_path),

                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    kinematic_enabled=True,
                    disable_gravity=False,
                ),
            ),
        )

    return shapes


@configclass
class BallRollingEnvCfg(DirectRLEnvCfg):

    viewer: ViewerCfg = ViewerCfg()


    viewer.eye = (0.55, -0.06, 0.025)
    viewer.lookat = (-4.8, 6.0, -0.2)

    debug_vis = True

    ui_window_class_type = CustomEnvWindow

    decimation = 1

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
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
        render=RenderCfg(enable_translucency=True),
    )


    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=1.5,
        replicate_physics=True,
        lazy_sensor_update=True,
    )


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


    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    robot: ArticulationCfg = FRANKA_PANDA_ARM_SINGLE_GSMINI_HIGH_PD_RIGID_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )

    shapes: dict = create_shapes_cfg()


    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
    marker_cfg.prim_path = "/Visuals/FrameTransformer"

    gsmini = GelSightMiniCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case",
        sensor_camera_cfg=GelSightMiniCfg.SensorCameraCfg(
            prim_path_appendix="/Camera",
            update_period=0,
            resolution=(320, 240),
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
                num_markers_col=11,
                num_markers_row=9,
                num_markers=99,
                x0=15,
                y0=26,
                dx=26,
                dy=29,
            ),
            tactile_img_res=(320, 240),
            device="cuda",
            frame_transformer_cfg=FrameTransformerCfg(
                prim_path="/World/envs/env_.*/Robot/gelsight_mini_gelpad",

                source_frame_offset=OffsetCfg(

                ),
                target_frames=[
                    FrameTransformerCfg.FrameCfg(prim_path=f"/World/envs/env_.*/{obj_name}")
                    for obj_name in list(shapes.keys())
                ],
                debug_vis=True,
                visualizer_cfg=marker_cfg,
            ),
        ),
        data_types=["marker_motion", "tactile_rgb"],
    )

    gsmini.optical_sim_cfg = gsmini.optical_sim_cfg.replace(
        with_shadow=False,
        device="cuda",
        tactile_img_res=(320, 240),
    )

    ik_controller_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")

    main_pose = [0.5, 0.0, 0.02, 1, 0, 0, 0]


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


        self.step_count = 0

        self.goal_prim_view = None

        self.main_pose = torch.tensor([self.cfg.main_pose], device=self.device)


        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        for obj_name, cfg in self.cfg.shapes.items():
            self.scene.rigid_objects[obj_name] = RigidObject(cfg)

        self.object = list(self.scene.rigid_objects.values())[0]


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


        ground = self.cfg.ground
        ground.spawn.func(
            ground.prim_path,
            ground.spawn,
            translation=ground.init_state.pos,
            orientation=ground.init_state.rot,
        )


        VisualCuboid(
            prim_path="/Goal",
            size=0.01,
            position=np.array([0.5, 0.0, 0.021]),
            orientation=np.array([0, 1, 0, 0]),
            visible=False,
        )


        VisualCuboid(
            prim_path="/Visuals/main_area",
            size=0.02,
            position=np.array([0.5, 0.0, -0.005]),
            color=np.array([255.0, 0.0, 0.0]),
        )


        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)



    def _pre_physics_step(self, actions: torch.Tensor):
        self._ik_controller.set_command(self.ik_commands)

    def _apply_action(self):

        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()
        joint_pos = self._robot.data.joint_pos[:, :]


        if ee_pos_curr_b.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            joint_pos_des = self._ik_controller.compute(ee_pos_curr_b, ee_quat_curr_b, jacobian, joint_pos)
        else:
            joint_pos_des = joint_pos.clone()
        self._robot.set_joint_position_target(joint_pos_des)

        self.step_count += 1




    def _get_dones(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass


    def _get_rewards(self) -> torch.Tensor:
        pass

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)


        if self.goal_prim_view is not None:

            goal_pos = self.main_pose[:, :3]
            goal_pos[:, 2] += 0.001

            goal_orient = torch.tensor([[0, 1, 0, 0]], device=self.device)
            self.goal_prim_view.set_world_poses(positions=goal_pos, orientations=goal_orient)


        joint_pos = self._robot.data.default_joint_pos[env_ids]

        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


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


def run_simulator(env: BallRollingEnv):
    """Runs the simulation loop."""

    if env.cfg.gsmini.debug_vis:
        for data_type in env.cfg.gsmini.data_types:
            env.gsmini._prim_view.prims[0].GetAttribute(f"debug_{data_type}").Set(True)






    print(f"Starting simulation with {env.num_envs} envs")

    env.reset()
    env.goal_prim_view = XFormPrim(prim_paths_expr="/Goal", name="Goal", usd=True)


    while simulation_app.is_running():

        env._pre_physics_step(None)
        env._apply_action()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)

        positions, orientations = env.goal_prim_view.get_world_poses()
        env.ik_commands[:, :3] = positions - env.scene.env_origins
        env.ik_commands[:, 3:] = orientations


        env.scene.update(dt=env.physics_dt)

        env.sim.render()




    env.close()

    pynvml.nvmlShutdown()


def main():
    """Main function."""

    env_cfg = BallRollingEnvCfg()

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
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
