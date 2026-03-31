




"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher


simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import math
import scipy.spatial.transform as tf
import torch

import isaacsim.core.utils.stage as stage_utils
import pytest

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg, OffsetCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass




from isaaclab_assets.robots.anymal import ANYMAL_C_CFG


def quat_from_euler_rpy(roll, pitch, yaw, degrees=False):
    """Converts Euler XYZ to Quaternion (w, x, y, z)."""
    quat = tf.Rotation.from_euler("xyz", (roll, pitch, yaw), degrees=degrees).as_quat()
    return tuple(quat[[3, 0, 1, 2]].tolist())


def euler_rpy_apply(rpy, xyz, degrees=False):
    """Applies rotation from Euler XYZ on position vector."""
    rot = tf.Rotation.from_euler("xyz", rpy, degrees=degrees)
    return tuple(rot.apply(xyz).tolist())


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Example scene configuration."""


    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")


    robot = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


    frame_transformer: FrameTransformerCfg = None


    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 0.0, 5)),
    )


@pytest.fixture
def sim():
    """Create a simulation context."""

    stage_utils.create_new_stage()

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device="cpu"))

    sim.set_camera_view(eye=(5.0, 5.0, 5.0), target=(0.0, 0.0, 0.0))
    yield sim

    sim.clear_all_callbacks()
    sim.clear_instance()


def test_frame_transformer_feet_wrt_base(sim):
    """Test feet transformations w.r.t. base source frame.

    In this test, the source frame is the robot base.
    """

    scene_cfg = MySceneCfg(num_envs=32, env_spacing=5.0, lazy_sensor_update=False)
    scene_cfg.frame_transformer = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                name="LF_FOOT_USER",
                prim_path="{ENV_REGEX_NS}/Robot/LF_SHANK",
                offset=OffsetCfg(
                    pos=euler_rpy_apply(rpy=(0, 0, -math.pi / 2), xyz=(0.08795, 0.01305, -0.33797)),
                    rot=quat_from_euler_rpy(0, 0, -math.pi / 2),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                name="RF_FOOT_USER",
                prim_path="{ENV_REGEX_NS}/Robot/RF_SHANK",
                offset=OffsetCfg(
                    pos=euler_rpy_apply(rpy=(0, 0, math.pi / 2), xyz=(0.08795, -0.01305, -0.33797)),
                    rot=quat_from_euler_rpy(0, 0, math.pi / 2),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                name="LH_FOOT_USER",
                prim_path="{ENV_REGEX_NS}/Robot/LH_SHANK",
                offset=OffsetCfg(
                    pos=euler_rpy_apply(rpy=(0, 0, -math.pi / 2), xyz=(-0.08795, 0.01305, -0.33797)),
                    rot=quat_from_euler_rpy(0, 0, -math.pi / 2),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                name="RH_FOOT_USER",
                prim_path="{ENV_REGEX_NS}/Robot/RH_SHANK",
                offset=OffsetCfg(
                    pos=euler_rpy_apply(rpy=(0, 0, math.pi / 2), xyz=(-0.08795, -0.01305, -0.33797)),
                    rot=quat_from_euler_rpy(0, 0, math.pi / 2),
                ),
            ),
        ],
    )
    scene = InteractiveScene(scene_cfg)


    sim.reset()


    feet_indices, feet_names = scene.articulations["robot"].find_bodies(["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"])

    target_frame_names = scene.sensors["frame_transformer"].data.target_frame_names


    target_frame_names = [name.split("_USER")[0] for name in target_frame_names]


    reordering_indices = [feet_names.index(name) for name in target_frame_names]
    feet_indices = [feet_indices[i] for i in reordering_indices]


    default_actions = scene.articulations["robot"].data.default_joint_pos.clone()

    sim_dt = sim.get_physics_dt()

    for count in range(100):

        if count % 25 == 0:

            root_state = scene.articulations["robot"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            joint_pos = scene.articulations["robot"].data.default_joint_pos
            joint_vel = scene.articulations["robot"].data.default_joint_vel


            scene.articulations["robot"].write_root_pose_to_sim(root_state[:, :7])
            scene.articulations["robot"].write_root_velocity_to_sim(root_state[:, 7:])
            scene.articulations["robot"].write_joint_state_to_sim(joint_pos, joint_vel)

            scene.reset()


        robot_actions = default_actions + 0.5 * torch.randn_like(default_actions)
        scene.articulations["robot"].set_joint_position_target(robot_actions)

        scene.write_data_to_sim()

        sim.step()

        scene.update(sim_dt)



        root_pose_w = scene.articulations["robot"].data.root_pose_w
        feet_pos_w_gt = scene.articulations["robot"].data.body_pos_w[:, feet_indices]
        feet_quat_w_gt = scene.articulations["robot"].data.body_quat_w[:, feet_indices]

        source_pos_w_tf = scene.sensors["frame_transformer"].data.source_pos_w
        source_quat_w_tf = scene.sensors["frame_transformer"].data.source_quat_w
        feet_pos_w_tf = scene.sensors["frame_transformer"].data.target_pos_w
        feet_quat_w_tf = scene.sensors["frame_transformer"].data.target_quat_w


        torch.testing.assert_close(root_pose_w[:, :3], source_pos_w_tf)
        torch.testing.assert_close(root_pose_w[:, 3:], source_quat_w_tf)
        torch.testing.assert_close(feet_pos_w_gt, feet_pos_w_tf)
        torch.testing.assert_close(feet_quat_w_gt, feet_quat_w_tf)


        feet_pos_source_tf = scene.sensors["frame_transformer"].data.target_pos_source
        feet_quat_source_tf = scene.sensors["frame_transformer"].data.target_quat_source
        for index in range(len(feet_indices)):

            foot_pos_b, foot_quat_b = math_utils.subtract_frame_transforms(
                root_pose_w[:, :3], root_pose_w[:, 3:], feet_pos_w_tf[:, index], feet_quat_w_tf[:, index]
            )

            torch.testing.assert_close(feet_pos_source_tf[:, index], foot_pos_b)
            torch.testing.assert_close(feet_quat_source_tf[:, index], foot_quat_b)


def test_frame_transformer_feet_wrt_thigh(sim):
    """Test feet transformation w.r.t. thigh source frame."""

    scene_cfg = MySceneCfg(num_envs=32, env_spacing=5.0, lazy_sensor_update=False)
    scene_cfg.frame_transformer = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/LF_THIGH",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                name="LF_FOOT_USER",
                prim_path="{ENV_REGEX_NS}/Robot/LF_SHANK",
                offset=OffsetCfg(
                    pos=euler_rpy_apply(rpy=(0, 0, -math.pi / 2), xyz=(0.08795, 0.01305, -0.33797)),
                    rot=quat_from_euler_rpy(0, 0, -math.pi / 2),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                name="RF_FOOT_USER",
                prim_path="{ENV_REGEX_NS}/Robot/RF_SHANK",
                offset=OffsetCfg(
                    pos=euler_rpy_apply(rpy=(0, 0, math.pi / 2), xyz=(0.08795, -0.01305, -0.33797)),
                    rot=quat_from_euler_rpy(0, 0, math.pi / 2),
                ),
            ),
        ],
    )
    scene = InteractiveScene(scene_cfg)


    sim.reset()


    source_frame_index = scene.articulations["robot"].find_bodies("LF_THIGH")[0][0]
    feet_indices, feet_names = scene.articulations["robot"].find_bodies(["LF_FOOT", "RF_FOOT"])

    user_feet_names = [f"{name}_USER" for name in feet_names]
    assert scene.sensors["frame_transformer"].data.target_frame_names == user_feet_names


    default_actions = scene.articulations["robot"].data.default_joint_pos.clone()

    sim_dt = sim.get_physics_dt()

    for count in range(100):

        if count % 25 == 0:

            root_state = scene.articulations["robot"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            joint_pos = scene.articulations["robot"].data.default_joint_pos
            joint_vel = scene.articulations["robot"].data.default_joint_vel


            scene.articulations["robot"].write_root_pose_to_sim(root_state[:, :7])
            scene.articulations["robot"].write_root_velocity_to_sim(root_state[:, 7:])
            scene.articulations["robot"].write_joint_state_to_sim(joint_pos, joint_vel)

            scene.reset()


        robot_actions = default_actions + 0.5 * torch.randn_like(default_actions)
        scene.articulations["robot"].set_joint_position_target(robot_actions)

        scene.write_data_to_sim()

        sim.step()

        scene.update(sim_dt)



        source_pose_w_gt = scene.articulations["robot"].data.body_state_w[:, source_frame_index, :7]
        feet_pos_w_gt = scene.articulations["robot"].data.body_pos_w[:, feet_indices]
        feet_quat_w_gt = scene.articulations["robot"].data.body_quat_w[:, feet_indices]

        source_pos_w_tf = scene.sensors["frame_transformer"].data.source_pos_w
        source_quat_w_tf = scene.sensors["frame_transformer"].data.source_quat_w
        feet_pos_w_tf = scene.sensors["frame_transformer"].data.target_pos_w
        feet_quat_w_tf = scene.sensors["frame_transformer"].data.target_quat_w

        torch.testing.assert_close(source_pose_w_gt[:, :3], source_pos_w_tf)
        torch.testing.assert_close(source_pose_w_gt[:, 3:], source_quat_w_tf)
        torch.testing.assert_close(feet_pos_w_gt, feet_pos_w_tf)
        torch.testing.assert_close(feet_quat_w_gt, feet_quat_w_tf)


        feet_pos_source_tf = scene.sensors["frame_transformer"].data.target_pos_source
        feet_quat_source_tf = scene.sensors["frame_transformer"].data.target_quat_source
        for index in range(len(feet_indices)):

            foot_pos_b, foot_quat_b = math_utils.subtract_frame_transforms(
                source_pose_w_gt[:, :3], source_pose_w_gt[:, 3:], feet_pos_w_tf[:, index], feet_quat_w_tf[:, index]
            )

            torch.testing.assert_close(feet_pos_source_tf[:, index], foot_pos_b)
            torch.testing.assert_close(feet_quat_source_tf[:, index], foot_quat_b)


def test_frame_transformer_robot_body_to_external_cube(sim):
    """Test transformation from robot body to a cube in the scene."""

    scene_cfg = MySceneCfg(num_envs=2, env_spacing=5.0, lazy_sensor_update=False)
    scene_cfg.frame_transformer = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                name="CUBE_USER",
                prim_path="{ENV_REGEX_NS}/cube",
            ),
        ],
    )
    scene = InteractiveScene(scene_cfg)


    sim.reset()


    default_actions = scene.articulations["robot"].data.default_joint_pos.clone()

    sim_dt = sim.get_physics_dt()

    for count in range(100):

        if count % 25 == 0:

            root_state = scene.articulations["robot"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            joint_pos = scene.articulations["robot"].data.default_joint_pos
            joint_vel = scene.articulations["robot"].data.default_joint_vel


            scene.articulations["robot"].write_root_pose_to_sim(root_state[:, :7])
            scene.articulations["robot"].write_root_velocity_to_sim(root_state[:, 7:])
            scene.articulations["robot"].write_joint_state_to_sim(joint_pos, joint_vel)

            scene.reset()


        robot_actions = default_actions + 0.5 * torch.randn_like(default_actions)
        scene.articulations["robot"].set_joint_position_target(robot_actions)

        scene.write_data_to_sim()

        sim.step()

        scene.update(sim_dt)



        root_pose_w = scene.articulations["robot"].data.root_pose_w
        cube_pos_w_gt = scene.rigid_objects["cube"].data.root_pos_w
        cube_quat_w_gt = scene.rigid_objects["cube"].data.root_quat_w

        source_pos_w_tf = scene.sensors["frame_transformer"].data.source_pos_w
        source_quat_w_tf = scene.sensors["frame_transformer"].data.source_quat_w
        cube_pos_w_tf = scene.sensors["frame_transformer"].data.target_pos_w.squeeze()
        cube_quat_w_tf = scene.sensors["frame_transformer"].data.target_quat_w.squeeze()


        torch.testing.assert_close(root_pose_w[:, :3], source_pos_w_tf)
        torch.testing.assert_close(root_pose_w[:, 3:], source_quat_w_tf)
        torch.testing.assert_close(cube_pos_w_gt, cube_pos_w_tf)
        torch.testing.assert_close(cube_quat_w_gt, cube_quat_w_tf)


        cube_pos_source_tf = scene.sensors["frame_transformer"].data.target_pos_source
        cube_quat_source_tf = scene.sensors["frame_transformer"].data.target_quat_source

        cube_pos_b, cube_quat_b = math_utils.subtract_frame_transforms(
            root_pose_w[:, :3], root_pose_w[:, 3:], cube_pos_w_tf, cube_quat_w_tf
        )

        torch.testing.assert_close(cube_pos_source_tf[:, 0], cube_pos_b)
        torch.testing.assert_close(cube_quat_source_tf[:, 0], cube_quat_b)


@pytest.mark.isaacsim_ci
def test_frame_transformer_offset_frames(sim):
    """Test body transformation w.r.t. base source frame.

    In this test, the source frame is the cube frame.
    """

    scene_cfg = MySceneCfg(num_envs=2, env_spacing=5.0, lazy_sensor_update=False)
    scene_cfg.frame_transformer = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/cube",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                name="CUBE_CENTER",
                prim_path="{ENV_REGEX_NS}/cube",
            ),
            FrameTransformerCfg.FrameCfg(
                name="CUBE_TOP",
                prim_path="{ENV_REGEX_NS}/cube",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.1),
                    rot=(1.0, 0.0, 0.0, 0.0),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                name="CUBE_BOTTOM",
                prim_path="{ENV_REGEX_NS}/cube",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, -0.1),
                    rot=(1.0, 0.0, 0.0, 0.0),
                ),
            ),
        ],
    )
    scene = InteractiveScene(scene_cfg)


    sim.reset()


    sim_dt = sim.get_physics_dt()

    for count in range(100):

        if count % 25 == 0:

            root_state = scene["cube"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins


            scene["cube"].write_root_pose_to_sim(root_state[:, :7])
            scene["cube"].write_root_velocity_to_sim(root_state[:, 7:])

            scene.reset()


        scene.write_data_to_sim()

        sim.step()

        scene.update(sim_dt)



        cube_pos_w_gt = scene["cube"].data.root_pos_w
        cube_quat_w_gt = scene["cube"].data.root_quat_w

        source_pos_w_tf = scene.sensors["frame_transformer"].data.source_pos_w
        source_quat_w_tf = scene.sensors["frame_transformer"].data.source_quat_w
        target_pos_w_tf = scene.sensors["frame_transformer"].data.target_pos_w.squeeze()
        target_quat_w_tf = scene.sensors["frame_transformer"].data.target_quat_w.squeeze()
        target_frame_names = scene.sensors["frame_transformer"].data.target_frame_names

        cube_center_idx = target_frame_names.index("CUBE_CENTER")
        cube_bottom_idx = target_frame_names.index("CUBE_BOTTOM")
        cube_top_idx = target_frame_names.index("CUBE_TOP")


        torch.testing.assert_close(cube_pos_w_gt, source_pos_w_tf)
        torch.testing.assert_close(cube_quat_w_gt, source_quat_w_tf)
        torch.testing.assert_close(cube_pos_w_gt, target_pos_w_tf[:, cube_center_idx])
        torch.testing.assert_close(cube_quat_w_gt, target_quat_w_tf[:, cube_center_idx])



        cube_pos_top = target_pos_w_tf[:, cube_top_idx]
        cube_quat_top = target_quat_w_tf[:, cube_top_idx]
        torch.testing.assert_close(cube_pos_top, cube_pos_w_gt + torch.tensor([0.0, 0.0, 0.1]))
        torch.testing.assert_close(cube_quat_top, cube_quat_w_gt)


        cube_pos_bottom = target_pos_w_tf[:, cube_bottom_idx]
        cube_quat_bottom = target_quat_w_tf[:, cube_bottom_idx]
        torch.testing.assert_close(cube_pos_bottom, cube_pos_w_gt + torch.tensor([0.0, 0.0, -0.1]))
        torch.testing.assert_close(cube_quat_bottom, cube_quat_w_gt)


@pytest.mark.isaacsim_ci
def test_frame_transformer_all_bodies(sim):
    """Test transformation of all bodies w.r.t. base source frame.

    In this test, the source frame is the robot base.

    The target_frames are all bodies in the robot, implemented using .* pattern.
    """

    scene_cfg = MySceneCfg(num_envs=2, env_spacing=5.0, lazy_sensor_update=False)
    scene_cfg.frame_transformer = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/.*",
            ),
        ],
    )
    scene = InteractiveScene(scene_cfg)


    sim.reset()

    target_frame_names = scene.sensors["frame_transformer"].data.target_frame_names
    articulation_body_names = scene.articulations["robot"].data.body_names

    reordering_indices = [target_frame_names.index(name) for name in articulation_body_names]


    default_actions = scene.articulations["robot"].data.default_joint_pos.clone()

    sim_dt = sim.get_physics_dt()

    for count in range(100):

        if count % 25 == 0:

            root_state = scene.articulations["robot"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            joint_pos = scene.articulations["robot"].data.default_joint_pos
            joint_vel = scene.articulations["robot"].data.default_joint_vel


            scene.articulations["robot"].write_root_pose_to_sim(root_state[:, :7])
            scene.articulations["robot"].write_root_velocity_to_sim(root_state[:, 7:])
            scene.articulations["robot"].write_joint_state_to_sim(joint_pos, joint_vel)

            scene.reset()


        robot_actions = default_actions + 0.5 * torch.randn_like(default_actions)
        scene.articulations["robot"].set_joint_position_target(robot_actions)

        scene.write_data_to_sim()

        sim.step()

        scene.update(sim_dt)



        root_pose_w = scene.articulations["robot"].data.root_pose_w
        bodies_pos_w_gt = scene.articulations["robot"].data.body_pos_w
        bodies_quat_w_gt = scene.articulations["robot"].data.body_quat_w


        source_pos_w_tf = scene.sensors["frame_transformer"].data.source_pos_w
        source_quat_w_tf = scene.sensors["frame_transformer"].data.source_quat_w
        bodies_pos_w_tf = scene.sensors["frame_transformer"].data.target_pos_w
        bodies_quat_w_tf = scene.sensors["frame_transformer"].data.target_quat_w


        torch.testing.assert_close(root_pose_w[:, :3], source_pos_w_tf)
        torch.testing.assert_close(root_pose_w[:, 3:], source_quat_w_tf)
        torch.testing.assert_close(bodies_pos_w_gt, bodies_pos_w_tf[:, reordering_indices])
        torch.testing.assert_close(bodies_quat_w_gt, bodies_quat_w_tf[:, reordering_indices])

        bodies_pos_source_tf = scene.sensors["frame_transformer"].data.target_pos_source
        bodies_quat_source_tf = scene.sensors["frame_transformer"].data.target_quat_source


        for index in range(len(articulation_body_names)):
            body_pos_b, body_quat_b = math_utils.subtract_frame_transforms(
                root_pose_w[:, :3], root_pose_w[:, 3:], bodies_pos_w_tf[:, index], bodies_quat_w_tf[:, index]
            )

            torch.testing.assert_close(bodies_pos_source_tf[:, index], body_pos_b)
            torch.testing.assert_close(bodies_quat_source_tf[:, index], body_quat_b)


@pytest.mark.isaacsim_ci
def test_sensor_print(sim):
    """Test sensor print is working correctly."""

    scene_cfg = MySceneCfg(num_envs=2, env_spacing=5.0, lazy_sensor_update=False)
    scene_cfg.frame_transformer = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/.*",
            ),
        ],
    )
    scene = InteractiveScene(scene_cfg)


    sim.reset()

    print(scene.sensors["frame_transformer"])
