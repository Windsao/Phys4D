







"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher


simulation_app = AppLauncher(headless=True, enable_cameras=True).app

"""Rest everything follows."""

import copy
import numpy as np
import os
import random
import scipy.spatial.transform as tf
import torch

import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
import omni.replicator.core as rep
import pytest
from isaacsim.core.prims import SingleGeometryPrim, SingleRigidPrim
from pxr import Gf, Usd, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.utils import convert_dict_to_backend
from isaaclab.utils.math import convert_quat
from isaaclab.utils.timer import Timer


POSITION = (2.5, 2.5, 2.5)
QUAT_ROS = (-0.17591989, 0.33985114, 0.82047325, -0.42470819)
QUAT_OPENGL = (0.33985113, 0.17591988, 0.42470818, 0.82047324)
QUAT_WORLD = (-0.3647052, -0.27984815, -0.1159169, 0.88047623)




HEIGHT = 240
WIDTH = 320


def setup() -> tuple[sim_utils.SimulationContext, CameraCfg, float]:
    camera_cfg = CameraCfg(
        height=HEIGHT,
        width=WIDTH,
        prim_path="/World/Camera",
        update_period=0,
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
    )

    stage_utils.create_new_stage()

    dt = 0.01

    sim_cfg = sim_utils.SimulationCfg(dt=dt)
    sim = sim_utils.SimulationContext(sim_cfg)

    _populate_scene()

    stage_utils.update_stage()
    return sim, camera_cfg, dt


def teardown(sim: sim_utils.SimulationContext):


    rep.vp_manager.destroy_hydra_textures("Replicator")


    sim._timeline.stop()

    sim.clear_all_callbacks()
    sim.clear_instance()


@pytest.fixture
def setup_sim_camera():
    """Create a simulation context."""
    sim, camera_cfg, dt = setup()
    yield sim, camera_cfg, dt
    teardown(sim)


def test_camera_init(setup_sim_camera):
    """Test camera initialization."""

    sim, camera_cfg, dt = setup_sim_camera

    camera = Camera(camera_cfg)

    assert sim.has_rtx_sensors()

    sim.reset()

    assert camera.is_initialized

    assert camera._sensor_prims[0].GetPath().pathString == camera_cfg.prim_path
    assert isinstance(camera._sensor_prims[0], UsdGeom.Camera)




    for _ in range(5):
        sim.step()


    assert camera.data.pos_w.shape == (1, 3)
    assert camera.data.quat_w_ros.shape == (1, 4)
    assert camera.data.quat_w_world.shape == (1, 4)
    assert camera.data.quat_w_opengl.shape == (1, 4)
    assert camera.data.intrinsic_matrices.shape == (1, 3, 3)
    assert camera.data.image_shape == (camera_cfg.height, camera_cfg.width)
    assert camera.data.info == [{camera_cfg.data_types[0]: None}]


    for _ in range(10):

        sim.step()

        camera.update(sim.cfg.dt)

        for im_data in camera.data.output.values():
            assert im_data.shape == (1, camera_cfg.height, camera_cfg.width, 1)


def test_camera_init_offset(setup_sim_camera):
    """Test camera initialization with offset using different conventions."""
    sim, camera_cfg, dt = setup_sim_camera


    cam_cfg_offset_ros = copy.deepcopy(camera_cfg)
    cam_cfg_offset_ros.update_latest_camera_pose = True
    cam_cfg_offset_ros.offset = CameraCfg.OffsetCfg(
        pos=POSITION,
        rot=QUAT_ROS,
        convention="ros",
    )
    cam_cfg_offset_ros.prim_path = "/World/CameraOffsetRos"
    camera_ros = Camera(cam_cfg_offset_ros)

    cam_cfg_offset_opengl = copy.deepcopy(camera_cfg)
    cam_cfg_offset_opengl.update_latest_camera_pose = True
    cam_cfg_offset_opengl.offset = CameraCfg.OffsetCfg(
        pos=POSITION,
        rot=QUAT_OPENGL,
        convention="opengl",
    )
    cam_cfg_offset_opengl.prim_path = "/World/CameraOffsetOpengl"
    camera_opengl = Camera(cam_cfg_offset_opengl)

    cam_cfg_offset_world = copy.deepcopy(camera_cfg)
    cam_cfg_offset_world.update_latest_camera_pose = True
    cam_cfg_offset_world.offset = CameraCfg.OffsetCfg(
        pos=POSITION,
        rot=QUAT_WORLD,
        convention="world",
    )
    cam_cfg_offset_world.prim_path = "/World/CameraOffsetWorld"
    camera_world = Camera(cam_cfg_offset_world)


    sim.reset()


    prim_tf_ros = camera_ros._sensor_prims[0].ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    prim_tf_opengl = camera_opengl._sensor_prims[0].ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    prim_tf_world = camera_world._sensor_prims[0].ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    prim_tf_ros = np.transpose(prim_tf_ros)
    prim_tf_opengl = np.transpose(prim_tf_opengl)
    prim_tf_world = np.transpose(prim_tf_world)


    np.testing.assert_allclose(prim_tf_ros[0:3, 3], cam_cfg_offset_ros.offset.pos)
    np.testing.assert_allclose(prim_tf_opengl[0:3, 3], cam_cfg_offset_opengl.offset.pos)
    np.testing.assert_allclose(prim_tf_world[0:3, 3], cam_cfg_offset_world.offset.pos)
    np.testing.assert_allclose(
        convert_quat(tf.Rotation.from_matrix(prim_tf_ros[:3, :3]).as_quat(), "wxyz"),
        cam_cfg_offset_opengl.offset.rot,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        convert_quat(tf.Rotation.from_matrix(prim_tf_opengl[:3, :3]).as_quat(), "wxyz"),
        cam_cfg_offset_opengl.offset.rot,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        convert_quat(tf.Rotation.from_matrix(prim_tf_world[:3, :3]).as_quat(), "wxyz"),
        cam_cfg_offset_opengl.offset.rot,
        rtol=1e-5,
    )




    for _ in range(5):
        sim.step()


    np.testing.assert_allclose(camera_ros.data.pos_w[0].cpu().numpy(), cam_cfg_offset_ros.offset.pos, rtol=1e-5)
    np.testing.assert_allclose(camera_ros.data.quat_w_ros[0].cpu().numpy(), QUAT_ROS, rtol=1e-5)
    np.testing.assert_allclose(camera_ros.data.quat_w_opengl[0].cpu().numpy(), QUAT_OPENGL, rtol=1e-5)
    np.testing.assert_allclose(camera_ros.data.quat_w_world[0].cpu().numpy(), QUAT_WORLD, rtol=1e-5)


def test_multi_camera_init(setup_sim_camera):
    """Test multi-camera initialization."""
    sim, camera_cfg, dt = setup_sim_camera


    cam_cfg_1 = copy.deepcopy(camera_cfg)
    cam_cfg_1.prim_path = "/World/Camera_1"
    cam_1 = Camera(cam_cfg_1)

    cam_cfg_2 = copy.deepcopy(camera_cfg)
    cam_cfg_2.prim_path = "/World/Camera_2"
    cam_2 = Camera(cam_cfg_2)


    sim.reset()




    for _ in range(5):
        sim.step()

    for _ in range(10):

        sim.step()

        cam_1.update(dt)
        cam_2.update(dt)

        for cam in [cam_1, cam_2]:
            for im_data in cam.data.output.values():
                assert im_data.shape == (1, camera_cfg.height, camera_cfg.width, 1)


def test_multi_camera_with_different_resolution(setup_sim_camera):
    """Test multi-camera initialization with cameras having different image resolutions."""
    sim, camera_cfg, dt = setup_sim_camera


    cam_cfg_1 = copy.deepcopy(camera_cfg)
    cam_cfg_1.prim_path = "/World/Camera_1"
    cam_1 = Camera(cam_cfg_1)

    cam_cfg_2 = copy.deepcopy(camera_cfg)
    cam_cfg_2.prim_path = "/World/Camera_2"
    cam_cfg_2.height = 240
    cam_cfg_2.width = 320
    cam_2 = Camera(cam_cfg_2)


    sim.reset()




    for _ in range(5):
        sim.step()

    sim.step()

    cam_1.update(dt)
    cam_2.update(dt)

    assert cam_1.data.output["distance_to_image_plane"].shape == (1, camera_cfg.height, camera_cfg.width, 1)
    assert cam_2.data.output["distance_to_image_plane"].shape == (1, cam_cfg_2.height, cam_cfg_2.width, 1)


def test_camera_init_intrinsic_matrix(setup_sim_camera):
    """Test camera initialization from intrinsic matrix."""
    sim, camera_cfg, dt = setup_sim_camera

    camera_1 = Camera(cfg=camera_cfg)

    sim.reset()
    intrinsic_matrix = camera_1.data.intrinsic_matrices[0].cpu().flatten().tolist()
    teardown(sim)

    sim, camera_cfg, dt = setup()
    camera_1 = Camera(cfg=camera_cfg)

    intrinsic_camera_cfg = CameraCfg(
        height=HEIGHT,
        width=WIDTH,
        prim_path="/World/Camera_2",
        update_period=0,
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=intrinsic_matrix,
            width=WIDTH,
            height=HEIGHT,
            focal_length=24.0,
            focus_distance=400.0,
            clipping_range=(0.1, 1.0e5),
        ),
    )
    camera_2 = Camera(cfg=intrinsic_camera_cfg)


    sim.reset()


    camera_1.update(dt)
    camera_2.update(dt)


    torch.testing.assert_close(
        camera_1.data.output["distance_to_image_plane"],
        camera_2.data.output["distance_to_image_plane"],
        rtol=5e-3,
        atol=1e-4,
    )

    torch.testing.assert_close(
        camera_1.data.intrinsic_matrices[0],
        camera_2.data.intrinsic_matrices[0],
        rtol=5e-3,
        atol=1e-4,
    )


def test_camera_set_world_poses(setup_sim_camera):
    """Test camera function to set specific world pose."""
    sim, camera_cfg, dt = setup_sim_camera

    camera_cfg.update_latest_camera_pose = True

    camera = Camera(camera_cfg)

    sim.reset()


    position = torch.tensor([POSITION], dtype=torch.float32, device=camera.device)
    orientation = torch.tensor([QUAT_WORLD], dtype=torch.float32, device=camera.device)

    camera.set_world_poses(position.clone(), orientation.clone(), convention="world")




    for _ in range(5):
        sim.step()


    torch.testing.assert_close(camera.data.pos_w, position)
    torch.testing.assert_close(camera.data.quat_w_world, orientation)


def test_camera_set_world_poses_from_view(setup_sim_camera):
    """Test camera function to set specific world pose from view."""
    sim, camera_cfg, dt = setup_sim_camera

    camera_cfg.update_latest_camera_pose = True

    camera = Camera(camera_cfg)

    sim.reset()


    eyes = torch.tensor([POSITION], dtype=torch.float32, device=camera.device)
    targets = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32, device=camera.device)
    quat_ros_gt = torch.tensor([QUAT_ROS], dtype=torch.float32, device=camera.device)

    camera.set_world_poses_from_view(eyes.clone(), targets.clone())




    for _ in range(5):
        sim.step()


    torch.testing.assert_close(camera.data.pos_w, eyes)
    torch.testing.assert_close(camera.data.quat_w_ros, quat_ros_gt)


def test_intrinsic_matrix(setup_sim_camera):
    """Checks that the camera's set and retrieve methods work for intrinsic matrix."""
    sim, camera_cfg, dt = setup_sim_camera

    camera_cfg.update_latest_camera_pose = True

    camera = Camera(camera_cfg)

    sim.reset()

    rs_intrinsic_matrix = [229.8, 0.0, 160.0, 0.0, 229.8, 120.0, 0.0, 0.0, 1.0]
    rs_intrinsic_matrix = torch.tensor(rs_intrinsic_matrix, device=camera.device).reshape(3, 3).unsqueeze(0)

    camera.set_intrinsic_matrices(rs_intrinsic_matrix.clone())




    for _ in range(5):
        sim.step()


    for _ in range(10):

        sim.step()

        camera.update(dt)

        torch.testing.assert_close(rs_intrinsic_matrix[0, 0, 0], camera.data.intrinsic_matrices[0, 0, 0])
        torch.testing.assert_close(rs_intrinsic_matrix[0, 1, 1], camera.data.intrinsic_matrices[0, 1, 1])
        torch.testing.assert_close(rs_intrinsic_matrix[0, 0, 2], camera.data.intrinsic_matrices[0, 0, 2])
        torch.testing.assert_close(rs_intrinsic_matrix[0, 1, 2], camera.data.intrinsic_matrices[0, 1, 2])


def test_depth_clipping(setup_sim_camera):
    """Test depth clipping.

    .. note::

        This test is the same for all camera models to enforce the same clipping behavior.
    """

    sim, _, dt = setup_sim_camera
    camera_cfg_zero = CameraCfg(
        prim_path="/World/CameraZero",
        offset=CameraCfg.OffsetCfg(pos=(2.5, 2.5, 6.0), rot=(-0.125, 0.362, 0.873, -0.302), convention="ros"),
        spawn=sim_utils.PinholeCameraCfg().from_intrinsic_matrix(
            focal_length=38.0,
            intrinsic_matrix=[380.08, 0.0, 467.79, 0.0, 380.08, 262.05, 0.0, 0.0, 1.0],
            height=540,
            width=960,
            clipping_range=(0.1, 10),
        ),
        height=540,
        width=960,
        data_types=["distance_to_image_plane", "distance_to_camera"],
        depth_clipping_behavior="zero",
    )
    camera_zero = Camera(camera_cfg_zero)

    camera_cfg_none = copy.deepcopy(camera_cfg_zero)
    camera_cfg_none.prim_path = "/World/CameraNone"
    camera_cfg_none.depth_clipping_behavior = "none"
    camera_none = Camera(camera_cfg_none)

    camera_cfg_max = copy.deepcopy(camera_cfg_zero)
    camera_cfg_max.prim_path = "/World/CameraMax"
    camera_cfg_max.depth_clipping_behavior = "max"
    camera_max = Camera(camera_cfg_max)


    sim.reset()



    for _ in range(5):
        sim.step()

    camera_zero.update(dt)
    camera_none.update(dt)
    camera_max.update(dt)


    assert torch.isinf(camera_none.data.output["distance_to_camera"]).any()
    assert torch.isinf(camera_none.data.output["distance_to_image_plane"]).any()
    assert (
        camera_none.data.output["distance_to_camera"][~torch.isinf(camera_none.data.output["distance_to_camera"])].min()
        >= camera_cfg_zero.spawn.clipping_range[0]
    )
    assert (
        camera_none.data.output["distance_to_camera"][~torch.isinf(camera_none.data.output["distance_to_camera"])].max()
        <= camera_cfg_zero.spawn.clipping_range[1]
    )
    assert (
        camera_none.data.output["distance_to_image_plane"][
            ~torch.isinf(camera_none.data.output["distance_to_image_plane"])
        ].min()
        >= camera_cfg_zero.spawn.clipping_range[0]
    )
    assert (
        camera_none.data.output["distance_to_image_plane"][
            ~torch.isinf(camera_none.data.output["distance_to_camera"])
        ].max()
        <= camera_cfg_zero.spawn.clipping_range[1]
    )


    assert torch.all(
        camera_zero.data.output["distance_to_camera"][torch.isinf(camera_none.data.output["distance_to_camera"])] == 0.0
    )
    assert torch.all(
        camera_zero.data.output["distance_to_image_plane"][
            torch.isinf(camera_none.data.output["distance_to_image_plane"])
        ]
        == 0.0
    )
    assert (
        camera_zero.data.output["distance_to_camera"][camera_zero.data.output["distance_to_camera"] != 0.0].min()
        >= camera_cfg_zero.spawn.clipping_range[0]
    )
    assert camera_zero.data.output["distance_to_camera"].max() <= camera_cfg_zero.spawn.clipping_range[1]
    assert (
        camera_zero.data.output["distance_to_image_plane"][
            camera_zero.data.output["distance_to_image_plane"] != 0.0
        ].min()
        >= camera_cfg_zero.spawn.clipping_range[0]
    )
    assert camera_zero.data.output["distance_to_image_plane"].max() <= camera_cfg_zero.spawn.clipping_range[1]


    assert torch.all(
        camera_max.data.output["distance_to_camera"][torch.isinf(camera_none.data.output["distance_to_camera"])]
        == camera_cfg_zero.spawn.clipping_range[1]
    )
    assert torch.all(
        camera_max.data.output["distance_to_image_plane"][
            torch.isinf(camera_none.data.output["distance_to_image_plane"])
        ]
        == camera_cfg_zero.spawn.clipping_range[1]
    )
    assert camera_max.data.output["distance_to_camera"].min() >= camera_cfg_zero.spawn.clipping_range[0]
    assert camera_max.data.output["distance_to_camera"].max() <= camera_cfg_zero.spawn.clipping_range[1]
    assert camera_max.data.output["distance_to_image_plane"].min() >= camera_cfg_zero.spawn.clipping_range[0]
    assert camera_max.data.output["distance_to_image_plane"].max() <= camera_cfg_zero.spawn.clipping_range[1]


def test_camera_resolution_all_colorize(setup_sim_camera):
    """Test camera resolution is correctly set for all types with colorization enabled."""

    sim, camera_cfg, dt = setup_sim_camera
    camera_cfg.data_types = [
        "rgb",
        "rgba",
        "depth",
        "distance_to_camera",
        "distance_to_image_plane",
        "normals",
        "motion_vectors",
        "semantic_segmentation",
        "instance_segmentation_fast",
        "instance_id_segmentation_fast",
    ]
    camera_cfg.colorize_instance_id_segmentation = True
    camera_cfg.colorize_instance_segmentation = True
    camera_cfg.colorize_semantic_segmentation = True

    camera = Camera(camera_cfg)


    sim.reset()




    for _ in range(5):
        sim.step()
    camera.update(dt)


    hw_1c_shape = (1, camera_cfg.height, camera_cfg.width, 1)
    hw_2c_shape = (1, camera_cfg.height, camera_cfg.width, 2)
    hw_3c_shape = (1, camera_cfg.height, camera_cfg.width, 3)
    hw_4c_shape = (1, camera_cfg.height, camera_cfg.width, 4)

    output = camera.data.output
    assert output["rgb"].shape == hw_3c_shape
    assert output["rgba"].shape == hw_4c_shape
    assert output["depth"].shape == hw_1c_shape
    assert output["distance_to_camera"].shape == hw_1c_shape
    assert output["distance_to_image_plane"].shape == hw_1c_shape
    assert output["normals"].shape == hw_3c_shape
    assert output["motion_vectors"].shape == hw_2c_shape
    assert output["semantic_segmentation"].shape == hw_4c_shape
    assert output["instance_segmentation_fast"].shape == hw_4c_shape
    assert output["instance_id_segmentation_fast"].shape == hw_4c_shape


    output = camera.data.output
    assert output["rgb"].dtype == torch.uint8
    assert output["rgba"].dtype == torch.uint8
    assert output["depth"].dtype == torch.float
    assert output["distance_to_camera"].dtype == torch.float
    assert output["distance_to_image_plane"].dtype == torch.float
    assert output["normals"].dtype == torch.float
    assert output["motion_vectors"].dtype == torch.float
    assert output["semantic_segmentation"].dtype == torch.uint8
    assert output["instance_segmentation_fast"].dtype == torch.uint8
    assert output["instance_id_segmentation_fast"].dtype == torch.uint8


def test_camera_resolution_no_colorize(setup_sim_camera):
    """Test camera resolution is correctly set for all types with no colorization enabled."""

    sim, camera_cfg, dt = setup_sim_camera
    camera_cfg.data_types = [
        "rgb",
        "rgba",
        "depth",
        "distance_to_camera",
        "distance_to_image_plane",
        "normals",
        "motion_vectors",
        "semantic_segmentation",
        "instance_segmentation_fast",
        "instance_id_segmentation_fast",
    ]
    camera_cfg.colorize_instance_id_segmentation = False
    camera_cfg.colorize_instance_segmentation = False
    camera_cfg.colorize_semantic_segmentation = False

    camera = Camera(camera_cfg)


    sim.reset()



    for _ in range(12):
        sim.step()
    camera.update(dt)


    hw_1c_shape = (1, camera_cfg.height, camera_cfg.width, 1)
    hw_2c_shape = (1, camera_cfg.height, camera_cfg.width, 2)
    hw_3c_shape = (1, camera_cfg.height, camera_cfg.width, 3)
    hw_4c_shape = (1, camera_cfg.height, camera_cfg.width, 4)

    output = camera.data.output
    assert output["rgb"].shape == hw_3c_shape
    assert output["rgba"].shape == hw_4c_shape
    assert output["depth"].shape == hw_1c_shape
    assert output["distance_to_camera"].shape == hw_1c_shape
    assert output["distance_to_image_plane"].shape == hw_1c_shape
    assert output["normals"].shape == hw_3c_shape
    assert output["motion_vectors"].shape == hw_2c_shape
    assert output["semantic_segmentation"].shape == hw_1c_shape
    assert output["instance_segmentation_fast"].shape == hw_1c_shape
    assert output["instance_id_segmentation_fast"].shape == hw_1c_shape


    output = camera.data.output
    assert output["rgb"].dtype == torch.uint8
    assert output["rgba"].dtype == torch.uint8
    assert output["depth"].dtype == torch.float
    assert output["distance_to_camera"].dtype == torch.float
    assert output["distance_to_image_plane"].dtype == torch.float
    assert output["normals"].dtype == torch.float
    assert output["motion_vectors"].dtype == torch.float
    assert output["semantic_segmentation"].dtype == torch.int32
    assert output["instance_segmentation_fast"].dtype == torch.int32
    assert output["instance_id_segmentation_fast"].dtype == torch.int32


def test_camera_large_resolution_all_colorize(setup_sim_camera):
    """Test camera resolution is correctly set for all types with colorization enabled."""

    sim, camera_cfg, dt = setup_sim_camera
    camera_cfg.data_types = [
        "rgb",
        "rgba",
        "depth",
        "distance_to_camera",
        "distance_to_image_plane",
        "normals",
        "motion_vectors",
        "semantic_segmentation",
        "instance_segmentation_fast",
        "instance_id_segmentation_fast",
    ]
    camera_cfg.colorize_instance_id_segmentation = True
    camera_cfg.colorize_instance_segmentation = True
    camera_cfg.colorize_semantic_segmentation = True
    camera_cfg.width = 512
    camera_cfg.height = 512

    camera = Camera(camera_cfg)


    sim.reset()




    for _ in range(5):
        sim.step()
    camera.update(dt)


    hw_1c_shape = (1, camera_cfg.height, camera_cfg.width, 1)
    hw_2c_shape = (1, camera_cfg.height, camera_cfg.width, 2)
    hw_3c_shape = (1, camera_cfg.height, camera_cfg.width, 3)
    hw_4c_shape = (1, camera_cfg.height, camera_cfg.width, 4)

    output = camera.data.output
    assert output["rgb"].shape == hw_3c_shape
    assert output["rgba"].shape == hw_4c_shape
    assert output["depth"].shape == hw_1c_shape
    assert output["distance_to_camera"].shape == hw_1c_shape
    assert output["distance_to_image_plane"].shape == hw_1c_shape
    assert output["normals"].shape == hw_3c_shape
    assert output["motion_vectors"].shape == hw_2c_shape
    assert output["semantic_segmentation"].shape == hw_4c_shape
    assert output["instance_segmentation_fast"].shape == hw_4c_shape
    assert output["instance_id_segmentation_fast"].shape == hw_4c_shape


    output = camera.data.output
    assert output["rgb"].dtype == torch.uint8
    assert output["rgba"].dtype == torch.uint8
    assert output["depth"].dtype == torch.float
    assert output["distance_to_camera"].dtype == torch.float
    assert output["distance_to_image_plane"].dtype == torch.float
    assert output["normals"].dtype == torch.float
    assert output["motion_vectors"].dtype == torch.float
    assert output["semantic_segmentation"].dtype == torch.uint8
    assert output["instance_segmentation_fast"].dtype == torch.uint8
    assert output["instance_id_segmentation_fast"].dtype == torch.uint8


def test_camera_resolution_rgb_only(setup_sim_camera):
    """Test camera resolution is correctly set for RGB only."""

    sim, camera_cfg, dt = setup_sim_camera
    camera_cfg.data_types = ["rgb"]

    camera = Camera(camera_cfg)


    sim.reset()




    for _ in range(5):
        sim.step()
    camera.update(dt)


    hw_3c_shape = (1, camera_cfg.height, camera_cfg.width, 3)

    output = camera.data.output
    assert output["rgb"].shape == hw_3c_shape

    assert output["rgb"].dtype == torch.uint8


def test_camera_resolution_rgba_only(setup_sim_camera):
    """Test camera resolution is correctly set for RGBA only."""

    sim, camera_cfg, dt = setup_sim_camera
    camera_cfg.data_types = ["rgba"]

    camera = Camera(camera_cfg)


    sim.reset()




    for _ in range(5):
        sim.step()
    camera.update(dt)


    hw_4c_shape = (1, camera_cfg.height, camera_cfg.width, 4)

    output = camera.data.output
    assert output["rgba"].shape == hw_4c_shape

    assert output["rgba"].dtype == torch.uint8


def test_camera_resolution_depth_only(setup_sim_camera):
    """Test camera resolution is correctly set for depth only."""

    sim, camera_cfg, dt = setup_sim_camera
    camera_cfg.data_types = ["depth"]

    camera = Camera(camera_cfg)


    sim.reset()




    for _ in range(5):
        sim.step()
    camera.update(dt)


    hw_1c_shape = (1, camera_cfg.height, camera_cfg.width, 1)

    output = camera.data.output
    assert output["depth"].shape == hw_1c_shape

    assert output["depth"].dtype == torch.float


def test_throughput(setup_sim_camera):
    """Checks that the single camera gets created properly with a rig."""

    file_dir = os.path.dirname(os.path.realpath(__file__))
    temp_dir = os.path.join(file_dir, "output", "camera", "throughput")
    os.makedirs(temp_dir, exist_ok=True)

    rep_writer = rep.BasicWriter(output_dir=temp_dir, frame_padding=3)

    sim, camera_cfg, dt = setup_sim_camera
    camera_cfg.height = 480
    camera_cfg.width = 640
    camera = Camera(camera_cfg)


    sim.reset()


    eyes = torch.tensor([[2.5, 2.5, 2.5]], dtype=torch.float32, device=camera.device)
    targets = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32, device=camera.device)
    camera.set_world_poses_from_view(eyes, targets)




    for _ in range(5):
        sim.step()

    for _ in range(5):

        sim.step()

        with Timer(f"Time taken for updating camera with shape {camera.image_shape}"):
            camera.update(dt)

        with Timer(f"Time taken for writing data with shape {camera.image_shape}   "):

            rep_output = {"annotators": {}}
            camera_data = convert_dict_to_backend({k: v[0] for k, v in camera.data.output.items()}, backend="numpy")
            for key, data, info in zip(camera_data.keys(), camera_data.values(), camera.data.info[0].values()):
                if info is not None:
                    rep_output["annotators"][key] = {"render_product": {"data": data, **info}}
                else:
                    rep_output["annotators"][key] = {"render_product": {"data": data}}

            rep_output["trigger_outputs"] = {"on_time": camera.frame[0]}
            rep_writer.write(rep_output)
        print("----------------------------------------")

        for im_data in camera.data.output.values():
            assert im_data.shape == (1, camera_cfg.height, camera_cfg.width, 1)


def test_sensor_print(setup_sim_camera):
    """Test sensor print is working correctly."""

    sim, camera_cfg, dt = setup_sim_camera
    sensor = Camera(cfg=camera_cfg)

    sim.reset()

    print(sensor)


def _populate_scene():
    """Add prims to the scene."""

    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    cfg = sim_utils.SphereLightCfg()
    cfg.func("/World/Light/GreySphere", cfg, translation=(4.5, 3.5, 10.0))
    cfg.func("/World/Light/WhiteSphere", cfg, translation=(-4.5, 3.5, 10.0))

    random.seed(0)
    for i in range(10):

        position = np.random.rand(3) - np.asarray([0.05, 0.05, -1.0])
        position *= np.asarray([1.5, 1.5, 0.5])

        prim_type = random.choice(["Cube", "Sphere", "Cylinder"])
        prim = prim_utils.create_prim(
            f"/World/Objects/Obj_{i:02d}",
            prim_type,
            translation=position,
            scale=(0.25, 0.25, 0.25),
            semantic_label=prim_type,
        )

        geom_prim = getattr(UsdGeom, prim_type)(prim)

        color = Gf.Vec3f(random.random(), random.random(), random.random())
        geom_prim.CreateDisplayColorAttr()
        geom_prim.GetDisplayColorAttr().Set([color])

        SingleGeometryPrim(f"/World/Objects/Obj_{i:02d}", collision=True)
        SingleRigidPrim(f"/World/Objects/Obj_{i:02d}", mass=5.0)
