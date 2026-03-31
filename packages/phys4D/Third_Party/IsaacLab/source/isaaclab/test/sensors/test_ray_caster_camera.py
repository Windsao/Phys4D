







"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher


simulation_app = AppLauncher(headless=True, enable_cameras=True).app

"""Rest everything follows."""

import copy
import numpy as np
import os
import torch

import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
import omni.replicator.core as rep
import pytest
from pxr import Gf

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sensors.ray_caster import RayCasterCamera, RayCasterCameraCfg, patterns
from isaaclab.sim import PinholeCameraCfg
from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.terrains.utils import create_prim_from_mesh
from isaaclab.utils import convert_dict_to_backend
from isaaclab.utils.timer import Timer


POSITION = [2.5, 2.5, 2.5]
QUAT_ROS = [-0.17591989, 0.33985114, 0.82047325, -0.42470819]
QUAT_OPENGL = [0.33985113, 0.17591988, 0.42470818, 0.82047324]
QUAT_WORLD = [-0.3647052, -0.27984815, -0.1159169, 0.88047623]

DEBUG_PLOTS = False


def setup() -> tuple[sim_utils.SimulationContext, RayCasterCameraCfg, float]:

    camera_pattern_cfg = patterns.PinholeCameraPatternCfg(
        focal_length=24.0,
        horizontal_aperture=20.955,
        height=480,
        width=640,
    )
    camera_cfg = RayCasterCameraCfg(
        prim_path="/World/Camera",
        mesh_prim_paths=["/World/defaultGroundPlane"],
        update_period=0,
        offset=RayCasterCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        debug_vis=False,
        pattern_cfg=camera_pattern_cfg,
        data_types=[
            "distance_to_image_plane",
        ],
    )

    stage_utils.create_new_stage()

    prim_utils.create_prim("/World/Camera", "Xform")

    dt = 0.01

    sim_cfg = sim_utils.SimulationCfg(dt=dt)
    sim = sim_utils.SimulationContext(sim_cfg)

    mesh = make_plane(size=(100, 100), height=0.0, center_zero=True)
    create_prim_from_mesh("/World/defaultGroundPlane", mesh)

    stage_utils.update_stage()
    return sim, camera_cfg, dt


def teardown(sim: sim_utils.SimulationContext):


    rep.vp_manager.destroy_hydra_textures("Replicator")


    sim._timeline.stop()

    sim.clear_all_callbacks()
    sim.clear_instance()


@pytest.fixture
def setup_sim():
    """Setup and teardown for each test."""
    sim, camera_cfg, dt = setup()
    yield sim, camera_cfg, dt
    teardown(sim)


@pytest.mark.isaacsim_ci
def test_camera_init(setup_sim):
    """Test camera initialization."""
    sim, camera_cfg, dt = setup_sim

    camera = RayCasterCamera(cfg=camera_cfg)

    sim.reset()

    assert camera.is_initialized

    assert camera.data.pos_w.shape == (1, 3)
    assert camera.data.quat_w_ros.shape == (1, 4)
    assert camera.data.quat_w_world.shape == (1, 4)
    assert camera.data.quat_w_opengl.shape == (1, 4)
    assert camera.data.intrinsic_matrices.shape == (1, 3, 3)
    assert camera.data.image_shape == (camera_cfg.pattern_cfg.height, camera_cfg.pattern_cfg.width)
    assert camera.data.info == [{camera_cfg.data_types[0]: None}]

    for _ in range(10):
        sim.step()
        camera.update(dt)

        for im_data in camera.data.output.values():
            assert im_data.shape == (1, camera_cfg.pattern_cfg.height, camera_cfg.pattern_cfg.width, 1)


    camera.reset()
    assert torch.all(camera.frame == 0)

    for _ in range(10):
        sim.step()
        camera.update(dt)
    camera.reset(env_ids=[0])
    assert camera.frame[0] == 0


@pytest.mark.isaacsim_ci
def test_camera_resolution(setup_sim):
    """Test camera resolution is correctly set."""
    sim, camera_cfg, dt = setup_sim

    camera = RayCasterCamera(cfg=camera_cfg)

    sim.reset()
    camera.update(dt)

    for im_data in camera.data.output.values():
        assert im_data.shape == (1, camera_cfg.pattern_cfg.height, camera_cfg.pattern_cfg.width, 1)


@pytest.mark.isaacsim_ci
def test_depth_clipping(setup_sim):
    """Test depth clipping.

    .. note::

        This test is the same for all camera models to enforce the same clipping behavior.
    """
    sim, camera_cfg, dt = setup_sim
    prim_utils.create_prim("/World/CameraZero", "Xform")
    prim_utils.create_prim("/World/CameraNone", "Xform")
    prim_utils.create_prim("/World/CameraMax", "Xform")


    camera_cfg_zero = RayCasterCameraCfg(
        prim_path="/World/CameraZero",
        mesh_prim_paths=["/World/defaultGroundPlane"],
        offset=RayCasterCameraCfg.OffsetCfg(pos=(2.5, 2.5, 6.0), rot=(0.9914449, 0.0, 0.1305, 0.0), convention="world"),
        pattern_cfg=patterns.PinholeCameraPatternCfg().from_intrinsic_matrix(
            focal_length=38.0,
            intrinsic_matrix=[380.08, 0.0, 467.79, 0.0, 380.08, 262.05, 0.0, 0.0, 1.0],
            height=540,
            width=960,
        ),
        max_distance=10.0,
        data_types=["distance_to_image_plane", "distance_to_camera"],
        depth_clipping_behavior="zero",
    )
    camera_zero = RayCasterCamera(camera_cfg_zero)

    camera_cfg_none = copy.deepcopy(camera_cfg_zero)
    camera_cfg_none.prim_path = "/World/CameraNone"
    camera_cfg_none.depth_clipping_behavior = "none"
    camera_none = RayCasterCamera(camera_cfg_none)

    camera_cfg_max = copy.deepcopy(camera_cfg_zero)
    camera_cfg_max.prim_path = "/World/CameraMax"
    camera_cfg_max.depth_clipping_behavior = "max"
    camera_max = RayCasterCamera(camera_cfg_max)


    sim.reset()

    camera_zero.update(dt)
    camera_none.update(dt)
    camera_max.update(dt)


    assert torch.isinf(camera_none.data.output["distance_to_camera"]).any()
    assert torch.isnan(camera_none.data.output["distance_to_image_plane"]).any()
    assert (
        camera_none.data.output["distance_to_camera"][~torch.isinf(camera_none.data.output["distance_to_camera"])].max()
        > camera_cfg_zero.max_distance
    )
    assert (
        camera_none.data.output["distance_to_image_plane"][
            ~torch.isnan(camera_none.data.output["distance_to_image_plane"])
        ].max()
        > camera_cfg_zero.max_distance
    )


    assert torch.all(
        camera_zero.data.output["distance_to_camera"][torch.isinf(camera_none.data.output["distance_to_camera"])] == 0.0
    )
    assert torch.all(
        camera_zero.data.output["distance_to_image_plane"][
            torch.isnan(camera_none.data.output["distance_to_image_plane"])
        ]
        == 0.0
    )
    assert camera_zero.data.output["distance_to_camera"].max() <= camera_cfg_zero.max_distance
    assert camera_zero.data.output["distance_to_image_plane"].max() <= camera_cfg_zero.max_distance


    assert torch.all(
        camera_max.data.output["distance_to_camera"][torch.isinf(camera_none.data.output["distance_to_camera"])]
        == camera_cfg_zero.max_distance
    )
    assert torch.all(
        camera_max.data.output["distance_to_image_plane"][
            torch.isnan(camera_none.data.output["distance_to_image_plane"])
        ]
        == camera_cfg_zero.max_distance
    )
    assert camera_max.data.output["distance_to_camera"].max() <= camera_cfg_zero.max_distance
    assert camera_max.data.output["distance_to_image_plane"].max() <= camera_cfg_zero.max_distance


@pytest.mark.isaacsim_ci
def test_camera_init_offset(setup_sim):
    """Test camera initialization with offset using different conventions."""
    sim, camera_cfg, dt = setup_sim


    cam_cfg_offset_ros = copy.deepcopy(camera_cfg)
    cam_cfg_offset_ros.offset = RayCasterCameraCfg.OffsetCfg(
        pos=(POSITION[0], POSITION[1], POSITION[2]),
        rot=(QUAT_ROS[0], QUAT_ROS[1], QUAT_ROS[2], QUAT_ROS[3]),
        convention="ros",
    )
    prim_utils.create_prim("/World/CameraOffsetRos", "Xform")
    cam_cfg_offset_ros.prim_path = "/World/CameraOffsetRos"
    camera_ros = RayCasterCamera(cam_cfg_offset_ros)

    cam_cfg_offset_opengl = copy.deepcopy(camera_cfg)
    cam_cfg_offset_opengl.offset = RayCasterCameraCfg.OffsetCfg(
        pos=(POSITION[0], POSITION[1], POSITION[2]),
        rot=(QUAT_OPENGL[0], QUAT_OPENGL[1], QUAT_OPENGL[2], QUAT_OPENGL[3]),
        convention="opengl",
    )
    prim_utils.create_prim("/World/CameraOffsetOpengl", "Xform")
    cam_cfg_offset_opengl.prim_path = "/World/CameraOffsetOpengl"
    camera_opengl = RayCasterCamera(cam_cfg_offset_opengl)

    cam_cfg_offset_world = copy.deepcopy(camera_cfg)
    cam_cfg_offset_world.offset = RayCasterCameraCfg.OffsetCfg(
        pos=(POSITION[0], POSITION[1], POSITION[2]),
        rot=(QUAT_WORLD[0], QUAT_WORLD[1], QUAT_WORLD[2], QUAT_WORLD[3]),
        convention="world",
    )
    prim_utils.create_prim("/World/CameraOffsetWorld", "Xform")
    cam_cfg_offset_world.prim_path = "/World/CameraOffsetWorld"
    camera_world = RayCasterCamera(cam_cfg_offset_world)


    sim.reset()


    camera_world.update(dt)
    camera_opengl.update(dt)
    camera_ros.update(dt)


    np.testing.assert_allclose(camera_ros.data.pos_w[0].cpu().numpy(), cam_cfg_offset_ros.offset.pos)
    np.testing.assert_allclose(camera_opengl.data.pos_w[0].cpu().numpy(), cam_cfg_offset_opengl.offset.pos)
    np.testing.assert_allclose(camera_world.data.pos_w[0].cpu().numpy(), cam_cfg_offset_world.offset.pos)


    np.testing.assert_allclose(camera_ros.data.pos_w[0].cpu().numpy(), cam_cfg_offset_ros.offset.pos, rtol=1e-5)
    np.testing.assert_allclose(camera_ros.data.quat_w_ros[0].cpu().numpy(), QUAT_ROS, rtol=1e-5)
    np.testing.assert_allclose(camera_ros.data.quat_w_opengl[0].cpu().numpy(), QUAT_OPENGL, rtol=1e-5)
    np.testing.assert_allclose(camera_ros.data.quat_w_world[0].cpu().numpy(), QUAT_WORLD, rtol=1e-5)


@pytest.mark.isaacsim_ci
def test_camera_init_intrinsic_matrix(setup_sim):
    """Test camera initialization from intrinsic matrix."""
    sim, camera_cfg, dt = setup_sim

    camera_1 = RayCasterCamera(cfg=camera_cfg)

    sim.reset()
    intrinsic_matrix = camera_1.data.intrinsic_matrices[0].cpu().flatten().tolist()
    teardown(sim)

    sim, camera_cfg, dt = setup()
    camera_1 = RayCasterCamera(cfg=camera_cfg)

    intrinsic_camera_cfg = RayCasterCameraCfg(
        prim_path="/World/Camera",
        mesh_prim_paths=["/World/defaultGroundPlane"],
        update_period=0,
        offset=RayCasterCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        debug_vis=False,
        pattern_cfg=patterns.PinholeCameraPatternCfg.from_intrinsic_matrix(
            intrinsic_matrix=intrinsic_matrix,
            height=camera_cfg.pattern_cfg.height,
            width=camera_cfg.pattern_cfg.width,
            focal_length=camera_cfg.pattern_cfg.focal_length,
        ),
        data_types=[
            "distance_to_image_plane",
        ],
    )
    camera_2 = RayCasterCamera(cfg=intrinsic_camera_cfg)


    sim.reset()
    sim.play()


    camera_1.update(dt)
    camera_2.update(dt)


    torch.testing.assert_close(
        camera_1.data.output["distance_to_image_plane"],
        camera_2.data.output["distance_to_image_plane"],
    )

    torch.testing.assert_close(
        camera_1.data.intrinsic_matrices[0],
        camera_2.data.intrinsic_matrices[0],
    )


@pytest.mark.isaacsim_ci
def test_multi_camera_init(setup_sim):
    """Test multi-camera initialization."""
    sim, camera_cfg, dt = setup_sim


    cam_cfg_1 = copy.deepcopy(camera_cfg)
    cam_cfg_1.prim_path = "/World/Camera_1"
    prim_utils.create_prim("/World/Camera_1", "Xform")

    cam_1 = RayCasterCamera(cam_cfg_1)

    cam_cfg_2 = copy.deepcopy(camera_cfg)
    cam_cfg_2.prim_path = "/World/Camera_2"
    prim_utils.create_prim("/World/Camera_2", "Xform")
    cam_2 = RayCasterCamera(cam_cfg_2)


    assert cam_1.meshes == cam_2.meshes


    sim.reset()


    for _ in range(5):
        sim.step()

    for _ in range(10):

        sim.step()

        cam_1.update(dt)
        cam_2.update(dt)

        for cam in [cam_1, cam_2]:
            for im_data in cam.data.output.values():
                assert im_data.shape == (1, camera_cfg.pattern_cfg.height, camera_cfg.pattern_cfg.width, 1)


@pytest.mark.isaacsim_ci
def test_camera_set_world_poses(setup_sim):
    """Test camera function to set specific world pose."""
    sim, camera_cfg, dt = setup_sim
    camera = RayCasterCamera(camera_cfg)

    sim.reset()


    position = torch.tensor([POSITION], dtype=torch.float32, device=camera.device)
    orientation = torch.tensor([QUAT_WORLD], dtype=torch.float32, device=camera.device)

    camera.set_world_poses(position.clone(), orientation.clone(), convention="world")


    torch.testing.assert_close(camera.data.pos_w, position)
    torch.testing.assert_close(camera.data.quat_w_world, orientation)


@pytest.mark.isaacsim_ci
def test_camera_set_world_poses_from_view(setup_sim):
    """Test camera function to set specific world pose from view."""
    sim, camera_cfg, dt = setup_sim
    camera = RayCasterCamera(camera_cfg)

    sim.reset()


    eyes = torch.tensor([POSITION], dtype=torch.float32, device=camera.device)
    targets = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32, device=camera.device)
    quat_ros_gt = torch.tensor([QUAT_ROS], dtype=torch.float32, device=camera.device)

    camera.set_world_poses_from_view(eyes.clone(), targets.clone())


    torch.testing.assert_close(camera.data.pos_w, eyes)
    torch.testing.assert_close(camera.data.quat_w_ros, quat_ros_gt)


@pytest.mark.isaacsim_ci
def test_intrinsic_matrix(setup_sim):
    """Checks that the camera's set and retrieve methods work for intrinsic matrix."""
    sim, camera_cfg, dt = setup_sim
    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.pattern_cfg.height = 240
    camera_cfg.pattern_cfg.width = 320
    camera = RayCasterCamera(camera_cfg)

    sim.reset()

    rs_intrinsic_matrix = [229.31640625, 0.0, 164.810546875, 0.0, 229.826171875, 122.1650390625, 0.0, 0.0, 1.0]
    rs_intrinsic_matrix = torch.tensor(rs_intrinsic_matrix, device=camera.device).reshape(3, 3).unsqueeze(0)

    camera.set_intrinsic_matrices(rs_intrinsic_matrix.clone())

    for _ in range(10):

        sim.step()

        camera.update(dt)

        torch.testing.assert_close(rs_intrinsic_matrix, camera.data.intrinsic_matrices)


@pytest.mark.isaacsim_ci
def test_throughput(setup_sim):
    """Checks that the single camera gets created properly with a rig."""
    sim, camera_cfg, dt = setup_sim

    file_dir = os.path.dirname(os.path.realpath(__file__))
    temp_dir = os.path.join(file_dir, "output", "camera", "throughput")
    os.makedirs(temp_dir, exist_ok=True)

    rep_writer = rep.BasicWriter(output_dir=temp_dir, frame_padding=3)

    camera_cfg.pattern_cfg.height = 480
    camera_cfg.pattern_cfg.width = 640
    camera = RayCasterCamera(camera_cfg)


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
            assert im_data.shape == (1, camera_cfg.pattern_cfg.height, camera_cfg.pattern_cfg.width, 1)


@pytest.mark.isaacsim_ci
def test_output_equal_to_usdcamera(setup_sim):
    sim, camera_cfg, dt = setup_sim
    camera_pattern_cfg = patterns.PinholeCameraPatternCfg(
        focal_length=24.0,
        horizontal_aperture=20.955,
        height=240,
        width=320,
    )
    prim_utils.create_prim("/World/Camera_warp", "Xform")
    camera_cfg_warp = RayCasterCameraCfg(
        prim_path="/World/Camera",
        mesh_prim_paths=["/World/defaultGroundPlane"],
        update_period=0,
        offset=RayCasterCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        debug_vis=False,
        pattern_cfg=camera_pattern_cfg,
        data_types=["distance_to_image_plane", "distance_to_camera", "normals"],
    )

    camera_warp = RayCasterCamera(camera_cfg_warp)


    camera_cfg_usd = CameraCfg(
        height=240,
        width=320,
        prim_path="/World/Camera_usd",
        update_period=0,
        data_types=["distance_to_image_plane", "distance_to_camera", "normals"],
        spawn=PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(1e-4, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    camera_usd = Camera(camera_cfg_usd)


    sim.reset()
    sim.play()


    eyes = torch.tensor([[2.5, 2.5, 4.5]], dtype=torch.float32, device=camera_warp.device)
    targets = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32, device=camera_warp.device)

    camera_warp.set_world_poses_from_view(eyes, targets)
    camera_usd.set_world_poses_from_view(eyes, targets)


    for _ in range(5):
        sim.step()


    camera_usd.update(dt)
    camera_warp.update(dt)


    torch.testing.assert_close(
        camera_usd.data.intrinsic_matrices,
        camera_warp.data.intrinsic_matrices,
    )


    torch.testing.assert_close(
        camera_usd._sensor_prims[0].GetHorizontalApertureAttr().Get(),
        camera_cfg_warp.pattern_cfg.horizontal_aperture,
    )
    torch.testing.assert_close(
        camera_usd._sensor_prims[0].GetVerticalApertureAttr().Get(),
        (
            camera_cfg_warp.pattern_cfg.horizontal_aperture
            * camera_cfg_warp.pattern_cfg.height
            / camera_cfg_warp.pattern_cfg.width
        ),
    )


    torch.testing.assert_close(
        camera_usd.data.output["distance_to_image_plane"],
        camera_warp.data.output["distance_to_image_plane"],
    )
    torch.testing.assert_close(
        camera_usd.data.output["distance_to_camera"],
        camera_warp.data.output["distance_to_camera"],
        atol=5e-5,
        rtol=5e-6,
    )



    torch.testing.assert_close(
        camera_usd.data.output["normals"][..., :3],
        camera_warp.data.output["normals"],
        rtol=1e-5,
        atol=1e-4,
    )


@pytest.mark.isaacsim_ci
def test_output_equal_to_usdcamera_offset(setup_sim):
    sim, camera_cfg, dt = setup_sim
    offset_rot = [-0.1251, 0.3617, 0.8731, -0.3020]

    camera_pattern_cfg = patterns.PinholeCameraPatternCfg(
        focal_length=24.0,
        horizontal_aperture=20.955,
        height=240,
        width=320,
    )
    prim_utils.create_prim("/World/Camera_warp", "Xform")
    camera_cfg_warp = RayCasterCameraCfg(
        prim_path="/World/Camera",
        mesh_prim_paths=["/World/defaultGroundPlane"],
        update_period=0,
        offset=RayCasterCameraCfg.OffsetCfg(pos=(2.5, 2.5, 4.0), rot=tuple(offset_rot), convention="ros"),
        debug_vis=False,
        pattern_cfg=camera_pattern_cfg,
        data_types=["distance_to_image_plane", "distance_to_camera", "normals"],
    )

    camera_warp = RayCasterCamera(camera_cfg_warp)


    camera_cfg_usd = CameraCfg(
        height=240,
        width=320,
        prim_path="/World/Camera_usd",
        update_period=0,
        data_types=["distance_to_image_plane", "distance_to_camera", "normals"],
        spawn=PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(1e-6, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(2.5, 2.5, 4.0), rot=(offset_rot[0], offset_rot[1], offset_rot[2], offset_rot[3]), convention="ros"
        ),
    )
    camera_usd = Camera(camera_cfg_usd)


    sim.reset()
    sim.play()


    for _ in range(5):
        sim.step()


    camera_usd.update(dt)
    camera_warp.update(dt)


    torch.testing.assert_close(
        camera_usd.data.output["distance_to_image_plane"],
        camera_warp.data.output["distance_to_image_plane"],
        rtol=1e-3,
        atol=1e-5,
    )
    torch.testing.assert_close(
        camera_usd.data.output["distance_to_camera"],
        camera_warp.data.output["distance_to_camera"],
        rtol=1e-3,
        atol=1e-5,
    )



    torch.testing.assert_close(
        camera_usd.data.output["normals"][..., :3],
        camera_warp.data.output["normals"],
        rtol=1e-5,
        atol=1e-4,
    )


@pytest.mark.isaacsim_ci
def test_output_equal_to_usdcamera_prim_offset(setup_sim):
    """Test that the output of the ray caster camera is equal to the output of the usd camera when both are placed
    under an XForm prim that is translated and rotated from the world origin
    ."""
    sim, camera_cfg, dt = setup_sim
    offset_rot = (-0.1251, 0.3617, 0.8731, -0.3020)


    gf_quatf = Gf.Quatd()
    gf_quatf.SetReal(QUAT_OPENGL[0])
    gf_quatf.SetImaginary(tuple(QUAT_OPENGL[1:]))

    camera_pattern_cfg = patterns.PinholeCameraPatternCfg(
        focal_length=24.0,
        horizontal_aperture=20.955,
        height=240,
        width=320,
    )
    prim_raycast_cam = prim_utils.create_prim("/World/Camera_warp", "Xform")
    prim_raycast_cam.GetAttribute("xformOp:translate").Set(tuple(POSITION))
    prim_raycast_cam.GetAttribute("xformOp:orient").Set(gf_quatf)

    camera_cfg_warp = RayCasterCameraCfg(
        prim_path="/World/Camera_warp",
        mesh_prim_paths=["/World/defaultGroundPlane"],
        update_period=0,
        offset=RayCasterCameraCfg.OffsetCfg(pos=(0, 0, 2.0), rot=offset_rot, convention="ros"),
        debug_vis=False,
        pattern_cfg=camera_pattern_cfg,
        data_types=["distance_to_image_plane", "distance_to_camera", "normals"],
    )

    camera_warp = RayCasterCamera(camera_cfg_warp)


    camera_cfg_usd = CameraCfg(
        height=240,
        width=320,
        prim_path="/World/Camera_usd/camera",
        update_period=0,
        data_types=["distance_to_image_plane", "distance_to_camera", "normals"],
        spawn=PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(1e-6, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0, 0, 2.0), rot=offset_rot, convention="ros"),
        update_latest_camera_pose=True,
    )
    prim_usd = prim_utils.create_prim("/World/Camera_usd", "Xform")
    prim_usd.GetAttribute("xformOp:translate").Set(tuple(POSITION))
    prim_usd.GetAttribute("xformOp:orient").Set(gf_quatf)

    camera_usd = Camera(camera_cfg_usd)


    sim.reset()
    sim.play()


    for _ in range(5):
        sim.step()


    camera_usd.update(dt)
    camera_warp.update(dt)


    torch.testing.assert_close(camera_warp.data.pos_w[0], camera_usd.data.pos_w[0])
    torch.testing.assert_close(camera_warp.data.quat_w_ros[0], camera_usd.data.quat_w_ros[0])


    torch.testing.assert_close(
        camera_usd.data.output["distance_to_image_plane"],
        camera_warp.data.output["distance_to_image_plane"],
        rtol=1e-3,
        atol=1e-5,
    )
    torch.testing.assert_close(
        camera_usd.data.output["distance_to_camera"],
        camera_warp.data.output["distance_to_camera"],
        rtol=4e-6,
        atol=2e-5,
    )



    torch.testing.assert_close(
        camera_usd.data.output["normals"][..., :3],
        camera_warp.data.output["normals"],
        rtol=1e-5,
        atol=1e-4,
    )


@pytest.mark.parametrize("focal_length", [0.193, 1.93, 19.3])
@pytest.mark.isaacsim_ci
def test_output_equal_to_usd_camera_intrinsics(setup_sim, focal_length):
    """
    Test that the output of the ray caster camera and usd camera are the same when both are
    initialized with the same intrinsic matrix.
    """

    sim, camera_cfg, dt = setup_sim

    offset_rot = (-0.1251, 0.3617, 0.8731, -0.3020)
    offset_pos = (2.5, 2.5, 4.0)
    intrinsics = [380.0831, 0.0, 480.0, 0.0, 380.0831, 270.0, 0.0, 0.0, 1.0]
    prim_utils.create_prim("/World/Camera_warp", "Xform")

    camera_warp_cfg = RayCasterCameraCfg(
        prim_path="/World/Camera_warp",
        mesh_prim_paths=["/World/defaultGroundPlane"],
        offset=RayCasterCameraCfg.OffsetCfg(pos=offset_pos, rot=offset_rot, convention="ros"),
        debug_vis=False,
        pattern_cfg=patterns.PinholeCameraPatternCfg.from_intrinsic_matrix(
            intrinsic_matrix=intrinsics,
            height=540,
            width=960,
            focal_length=focal_length,
        ),
        depth_clipping_behavior="max",
        max_distance=20.0,
        data_types=["distance_to_image_plane"],
    )
    camera_usd_cfg = CameraCfg(
        prim_path="/World/Camera_usd",
        offset=CameraCfg.OffsetCfg(pos=offset_pos, rot=offset_rot, convention="ros"),
        spawn=PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=intrinsics,
            height=540,
            width=960,
            clipping_range=(0.01, 20),
            focal_length=focal_length,
        ),
        height=540,
        width=960,
        depth_clipping_behavior="max",
        data_types=["distance_to_image_plane"],
    )


    camera_warp_cfg.pattern_cfg.horizontal_aperture_offset = 0
    camera_warp_cfg.pattern_cfg.vertical_aperture_offset = 0
    camera_usd_cfg.spawn.horizontal_aperture_offset = 0
    camera_usd_cfg.spawn.vertical_aperture_offset = 0

    camera_warp = RayCasterCamera(camera_warp_cfg)
    camera_usd = Camera(camera_usd_cfg)


    sim.reset()
    sim.play()


    for _ in range(5):
        sim.step()


    camera_usd.update(dt)
    camera_warp.update(dt)


    cam_warp_output = camera_warp.data.output["distance_to_image_plane"].clone()
    cam_usd_output = camera_usd.data.output["distance_to_image_plane"].clone()
    cam_warp_output[torch.isnan(cam_warp_output)] = 0
    cam_warp_output[torch.isinf(cam_warp_output)] = 0
    cam_usd_output[torch.isnan(cam_usd_output)] = 0
    cam_usd_output[torch.isinf(cam_usd_output)] = 0


    torch.testing.assert_close(camera_warp.data.intrinsic_matrices[0], camera_usd.data.intrinsic_matrices[0])


    torch.testing.assert_close(
        camera_usd._sensor_prims[0].GetHorizontalApertureAttr().Get(),
        camera_warp_cfg.pattern_cfg.horizontal_aperture,
    )
    torch.testing.assert_close(
        camera_usd._sensor_prims[0].GetVerticalApertureAttr().Get(),
        camera_warp_cfg.pattern_cfg.vertical_aperture,
    )

    if DEBUG_PLOTS:

        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        usd_plt = axs[0].imshow(cam_usd_output[0].cpu().numpy())
        fig.colorbar(usd_plt, ax=axs[0])
        axs[0].set_title("USD")
        warp_plt = axs[1].imshow(cam_warp_output[0].cpu().numpy())
        fig.colorbar(warp_plt, ax=axs[1])
        axs[1].set_title("WARP")
        diff_plt = axs[2].imshow(torch.abs(cam_usd_output - cam_warp_output)[0].cpu().numpy())
        fig.colorbar(diff_plt, ax=axs[2])
        axs[2].set_title("Difference")

        plt.tight_layout()
        plt.savefig(
            f"{os.path.dirname(os.path.abspath(__file__))}/output/test_output_equal_to_usd_camera_intrinsics_{focal_length}.png"
        )
        plt.close()


    if focal_length != 0.193:

        torch.testing.assert_close(
            cam_warp_output,
            cam_usd_output,
            atol=5e-5,
            rtol=5e-6,
        )

    del camera_warp, camera_usd


@pytest.mark.parametrize("focal_length_aperture", [(0.193, 0.20955), (1.93, 2.0955), (19.3, 20.955), (0.193, 20.955)])
@pytest.mark.isaacsim_ci
def test_output_equal_to_usd_camera_when_intrinsics_set(setup_sim, focal_length_aperture):
    """
    Test that the output of the ray caster camera is equal to the output of the usd camera when both are placed
    under an XForm prim and an intrinsic matrix is set.
    """

    focal_length, aperture = focal_length_aperture

    sim, camera_cfg, dt = setup_sim
    camera_pattern_cfg = patterns.PinholeCameraPatternCfg(
        focal_length=focal_length,
        horizontal_aperture=aperture,
        height=540,
        width=960,
    )
    camera_cfg_warp = RayCasterCameraCfg(
        prim_path="/World/Camera",
        mesh_prim_paths=["/World/defaultGroundPlane"],
        update_period=0,
        offset=RayCasterCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        debug_vis=False,
        pattern_cfg=camera_pattern_cfg,
        data_types=["distance_to_camera"],
    )

    camera_warp = RayCasterCamera(camera_cfg_warp)


    camera_cfg_usd = CameraCfg(
        height=540,
        width=960,
        prim_path="/World/Camera_usd",
        update_period=0,
        data_types=["distance_to_camera"],
        spawn=PinholeCameraCfg(
            focal_length=focal_length, focus_distance=400.0, horizontal_aperture=aperture, clipping_range=(1e-4, 1.0e5)
        ),
    )
    camera_usd = Camera(camera_cfg_usd)


    sim.reset()
    sim.play()











    camera_warp.set_world_poses_from_view(
        eyes=torch.tensor([[0.0, 0.0, 5.0]], device=camera_warp.device),
        targets=torch.tensor([[0.0, 0.0, 0.0]], device=camera_warp.device),
    )
    camera_usd.set_world_poses_from_view(
        eyes=torch.tensor([[0.0, 0.0, 5.0]], device=camera_usd.device),
        targets=torch.tensor([[0.0, 0.0, 0.0]], device=camera_usd.device),
    )


    for _ in range(5):
        sim.step()


    camera_usd.update(dt)
    camera_warp.update(dt)

    if DEBUG_PLOTS:

        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        usd_plt = axs[0].imshow(camera_usd.data.output["distance_to_camera"][0].cpu().numpy())
        fig.colorbar(usd_plt, ax=axs[0])
        axs[0].set_title("USD")
        warp_plt = axs[1].imshow(camera_warp.data.output["distance_to_camera"][0].cpu().numpy())
        fig.colorbar(warp_plt, ax=axs[1])
        axs[1].set_title("WARP")
        diff_plt = axs[2].imshow(
            torch.abs(camera_usd.data.output["distance_to_camera"] - camera_warp.data.output["distance_to_camera"])[0]
            .cpu()
            .numpy()
        )
        fig.colorbar(diff_plt, ax=axs[2])
        axs[2].set_title("Difference")

        plt.tight_layout()
        plt.savefig(
            f"{os.path.dirname(os.path.abspath(__file__))}/output/test_output_equal_to_usd_camera_when_intrinsics_set_{focal_length}_{aperture}.png"
        )
        plt.close()


    if focal_length != 0.193:

        torch.testing.assert_close(
            camera_usd.data.output["distance_to_camera"],
            camera_warp.data.output["distance_to_camera"],
            rtol=5e-3,
            atol=1e-4,
        )

    del camera_warp, camera_usd


@pytest.mark.isaacsim_ci
def test_sensor_print(setup_sim):
    """Test sensor print is working correctly."""
    sim, camera_cfg, dt = setup_sim

    sensor = RayCasterCamera(cfg=camera_cfg)

    sim.reset()

    print(sensor)
