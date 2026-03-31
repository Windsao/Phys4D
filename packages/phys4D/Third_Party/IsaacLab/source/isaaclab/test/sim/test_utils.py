




"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher


simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import numpy as np
import torch

import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
import pytest
from pxr import Sdf, Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR


@pytest.fixture(autouse=True)
def test_setup_teardown():
    """Create a blank new stage for each test."""

    stage_utils.create_new_stage()
    stage_utils.update_stage()


    yield


    stage_utils.clear_stage()


def test_get_all_matching_child_prims():
    """Test get_all_matching_child_prims() function."""

    prim_utils.create_prim("/World/Floor")
    prim_utils.create_prim("/World/Floor/Box", "Cube", position=np.array([75, 75, -150.1]), attributes={"size": 300})
    prim_utils.create_prim("/World/Wall", "Sphere", attributes={"radius": 1e3})


    isaac_sim_result = prim_utils.get_all_matching_child_prims("/World")
    isaaclab_result = sim_utils.get_all_matching_child_prims("/World")
    assert isaac_sim_result == isaaclab_result




    prim_utils.create_prim(
        "/World/Franka", "Xform", usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd"
    )


    isaaclab_result = sim_utils.get_all_matching_child_prims("/World", predicate=lambda x: x.GetTypeName() == "Cube")
    assert len(isaaclab_result) == 1
    assert isaaclab_result[0].GetPrimPath() == "/World/Floor/Box"


    isaaclab_result = sim_utils.get_all_matching_child_prims(
        "/World/Franka/panda_hand/visuals", predicate=lambda x: x.GetTypeName() == "Mesh"
    )
    assert len(isaaclab_result) == 1
    assert isaaclab_result[0].GetPrimPath() == "/World/Franka/panda_hand/visuals/panda_hand"


    with pytest.raises(ValueError):
        sim_utils.get_all_matching_child_prims("World/Room")


def test_get_first_matching_child_prim():
    """Test get_first_matching_child_prim() function."""

    prim_utils.create_prim("/World/Floor")
    prim_utils.create_prim(
        "/World/env_1/Franka", "Xform", usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd"
    )
    prim_utils.create_prim(
        "/World/env_2/Franka", "Xform", usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd"
    )
    prim_utils.create_prim(
        "/World/env_0/Franka", "Xform", usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd"
    )


    isaaclab_result = sim_utils.get_first_matching_child_prim(
        "/World", predicate=lambda prim: prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    )
    assert isaaclab_result is not None
    assert isaaclab_result.GetPrimPath() == "/World/env_1/Franka"


    isaaclab_result = sim_utils.get_first_matching_child_prim(
        "/World/env_1/Franka", predicate=lambda prim: prim.GetTypeName() == "Mesh"
    )
    assert isaaclab_result is not None
    assert isaaclab_result.GetPrimPath() == "/World/env_1/Franka/panda_link0/visuals/panda_link0"


def test_find_matching_prim_paths():
    """Test find_matching_prim_paths() function."""

    for index in range(2048):
        random_pos = np.random.uniform(-100, 100, size=3)
        prim_utils.create_prim(f"/World/Floor_{index}", "Cube", position=random_pos, attributes={"size": 2.0})
        prim_utils.create_prim(f"/World/Floor_{index}/Sphere", "Sphere", attributes={"radius": 10})
        prim_utils.create_prim(f"/World/Floor_{index}/Sphere/childSphere", "Sphere", attributes={"radius": 1})
        prim_utils.create_prim(f"/World/Floor_{index}/Sphere/childSphere2", "Sphere", attributes={"radius": 1})


    isaac_sim_result = prim_utils.find_matching_prim_paths("/World/Floor_.*/Sphere")
    isaaclab_result = sim_utils.find_matching_prim_paths("/World/Floor_.*/Sphere")
    assert isaac_sim_result == isaaclab_result


    isaac_sim_result = prim_utils.find_matching_prim_paths("/World/Floor_.*")
    isaaclab_result = sim_utils.find_matching_prim_paths("/World/Floor_.*")
    assert isaac_sim_result == isaaclab_result


    isaac_sim_result = prim_utils.find_matching_prim_paths("/World/Floor_.*/Sphere/childSphere.*")
    isaaclab_result = sim_utils.find_matching_prim_paths("/World/Floor_.*/Sphere/childSphere.*")
    assert isaac_sim_result == isaaclab_result


    with pytest.raises(ValueError):
        sim_utils.get_all_matching_child_prims("World/Floor_.*")


def test_find_global_fixed_joint_prim():
    """Test find_global_fixed_joint_prim() function."""

    prim_utils.create_prim("/World")
    prim_utils.create_prim("/World/ANYmal", usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-C/anymal_c.usd")
    prim_utils.create_prim(
        "/World/Franka", usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd"
    )
    if "4.5" in ISAAC_NUCLEUS_DIR:
        franka_usd = f"{ISAAC_NUCLEUS_DIR}/Robots/Franka/franka.usd"
    else:
        franka_usd = f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaPanda/franka.usd"
    prim_utils.create_prim("/World/Franka_Isaac", usd_path=franka_usd)


    assert sim_utils.find_global_fixed_joint_prim("/World/ANYmal") is None
    assert sim_utils.find_global_fixed_joint_prim("/World/Franka") is not None
    assert sim_utils.find_global_fixed_joint_prim("/World/Franka_Isaac") is not None


    joint_prim = sim_utils.find_global_fixed_joint_prim("/World/Franka")
    joint_prim.GetJointEnabledAttr().Set(False)
    assert sim_utils.find_global_fixed_joint_prim("/World/Franka") is not None
    assert sim_utils.find_global_fixed_joint_prim("/World/Franka", check_enabled_only=True) is None


def test_select_usd_variants():
    """Test select_usd_variants() function."""
    stage = stage_utils.get_current_stage()
    prim: Usd.Prim = UsdGeom.Xform.Define(stage, Sdf.Path("/World")).GetPrim()
    stage.SetDefaultPrim(prim)


    variants = ["red", "blue", "green"]
    variant_set = prim.GetVariantSets().AddVariantSet("colors")
    for variant in variants:
        variant_set.AddVariant(variant)


    sim_utils.utils.select_usd_variants("/World", {"colors": "red"}, stage)


    assert variant_set.GetVariantSelection() == "red"


def test_resolve_prim_pose():
    """Test resolve_prim_pose() function."""

    num_objects = 20

    rand_scales = np.random.uniform(0.5, 1.5, size=(num_objects, 3, 3))
    rand_widths = np.random.uniform(0.1, 10.0, size=(num_objects,))

    rand_positions = np.random.uniform(-100, 100, size=(num_objects, 3, 3))

    rand_quats = np.random.randn(num_objects, 3, 4)
    rand_quats /= np.linalg.norm(rand_quats, axis=2, keepdims=True)


    for i in range(num_objects):

        cube_prim = prim_utils.create_prim(
            f"/World/Cubes/instance_{i:02d}",
            "Cube",
            translation=rand_positions[i, 0],
            orientation=rand_quats[i, 0],
            scale=rand_scales[i, 0],
            attributes={"size": rand_widths[i]},
        )

        xform_prim = prim_utils.create_prim(
            f"/World/Xform/instance_{i:02d}",
            "Xform",
            translation=rand_positions[i, 1],
            orientation=rand_quats[i, 1],
            scale=rand_scales[i, 1],
        )
        geometry_prim = prim_utils.create_prim(
            f"/World/Xform/instance_{i:02d}/geometry",
            "Sphere",
            translation=rand_positions[i, 2],
            orientation=rand_quats[i, 2],
            scale=rand_scales[i, 2],
            attributes={"radius": rand_widths[i]},
        )
        dummy_prim = prim_utils.create_prim(
            f"/World/Xform/instance_{i:02d}/dummy",
            "Sphere",
        )


        pos, quat = sim_utils.resolve_prim_pose(cube_prim)
        pos, quat = np.array(pos), np.array(quat)
        quat = quat if np.sign(rand_quats[i, 0, 0]) == np.sign(quat[0]) else -quat
        np.testing.assert_allclose(pos, rand_positions[i, 0], atol=1e-3)
        np.testing.assert_allclose(quat, rand_quats[i, 0], atol=1e-3)

        pos, quat = sim_utils.resolve_prim_pose(xform_prim)
        pos, quat = np.array(pos), np.array(quat)
        quat = quat if np.sign(rand_quats[i, 1, 0]) == np.sign(quat[0]) else -quat
        np.testing.assert_allclose(pos, rand_positions[i, 1], atol=1e-3)
        np.testing.assert_allclose(quat, rand_quats[i, 1], atol=1e-3)

        pos, quat = sim_utils.resolve_prim_pose(dummy_prim)
        pos, quat = np.array(pos), np.array(quat)
        quat = quat if np.sign(rand_quats[i, 1, 0]) == np.sign(quat[0]) else -quat
        np.testing.assert_allclose(pos, rand_positions[i, 1], atol=1e-3)
        np.testing.assert_allclose(quat, rand_quats[i, 1], atol=1e-3)


        pos, quat = sim_utils.resolve_prim_pose(geometry_prim, ref_prim=xform_prim)
        pos, quat = np.array(pos), np.array(quat)
        quat = quat if np.sign(rand_quats[i, 2, 0]) == np.sign(quat[0]) else -quat
        np.testing.assert_allclose(pos, rand_positions[i, 2] * rand_scales[i, 1], atol=1e-3)



        np.testing.assert_allclose(quat, rand_quats[i, 2], atol=1e-3)


        pos, quat = sim_utils.resolve_prim_pose(dummy_prim, ref_prim=xform_prim)
        pos, quat = np.array(pos), np.array(quat)
        np.testing.assert_allclose(pos, np.zeros(3), atol=1e-3)
        np.testing.assert_allclose(quat, np.array([1, 0, 0, 0]), atol=1e-3)

        pos, quat = sim_utils.resolve_prim_pose(xform_prim, ref_prim=cube_prim)
        pos, quat = np.array(pos), np.array(quat)

        gt_pos, gt_quat = math_utils.subtract_frame_transforms(
            torch.from_numpy(rand_positions[i, 0]).unsqueeze(0),
            torch.from_numpy(rand_quats[i, 0]).unsqueeze(0),
            torch.from_numpy(rand_positions[i, 1]).unsqueeze(0),
            torch.from_numpy(rand_quats[i, 1]).unsqueeze(0),
        )
        gt_pos, gt_quat = gt_pos.squeeze(0).numpy(), gt_quat.squeeze(0).numpy()
        quat = quat if np.sign(gt_quat[0]) == np.sign(quat[0]) else -quat
        np.testing.assert_allclose(pos, gt_pos, atol=1e-3)
        np.testing.assert_allclose(quat, gt_quat, atol=1e-3)


def test_resolve_prim_scale():
    """Test resolve_prim_scale() function.

    To simplify the test, we assume that the effective scale at a prim
    is the product of the scales of the prims in the hierarchy:

        scale = scale_of_xform * scale_of_geometry_prim

    This is only true when rotations are identity or the transforms are
    orthogonal and uniformly scaled. Otherwise, scale is not composable
    like that in local component-wise fashion.
    """

    num_objects = 20

    rand_scales = np.random.uniform(0.5, 1.5, size=(num_objects, 3, 3))
    rand_widths = np.random.uniform(0.1, 10.0, size=(num_objects,))

    rand_positions = np.random.uniform(-100, 100, size=(num_objects, 3, 3))


    for i in range(num_objects):

        cube_prim = prim_utils.create_prim(
            f"/World/Cubes/instance_{i:02d}",
            "Cube",
            translation=rand_positions[i, 0],
            scale=rand_scales[i, 0],
            attributes={"size": rand_widths[i]},
        )

        xform_prim = prim_utils.create_prim(
            f"/World/Xform/instance_{i:02d}",
            "Xform",
            translation=rand_positions[i, 1],
            scale=rand_scales[i, 1],
        )
        geometry_prim = prim_utils.create_prim(
            f"/World/Xform/instance_{i:02d}/geometry",
            "Sphere",
            translation=rand_positions[i, 2],
            scale=rand_scales[i, 2],
            attributes={"radius": rand_widths[i]},
        )
        dummy_prim = prim_utils.create_prim(
            f"/World/Xform/instance_{i:02d}/dummy",
            "Sphere",
        )


        scale = sim_utils.resolve_prim_scale(cube_prim)
        scale = np.array(scale)
        np.testing.assert_allclose(scale, rand_scales[i, 0], atol=1e-5)

        scale = sim_utils.resolve_prim_scale(xform_prim)
        scale = np.array(scale)
        np.testing.assert_allclose(scale, rand_scales[i, 1], atol=1e-5)

        scale = sim_utils.resolve_prim_scale(geometry_prim)
        scale = np.array(scale)
        np.testing.assert_allclose(scale, rand_scales[i, 1] * rand_scales[i, 2], atol=1e-5)

        scale = sim_utils.resolve_prim_scale(dummy_prim)
        scale = np.array(scale)
        np.testing.assert_allclose(scale, rand_scales[i, 1], atol=1e-5)
