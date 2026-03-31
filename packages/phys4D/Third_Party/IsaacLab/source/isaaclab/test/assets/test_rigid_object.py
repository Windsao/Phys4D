








"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher


simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import ctypes
import torch
from typing import Literal

import isaacsim.core.utils.prims as prim_utils
import pytest
from flaky import flaky

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import build_simulation_context
from isaaclab.sim.spawners import materials
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.math import (
    combine_frame_transforms,
    default_orientation,
    quat_apply_inverse,
    quat_inv,
    quat_mul,
    quat_rotate,
    random_orientation,
)


def generate_cubes_scene(
    num_cubes: int = 1,
    height=1.0,
    api: Literal["none", "rigid_body", "articulation_root"] = "rigid_body",
    kinematic_enabled: bool = False,
    device: str = "cuda:0",
) -> tuple[RigidObject, torch.Tensor]:
    """Generate a scene with the provided number of cubes.

    Args:
        num_cubes: Number of cubes to generate.
        height: Height of the cubes.
        api: The type of API that the cubes should have.
        kinematic_enabled: Whether the cubes are kinematic.
        device: Device to use for the simulation.

    Returns:
        A tuple containing the rigid object representing the cubes and the origins of the cubes.

    """
    origins = torch.tensor([(i * 1.0, 0, height) for i in range(num_cubes)]).to(device)

    for i, origin in enumerate(origins):
        prim_utils.create_prim(f"/World/Table_{i}", "Xform", translation=origin)


    if api == "none":

        spawn_cfg = sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
    elif api == "rigid_body":
        spawn_cfg = sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=kinematic_enabled),
        )
    elif api == "articulation_root":
        spawn_cfg = sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Tests/RigidObject/Cube/dex_cube_instanceable_with_articulation_root.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=kinematic_enabled),
        )
    else:
        raise ValueError(f"Unknown api: {api}")


    cube_object_cfg = RigidObjectCfg(
        prim_path="/World/Table_.*/Object",
        spawn=spawn_cfg,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, height)),
    )
    cube_object = RigidObject(cfg=cube_object_cfg)

    return cube_object, origins


@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_initialization(num_cubes, device):
    """Test initialization for prim with rigid body API at the provided prim path."""
    with build_simulation_context(device=device, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None

        cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, device=device)


        assert ctypes.c_long.from_address(id(cube_object)).value == 1


        sim.reset()


        assert cube_object.is_initialized
        assert len(cube_object.body_names) == 1


        assert cube_object.data.root_pos_w.shape == (num_cubes, 3)
        assert cube_object.data.root_quat_w.shape == (num_cubes, 4)
        assert cube_object.data.default_mass.shape == (num_cubes, 1)
        assert cube_object.data.default_inertia.shape == (num_cubes, 9)


        for _ in range(2):

            sim.step()

            cube_object.update(sim.cfg.dt)


@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_initialization_with_kinematic_enabled(num_cubes, device):
    """Test that initialization for prim with kinematic flag enabled."""
    with build_simulation_context(device=device, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None

        cube_object, origins = generate_cubes_scene(num_cubes=num_cubes, kinematic_enabled=True, device=device)


        assert ctypes.c_long.from_address(id(cube_object)).value == 1


        sim.reset()


        assert cube_object.is_initialized
        assert len(cube_object.body_names) == 1


        assert cube_object.data.root_pos_w.shape == (num_cubes, 3)
        assert cube_object.data.root_quat_w.shape == (num_cubes, 4)


        for _ in range(2):

            sim.step()

            cube_object.update(sim.cfg.dt)

            default_root_state = cube_object.data.default_root_state.clone()
            default_root_state[:, :3] += origins
            torch.testing.assert_close(cube_object.data.root_state_w, default_root_state)


@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_initialization_with_no_rigid_body(num_cubes, device):
    """Test that initialization fails when no rigid body is found at the provided prim path."""
    with build_simulation_context(device=device, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None

        cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, api="none", device=device)


        assert ctypes.c_long.from_address(id(cube_object)).value == 1


        with pytest.raises(RuntimeError):
            sim.reset()


@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_initialization_with_articulation_root(num_cubes, device):
    """Test that initialization fails when an articulation root is found at the provided prim path."""
    with build_simulation_context(device=device, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None

        cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, api="articulation_root", device=device)


        assert ctypes.c_long.from_address(id(cube_object)).value == 1


        with pytest.raises(RuntimeError):
            sim.reset()


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_external_force_buffer(device):
    """Test if external force buffer correctly updates in the force value is zero case.

    In this test, we apply a non-zero force, then a zero force, then finally a non-zero force
    to an object. We check if the force buffer is properly updated at each step.
    """


    with build_simulation_context(device=device, add_ground_plane=True, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        cube_object, origins = generate_cubes_scene(num_cubes=1, device=device)


        sim.reset()


        body_ids, body_names = cube_object.find_bodies(".*")


        cube_object.reset()


        for step in range(5):


            external_wrench_b = torch.zeros(cube_object.num_instances, len(body_ids), 6, device=sim.device)
            external_wrench_positions_b = torch.zeros(cube_object.num_instances, len(body_ids), 3, device=sim.device)

            if step == 0 or step == 3:

                force = 1
                position = 1
                is_global = True
            else:

                force = 0
                position = 0
                is_global = False


            external_wrench_b[:, :, 0] = force
            external_wrench_b[:, :, 3] = force
            external_wrench_positions_b[:, :, 0] = position


            if step == 0 or step == 3:
                cube_object.set_external_force_and_torque(
                    external_wrench_b[..., :3],
                    external_wrench_b[..., 3:],
                    body_ids=body_ids,
                    positions=external_wrench_positions_b,
                    is_global=is_global,
                )
            else:
                cube_object.set_external_force_and_torque(
                    external_wrench_b[..., :3],
                    external_wrench_b[..., 3:],
                    body_ids=body_ids,
                    is_global=is_global,
                )


            assert cube_object._external_force_b[0, 0, 0].item() == force
            assert cube_object._external_torque_b[0, 0, 0].item() == force
            assert cube_object._external_wrench_positions_b[0, 0, 0].item() == position
            assert cube_object._use_global_wrench_frame == (step == 0 or step == 3)


            cube_object.write_data_to_sim()


            sim.step()


            cube_object.update(sim.cfg.dt)


@pytest.mark.parametrize("num_cubes", [2, 4])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_external_force_on_single_body(num_cubes, device):
    """Test application of external force on the base of the object.

    In this test, we apply a force equal to the weight of an object on the base of
    one of the objects. We check that the object does not move. For the other object,
    we do not apply any force and check that it falls down.
    """

    with build_simulation_context(device=device, add_ground_plane=True, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        cube_object, origins = generate_cubes_scene(num_cubes=num_cubes, device=device)


        sim.reset()


        body_ids, body_names = cube_object.find_bodies(".*")


        external_wrench_b = torch.zeros(cube_object.num_instances, len(body_ids), 6, device=sim.device)

        external_wrench_b[0::2, :, 2] = 9.81 * cube_object.root_physx_view.get_masses()[0]


        for _ in range(5):

            root_state = cube_object.data.default_root_state.clone()


            root_state[:, :3] = origins
            cube_object.write_root_pose_to_sim(root_state[:, :7])
            cube_object.write_root_velocity_to_sim(root_state[:, 7:])


            cube_object.reset()


            cube_object.set_external_force_and_torque(
                external_wrench_b[..., :3], external_wrench_b[..., 3:], body_ids=body_ids
            )

            for _ in range(5):

                cube_object.write_data_to_sim()


                sim.step()


                cube_object.update(sim.cfg.dt)


            torch.testing.assert_close(
                cube_object.data.root_pos_w[0::2, 2], torch.ones(num_cubes // 2, device=sim.device)
            )

            assert torch.all(cube_object.data.root_pos_w[1::2, 2] < 1.0)


@pytest.mark.parametrize("num_cubes", [2, 4])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_external_force_on_single_body_at_position(num_cubes, device):
    """Test application of external force on the base of the object at a specific position.

    In this test, we apply a force equal to the weight of an object on the base of
    one of the objects at 1m in the Y direction, we check that the object rotates around it's X axis.
    For the other object, we do not apply any force and check that it falls down.
    """

    with build_simulation_context(device=device, add_ground_plane=True, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        cube_object, origins = generate_cubes_scene(num_cubes=num_cubes, device=device)


        sim.reset()


        body_ids, body_names = cube_object.find_bodies(".*")


        external_wrench_b = torch.zeros(cube_object.num_instances, len(body_ids), 6, device=sim.device)
        external_wrench_positions_b = torch.zeros(cube_object.num_instances, len(body_ids), 3, device=sim.device)

        external_wrench_b[0::2, :, 2] = 9.81 * cube_object.root_physx_view.get_masses()[0]
        external_wrench_positions_b[0::2, :, 1] = 1.0


        for _ in range(5):

            root_state = cube_object.data.default_root_state.clone()


            root_state[:, :3] = origins
            cube_object.write_root_pose_to_sim(root_state[:, :7])
            cube_object.write_root_velocity_to_sim(root_state[:, 7:])


            cube_object.reset()


            cube_object.set_external_force_and_torque(
                external_wrench_b[..., :3],
                external_wrench_b[..., 3:],
                positions=external_wrench_positions_b,
                body_ids=body_ids,
            )

            for _ in range(5):

                cube_object.write_data_to_sim()


                sim.step()


                cube_object.update(sim.cfg.dt)


            assert torch.all(torch.abs(cube_object.data.root_ang_vel_b[0::2, 0]) > 0.1)

            assert torch.all(cube_object.data.root_pos_w[1::2, 2] < 1.0)


@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_set_rigid_object_state(num_cubes, device):
    """Test setting the state of the rigid object.

    In this test, we set the state of the rigid object to a random state and check
    that the object is in that state after simulation. We set gravity to zero as
    we don't want any external forces acting on the object to ensure state remains static.
    """


    with build_simulation_context(device=device, gravity_enabled=False, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None

        cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, device=device)


        sim.reset()

        state_types = ["root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w"]


        for state_type_to_randomize in state_types:
            state_dict = {
                "root_pos_w": torch.zeros_like(cube_object.data.root_pos_w, device=sim.device),
                "root_quat_w": default_orientation(num=num_cubes, device=sim.device),
                "root_lin_vel_w": torch.zeros_like(cube_object.data.root_lin_vel_w, device=sim.device),
                "root_ang_vel_w": torch.zeros_like(cube_object.data.root_ang_vel_w, device=sim.device),
            }


            for _ in range(5):

                cube_object.reset()


                if state_type_to_randomize == "root_quat_w":
                    state_dict[state_type_to_randomize] = random_orientation(num=num_cubes, device=sim.device)
                else:
                    state_dict[state_type_to_randomize] = torch.randn(num_cubes, 3, device=sim.device)


                for _ in range(5):
                    root_state = torch.cat(
                        [
                            state_dict["root_pos_w"],
                            state_dict["root_quat_w"],
                            state_dict["root_lin_vel_w"],
                            state_dict["root_ang_vel_w"],
                        ],
                        dim=-1,
                    )

                    cube_object.write_root_pose_to_sim(root_state[:, :7])
                    cube_object.write_root_velocity_to_sim(root_state[:, 7:])

                    sim.step()


                    for key, expected_value in state_dict.items():
                        value = getattr(cube_object.data, key)
                        torch.testing.assert_close(value, expected_value, rtol=1e-5, atol=1e-5)

                    cube_object.update(sim.cfg.dt)


@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_reset_rigid_object(num_cubes, device):
    """Test resetting the state of the rigid object."""
    with build_simulation_context(device=device, gravity_enabled=True, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None

        cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, device=device)


        sim.reset()

        for i in range(5):

            sim.step()


            cube_object.update(sim.cfg.dt)


            root_state = cube_object.data.default_root_state.clone()
            root_state[:, :3] = torch.randn(num_cubes, 3, device=sim.device)


            root_state[:, 3:7] = random_orientation(num=num_cubes, device=sim.device)
            cube_object.write_root_pose_to_sim(root_state[:, :7])
            cube_object.write_root_velocity_to_sim(root_state[:, 7:])

            if i % 2 == 0:

                cube_object.reset()


                assert not cube_object.has_external_wrench
                assert torch.count_nonzero(cube_object._external_force_b) == 0
                assert torch.count_nonzero(cube_object._external_torque_b) == 0


@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_rigid_body_set_material_properties(num_cubes, device):
    """Test getting and setting material properties of rigid object."""
    with build_simulation_context(
        device=device, gravity_enabled=True, add_ground_plane=True, auto_add_lighting=True
    ) as sim:
        sim._app_control_on_stop_handle = None

        cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, device=device)


        sim.reset()


        static_friction = torch.FloatTensor(num_cubes, 1).uniform_(0.4, 0.8)
        dynamic_friction = torch.FloatTensor(num_cubes, 1).uniform_(0.4, 0.8)
        restitution = torch.FloatTensor(num_cubes, 1).uniform_(0.0, 0.2)

        materials = torch.cat([static_friction, dynamic_friction, restitution], dim=-1)

        indices = torch.tensor(range(num_cubes), dtype=torch.int)

        cube_object.root_physx_view.set_material_properties(materials, indices)



        sim.step()

        cube_object.update(sim.cfg.dt)


        materials_to_check = cube_object.root_physx_view.get_material_properties()


        torch.testing.assert_close(materials_to_check.reshape(num_cubes, 3), materials)


@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_rigid_body_no_friction(num_cubes, device):
    """Test that a rigid object with no friction will maintain it's velocity when sliding across a plane."""
    with build_simulation_context(device=device, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None

        cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, height=0.0, device=device)


        cfg = sim_utils.GroundPlaneCfg(
            physics_material=materials.RigidBodyMaterialCfg(
                static_friction=0.0,
                dynamic_friction=0.0,
                restitution=0.0,
            )
        )
        cfg.func("/World/GroundPlane", cfg)


        sim.reset()


        static_friction = torch.zeros(num_cubes, 1)
        dynamic_friction = torch.zeros(num_cubes, 1)
        restitution = torch.FloatTensor(num_cubes, 1).uniform_(0.0, 0.2)

        cube_object_materials = torch.cat([static_friction, dynamic_friction, restitution], dim=-1)
        indices = torch.tensor(range(num_cubes), dtype=torch.int)

        cube_object.root_physx_view.set_material_properties(cube_object_materials, indices)



        initial_velocity = torch.zeros((num_cubes, 6), device=sim.cfg.device)
        initial_velocity[:, 0] = 0.1

        cube_object.write_root_velocity_to_sim(initial_velocity)


        for _ in range(5):

            sim.step()

            cube_object.update(sim.cfg.dt)


            if device == "cuda:0":
                tolerance = 1e-2
            else:
                tolerance = 1e-5

            torch.testing.assert_close(
                cube_object.data.root_lin_vel_w, initial_velocity[:, :3], rtol=1e-5, atol=tolerance
            )


@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.isaacsim_ci
def test_rigid_body_with_static_friction(num_cubes, device):
    """Test that static friction applied to rigid object works as expected.

    This test works by applying a force to the object and checking if the object moves or not based on the
    mu (coefficient of static friction) value set for the object. We set the static friction to be non-zero and
    apply a force to the object. When the force applied is below mu, the object should not move. When the force
    applied is above mu, the object should move.
    """
    with build_simulation_context(device=device, dt=0.01, add_ground_plane=False, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, height=0.03125, device=device)


        static_friction_coefficient = 0.5
        cfg = sim_utils.GroundPlaneCfg(
            physics_material=materials.RigidBodyMaterialCfg(
                static_friction=static_friction_coefficient,
                dynamic_friction=static_friction_coefficient,
            )
        )
        cfg.func("/World/GroundPlane", cfg)


        sim.reset()



        static_friction = torch.Tensor([[static_friction_coefficient]] * num_cubes)
        dynamic_friction = torch.Tensor([[static_friction_coefficient]] * num_cubes)
        restitution = torch.zeros(num_cubes, 1)

        cube_object_materials = torch.cat([static_friction, dynamic_friction, restitution], dim=-1)

        indices = torch.tensor(range(num_cubes), dtype=torch.int)


        cube_object.root_physx_view.set_material_properties(cube_object_materials, indices)


        for _ in range(100):
            sim.step()
            cube_object.update(sim.cfg.dt)
        cube_object.write_root_velocity_to_sim(torch.zeros((num_cubes, 6), device=sim.device))
        cube_mass = cube_object.root_physx_view.get_masses()
        gravity_magnitude = abs(sim.cfg.gravity[2])



        for force in "below_mu", "above_mu":

            cube_object.write_root_velocity_to_sim(torch.zeros((num_cubes, 6), device=sim.device))

            external_wrench_b = torch.zeros((num_cubes, 1, 6), device=sim.device)
            if force == "below_mu":
                external_wrench_b[..., 0] = static_friction_coefficient * cube_mass * gravity_magnitude * 0.99
            else:
                external_wrench_b[..., 0] = static_friction_coefficient * cube_mass * gravity_magnitude * 1.01

            cube_object.set_external_force_and_torque(
                external_wrench_b[..., :3],
                external_wrench_b[..., 3:],
            )


            initial_root_pos = cube_object.data.root_pos_w.clone()

            for _ in range(200):

                cube_object.write_data_to_sim()
                sim.step()

                cube_object.update(sim.cfg.dt)
                if force == "below_mu":

                    torch.testing.assert_close(cube_object.data.root_pos_w, initial_root_pos, rtol=2e-3, atol=2e-3)
            if force == "above_mu":
                assert (cube_object.data.root_state_w[..., 0] - initial_root_pos[..., 0] > 0.02).all()


@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_rigid_body_with_restitution(num_cubes, device):
    """Test that restitution when applied to rigid object works as expected.

    This test works by dropping a block from a height and checking if the block bounces or not based on the
    restitution value set for the object. We set the restitution to be non-zero and drop the block from a height.
    When the restitution is 0, the block should not bounce. When the restitution is between 0 and 1, the block
    should bounce with less energy.
    """
    for expected_collision_type in "partially_elastic", "inelastic":
        with build_simulation_context(device=device, add_ground_plane=False, auto_add_lighting=True) as sim:
            sim._app_control_on_stop_handle = None
            cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, height=1.0, device=device)


            if expected_collision_type == "inelastic":
                restitution_coefficient = 0.0
            elif expected_collision_type == "partially_elastic":
                restitution_coefficient = 0.5


            cfg = sim_utils.GroundPlaneCfg(
                physics_material=materials.RigidBodyMaterialCfg(
                    restitution=restitution_coefficient,
                )
            )
            cfg.func("/World/GroundPlane", cfg)

            indices = torch.tensor(range(num_cubes), dtype=torch.int)


            sim.reset()

            root_state = torch.zeros(num_cubes, 13, device=sim.device)
            root_state[:, 3] = 1.0
            for i in range(num_cubes):
                root_state[i, 1] = 1.0 * i
            root_state[:, 2] = 1.0
            root_state[:, 9] = -1.0

            cube_object.write_root_pose_to_sim(root_state[:, :7])
            cube_object.write_root_velocity_to_sim(root_state[:, 7:])

            static_friction = torch.zeros(num_cubes, 1)
            dynamic_friction = torch.zeros(num_cubes, 1)
            restitution = torch.Tensor([[restitution_coefficient]] * num_cubes)

            cube_object_materials = torch.cat([static_friction, dynamic_friction, restitution], dim=-1)


            cube_object.root_physx_view.set_material_properties(cube_object_materials, indices)

            curr_z_velocity = cube_object.data.root_lin_vel_w[:, 2].clone()

            for _ in range(100):
                sim.step()


                cube_object.update(sim.cfg.dt)
                curr_z_velocity = cube_object.data.root_lin_vel_w[:, 2].clone()

                if expected_collision_type == "inelastic":

                    assert (curr_z_velocity <= 0.0).all()

                if torch.all(curr_z_velocity <= 0.0):

                    prev_z_velocity = curr_z_velocity
                else:

                    break

            if expected_collision_type == "partially_elastic":

                assert torch.all(torch.le(abs(curr_z_velocity), abs(prev_z_velocity)))
                assert (curr_z_velocity > 0.0).all()


@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_rigid_body_set_mass(num_cubes, device):
    """Test getting and setting mass of rigid object."""
    with build_simulation_context(
        device=device, gravity_enabled=False, add_ground_plane=True, auto_add_lighting=True
    ) as sim:
        sim._app_control_on_stop_handle = None

        cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, height=1.0, device=device)


        sim.reset()


        original_masses = cube_object.root_physx_view.get_masses()

        assert original_masses.shape == (num_cubes, 1)


        masses = original_masses + torch.FloatTensor(num_cubes, 1).uniform_(4, 8)

        indices = torch.tensor(range(num_cubes), dtype=torch.int)


        cube_object.root_physx_view.set_masses(masses, indices)

        torch.testing.assert_close(cube_object.root_physx_view.get_masses(), masses)



        sim.step()

        cube_object.update(sim.cfg.dt)

        masses_to_check = cube_object.root_physx_view.get_masses()


        torch.testing.assert_close(masses, masses_to_check)


@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("gravity_enabled", [True, False])
@pytest.mark.isaacsim_ci
def test_gravity_vec_w(num_cubes, device, gravity_enabled):
    """Test that gravity vector direction is set correctly for the rigid object."""
    with build_simulation_context(device=device, gravity_enabled=gravity_enabled) as sim:
        sim._app_control_on_stop_handle = None

        cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, device=device)


        if gravity_enabled:
            gravity_dir = (0.0, 0.0, -1.0)
        else:
            gravity_dir = (0.0, 0.0, 0.0)


        sim.reset()


        assert cube_object.data.GRAVITY_VEC_W[0, 0] == gravity_dir[0]
        assert cube_object.data.GRAVITY_VEC_W[0, 1] == gravity_dir[1]
        assert cube_object.data.GRAVITY_VEC_W[0, 2] == gravity_dir[2]


        for _ in range(2):

            sim.step()

            cube_object.update(sim.cfg.dt)


            gravity = torch.zeros(num_cubes, 1, 6, device=device)
            if gravity_enabled:
                gravity[:, :, 2] = -9.81

            torch.testing.assert_close(cube_object.data.body_acc_w, gravity)


@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("with_offset", [True, False])
@pytest.mark.isaacsim_ci
@flaky(max_runs=3, min_passes=1)
def test_body_root_state_properties(num_cubes, device, with_offset):
    """Test the root_com_state_w, root_link_state_w, body_com_state_w, and body_link_state_w properties."""
    with build_simulation_context(device=device, gravity_enabled=False, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None

        cube_object, env_pos = generate_cubes_scene(num_cubes=num_cubes, height=0.0, device=device)
        env_idx = torch.tensor([x for x in range(num_cubes)])


        sim.reset()


        assert cube_object.is_initialized


        if with_offset:
            offset = torch.tensor([0.1, 0.0, 0.0], device=device).repeat(num_cubes, 1)
        else:
            offset = torch.tensor([0.0, 0.0, 0.0], device=device).repeat(num_cubes, 1)

        com = cube_object.root_physx_view.get_coms()
        com[..., :3] = offset.to("cpu")
        cube_object.root_physx_view.set_coms(com, env_idx)


        torch.testing.assert_close(cube_object.root_physx_view.get_coms(), com)


        spin_twist = torch.zeros(6, device=device)
        spin_twist[5] = torch.randn(1, device=device)


        for _ in range(100):

            cube_object.write_root_velocity_to_sim(spin_twist.repeat(num_cubes, 1))

            sim.step()

            cube_object.update(sim.cfg.dt)


            root_state_w = cube_object.data.root_state_w
            root_link_state_w = cube_object.data.root_link_state_w
            root_com_state_w = cube_object.data.root_com_state_w
            body_state_w = cube_object.data.body_state_w
            body_link_state_w = cube_object.data.body_link_state_w
            body_com_state_w = cube_object.data.body_com_state_w


            if not with_offset:
                torch.testing.assert_close(root_state_w, root_com_state_w)
                torch.testing.assert_close(root_state_w, root_link_state_w)
                torch.testing.assert_close(body_state_w, body_com_state_w)
                torch.testing.assert_close(body_state_w, body_link_state_w)
            else:



                torch.testing.assert_close(env_pos + offset, root_com_state_w[..., :3])
                torch.testing.assert_close(env_pos + offset, body_com_state_w[..., :3].squeeze(-2))

                root_link_state_pos_rel_com = quat_apply_inverse(
                    root_link_state_w[..., 3:7],
                    root_link_state_w[..., :3] - root_com_state_w[..., :3],
                )
                torch.testing.assert_close(-offset, root_link_state_pos_rel_com)
                body_link_state_pos_rel_com = quat_apply_inverse(
                    body_link_state_w[..., 3:7],
                    body_link_state_w[..., :3] - body_com_state_w[..., :3],
                )
                torch.testing.assert_close(-offset, body_link_state_pos_rel_com.squeeze(-2))


                com_quat_b = cube_object.data.body_com_quat_b
                com_quat_w = quat_mul(body_link_state_w[..., 3:7], com_quat_b)
                torch.testing.assert_close(com_quat_w, body_com_state_w[..., 3:7])
                torch.testing.assert_close(com_quat_w.squeeze(-2), root_com_state_w[..., 3:7])


                torch.testing.assert_close(root_state_w[..., 3:7], root_link_state_w[..., 3:7])
                torch.testing.assert_close(body_state_w[..., 3:7], body_link_state_w[..., 3:7])



                torch.testing.assert_close(torch.zeros_like(root_com_state_w[..., 7:10]), root_com_state_w[..., 7:10])
                torch.testing.assert_close(torch.zeros_like(body_com_state_w[..., 7:10]), body_com_state_w[..., 7:10])

                lin_vel_rel_root_gt = quat_apply_inverse(root_link_state_w[..., 3:7], root_link_state_w[..., 7:10])
                lin_vel_rel_body_gt = quat_apply_inverse(body_link_state_w[..., 3:7], body_link_state_w[..., 7:10])
                lin_vel_rel_gt = torch.linalg.cross(spin_twist.repeat(num_cubes, 1)[..., 3:], -offset)
                torch.testing.assert_close(lin_vel_rel_gt, lin_vel_rel_root_gt, atol=1e-4, rtol=1e-4)
                torch.testing.assert_close(lin_vel_rel_gt, lin_vel_rel_body_gt.squeeze(-2), atol=1e-4, rtol=1e-4)


                torch.testing.assert_close(root_state_w[..., 10:], root_com_state_w[..., 10:])
                torch.testing.assert_close(root_state_w[..., 10:], root_link_state_w[..., 10:])
                torch.testing.assert_close(body_state_w[..., 10:], body_com_state_w[..., 10:])
                torch.testing.assert_close(body_state_w[..., 10:], body_link_state_w[..., 10:])


@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("with_offset", [True, False])
@pytest.mark.parametrize("state_location", ["com", "link"])
@pytest.mark.isaacsim_ci
def test_write_root_state(num_cubes, device, with_offset, state_location):
    """Test the setters for root_state using both the link frame and center of mass as reference frame."""
    with build_simulation_context(device=device, gravity_enabled=False, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None

        cube_object, env_pos = generate_cubes_scene(num_cubes=num_cubes, height=0.0, device=device)
        env_idx = torch.tensor([x for x in range(num_cubes)])


        sim.reset()


        assert cube_object.is_initialized


        if with_offset:
            offset = torch.tensor([0.1, 0.0, 0.0], device=device).repeat(num_cubes, 1)
        else:
            offset = torch.tensor([0.0, 0.0, 0.0], device=device).repeat(num_cubes, 1)

        com = cube_object.root_physx_view.get_coms()
        com[..., :3] = offset.to("cpu")
        cube_object.root_physx_view.set_coms(com, env_idx)


        torch.testing.assert_close(cube_object.root_physx_view.get_coms(), com)

        rand_state = torch.zeros_like(cube_object.data.root_state_w)
        rand_state[..., :7] = cube_object.data.default_root_state[..., :7]
        rand_state[..., :3] += env_pos

        rand_state[..., 3:7] = torch.nn.functional.normalize(rand_state[..., 3:7], dim=-1)

        env_idx = env_idx.to(device)
        for i in range(10):


            sim.step()

            cube_object.update(sim.cfg.dt)

            if state_location == "com":
                if i % 2 == 0:
                    cube_object.write_root_com_state_to_sim(rand_state)
                else:
                    cube_object.write_root_com_state_to_sim(rand_state, env_ids=env_idx)
            elif state_location == "link":
                if i % 2 == 0:
                    cube_object.write_root_link_state_to_sim(rand_state)
                else:
                    cube_object.write_root_link_state_to_sim(rand_state, env_ids=env_idx)

            if state_location == "com":
                torch.testing.assert_close(rand_state, cube_object.data.root_com_state_w)
            elif state_location == "link":
                torch.testing.assert_close(rand_state, cube_object.data.root_link_state_w)


@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("with_offset", [True])
@pytest.mark.parametrize("state_location", ["com", "link", "root"])
@pytest.mark.isaacsim_ci
def test_write_state_functions_data_consistency(num_cubes, device, with_offset, state_location):
    """Test the setters for root_state using both the link frame and center of mass as reference frame."""
    with build_simulation_context(device=device, gravity_enabled=False, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None

        cube_object, env_pos = generate_cubes_scene(num_cubes=num_cubes, height=0.0, device=device)
        env_idx = torch.tensor([x for x in range(num_cubes)])


        sim.reset()


        assert cube_object.is_initialized


        if with_offset:
            offset = torch.tensor([0.1, 0.0, 0.0], device=device).repeat(num_cubes, 1)
        else:
            offset = torch.tensor([0.0, 0.0, 0.0], device=device).repeat(num_cubes, 1)

        com = cube_object.root_physx_view.get_coms()
        com[..., :3] = offset.to("cpu")
        cube_object.root_physx_view.set_coms(com, env_idx)


        torch.testing.assert_close(cube_object.root_physx_view.get_coms(), com)

        rand_state = torch.rand_like(cube_object.data.root_state_w)

        rand_state[..., :3] += env_pos

        rand_state[..., 3:7] = torch.nn.functional.normalize(rand_state[..., 3:7], dim=-1)

        env_idx = env_idx.to(device)


        sim.step()

        cube_object.update(sim.cfg.dt)

        if state_location == "com":
            cube_object.write_root_com_state_to_sim(rand_state)
        elif state_location == "link":
            cube_object.write_root_link_state_to_sim(rand_state)
        elif state_location == "root":
            cube_object.write_root_state_to_sim(rand_state)

        if state_location == "com":
            expected_root_link_pos, expected_root_link_quat = combine_frame_transforms(
                cube_object.data.root_com_state_w[:, :3],
                cube_object.data.root_com_state_w[:, 3:7],
                quat_rotate(
                    quat_inv(cube_object.data.body_com_pose_b[:, 0, 3:7]), -cube_object.data.body_com_pose_b[:, 0, :3]
                ),
                quat_inv(cube_object.data.body_com_pose_b[:, 0, 3:7]),
            )
            expected_root_link_pose = torch.cat((expected_root_link_pos, expected_root_link_quat), dim=1)

            torch.testing.assert_close(expected_root_link_pose, cube_object.data.root_link_state_w[:, :7])


            torch.testing.assert_close(
                cube_object.data.root_com_state_w[:, 10:], cube_object.data.root_link_state_w[:, 10:]
            )
            torch.testing.assert_close(expected_root_link_pose, cube_object.data.root_state_w[:, :7])
            torch.testing.assert_close(cube_object.data.root_com_state_w[:, 10:], cube_object.data.root_state_w[:, 10:])
        elif state_location == "link":
            expected_com_pos, expected_com_quat = combine_frame_transforms(
                cube_object.data.root_link_state_w[:, :3],
                cube_object.data.root_link_state_w[:, 3:7],
                cube_object.data.body_com_pose_b[:, 0, :3],
                cube_object.data.body_com_pose_b[:, 0, 3:7],
            )
            expected_com_pose = torch.cat((expected_com_pos, expected_com_quat), dim=1)

            torch.testing.assert_close(expected_com_pose, cube_object.data.root_com_state_w[:, :7])


            torch.testing.assert_close(
                cube_object.data.root_link_state_w[:, 10:], cube_object.data.root_com_state_w[:, 10:]
            )
            torch.testing.assert_close(cube_object.data.root_link_state_w[:, :7], cube_object.data.root_state_w[:, :7])
            torch.testing.assert_close(
                cube_object.data.root_link_state_w[:, 10:], cube_object.data.root_state_w[:, 10:]
            )
        elif state_location == "root":
            expected_com_pos, expected_com_quat = combine_frame_transforms(
                cube_object.data.root_state_w[:, :3],
                cube_object.data.root_state_w[:, 3:7],
                cube_object.data.body_com_pose_b[:, 0, :3],
                cube_object.data.body_com_pose_b[:, 0, 3:7],
            )
            expected_com_pose = torch.cat((expected_com_pos, expected_com_quat), dim=1)

            torch.testing.assert_close(expected_com_pose, cube_object.data.root_com_state_w[:, :7])
            torch.testing.assert_close(cube_object.data.root_state_w[:, 7:], cube_object.data.root_com_state_w[:, 7:])
            torch.testing.assert_close(cube_object.data.root_state_w[:, :7], cube_object.data.root_link_state_w[:, :7])
            torch.testing.assert_close(
                cube_object.data.root_state_w[:, 10:], cube_object.data.root_link_state_w[:, 10:]
            )
