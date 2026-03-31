




"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher


simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import numpy as np
import torch
import trimesh
from typing import Literal

import isaacsim.core.utils.prims as prim_utils
import omni.kit
import omni.kit.commands
import pytest
from isaacsim.core.api.materials import PhysicsMaterial, PreviewSurface
from isaacsim.core.api.objects import DynamicSphere
from isaacsim.core.cloner import GridCloner
from isaacsim.core.prims import RigidPrim, SingleGeometryPrim, SingleRigidPrim
from isaacsim.core.utils.extensions import enable_extension
from pxr import UsdGeom

import isaaclab.terrains as terrain_gen
from isaaclab.sim import PreviewSurfaceCfg, SimulationContext, build_simulation_context, get_first_matching_child_prim
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("env_spacing", [1.0, 4.325, 8.0])
@pytest.mark.parametrize("num_envs", [1, 4, 125, 379, 1024])
def test_grid_clone_env_origins(device, env_spacing, num_envs):
    """Tests that env origins are consistent when computed using the TerrainImporter and IsaacSim GridCloner."""
    with build_simulation_context(device=device, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None

        terrain_importer_cfg = TerrainImporterCfg(
            num_envs=num_envs,
            env_spacing=env_spacing,
            prim_path="/World/ground",
            terrain_type="plane",
            terrain_generator=None,
        )
        terrain_importer = TerrainImporter(terrain_importer_cfg)

        terrain_importer_origins = terrain_importer.env_origins


        grid_cloner_origins = _obtain_grid_cloner_env_origins(num_envs, env_spacing, device=sim.device)


        torch.testing.assert_close(terrain_importer_origins, grid_cloner_origins, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_terrain_generation(device):
    """Generates assorted terrains and tests that the resulting mesh has the correct size."""
    with build_simulation_context(device=device, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None

        terrain_importer_cfg = terrain_gen.TerrainImporterCfg(
            prim_path="/World/ground",
            max_init_terrain_level=None,
            terrain_type="generator",
            terrain_generator=ROUGH_TERRAINS_CFG,
            num_envs=1,
        )
        terrain_importer = TerrainImporter(terrain_importer_cfg)


        mesh_prim_path = terrain_importer.cfg.prim_path + "/terrain"
        assert mesh_prim_path in terrain_importer.terrain_prim_paths


        mesh = _obtain_collision_mesh(mesh_prim_path, mesh_type="Mesh")
        assert mesh is not None


        cfg = terrain_importer.cfg.terrain_generator
        assert cfg is not None
        expectedSizeX = cfg.size[0] * cfg.num_rows + 2 * cfg.border_width
        expectedSizeY = cfg.size[1] * cfg.num_cols + 2 * cfg.border_width


        bounds = mesh.bounds
        actualSize = abs(bounds[1] - bounds[0])

        assert actualSize[0] == pytest.approx(expectedSizeX)
        assert actualSize[1] == pytest.approx(expectedSizeY)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("use_custom_material", [True, False])
def test_plane(device, use_custom_material):
    """Generates a plane and tests that the resulting mesh has the correct size."""
    with build_simulation_context(device=device, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None


        visual_material = PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)) if use_custom_material else None

        terrain_importer_cfg = terrain_gen.TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            num_envs=1,
            env_spacing=1.0,
            visual_material=visual_material,
        )
        terrain_importer = TerrainImporter(terrain_importer_cfg)


        mesh_prim_path = terrain_importer.cfg.prim_path + "/terrain"
        assert mesh_prim_path in terrain_importer.terrain_prim_paths


        mesh = _obtain_collision_mesh(mesh_prim_path, mesh_type="Plane")
        assert mesh is None


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_usd(device):
    """Imports terrain from a usd and tests that the resulting mesh has the correct size."""
    with build_simulation_context(device=device, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None

        terrain_importer_cfg = terrain_gen.TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="usd",
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/rough_plane.usd",
            num_envs=1,
            env_spacing=1.0,
        )
        terrain_importer = TerrainImporter(terrain_importer_cfg)


        mesh_prim_path = terrain_importer.cfg.prim_path + "/terrain"
        assert mesh_prim_path in terrain_importer.terrain_prim_paths


        mesh = _obtain_collision_mesh(mesh_prim_path, mesh_type="Mesh")
        assert mesh is not None


        expectedSizeX = 96
        expectedSizeY = 96


        bounds = mesh.bounds
        actualSize = abs(bounds[1] - bounds[0])

        assert actualSize[0] == pytest.approx(expectedSizeX)
        assert actualSize[1] == pytest.approx(expectedSizeY)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_ball_drop(device):
    """Generates assorted terrains and spheres created as meshes.

    Tests that spheres fall onto terrain and do not pass through it. This ensures that the triangle mesh
    collision works as expected.
    """
    with build_simulation_context(device=device, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None

        _populate_scene(geom_sphere=False, sim=sim)


        ball_view = RigidPrim("/World/envs/env_.*/ball", reset_xform_properties=False)


        sim.reset()

        ball_view.initialize()


        for _ in range(500):
            sim.step(render=False)



        max_velocity_z = torch.max(torch.abs(ball_view.get_linear_velocities()[:, 2]))
        assert max_velocity_z.item() <= 0.5


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_ball_drop_geom_sphere(device):
    """Generates assorted terrains and geom spheres.

    Tests that spheres fall onto terrain and do not pass through it. This ensures that the sphere collision
    works as expected.
    """
    with build_simulation_context(device=device, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None




        _populate_scene(geom_sphere=False, sim=sim)


        ball_view = RigidPrim("/World/envs/env_.*/ball", reset_xform_properties=False)


        sim.reset()

        ball_view.initialize()


        for _ in range(500):
            sim.step(render=False)



        max_velocity_z = torch.max(torch.abs(ball_view.get_linear_velocities()[:, 2]))
        assert max_velocity_z.item() <= 0.5


def _obtain_collision_mesh(mesh_prim_path: str, mesh_type: Literal["Mesh", "Plane"]) -> trimesh.Trimesh | None:
    """Get the collision mesh from the terrain."""

    mesh_prim = get_first_matching_child_prim(mesh_prim_path, lambda prim: prim.GetTypeName() == mesh_type)

    assert mesh_prim.IsValid()

    if mesh_prim.GetTypeName() == "Mesh":

        mesh_prim = UsdGeom.Mesh(mesh_prim)

        vertices = np.asarray(mesh_prim.GetPointsAttr().Get())
        faces = np.asarray(mesh_prim.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        return trimesh.Trimesh(vertices=vertices, faces=faces)
    else:
        return None


def _obtain_grid_cloner_env_origins(num_envs: int, env_spacing: float, device: str) -> torch.Tensor:
    """Obtain the env origins generated by IsaacSim GridCloner (grid_cloner.py)."""

    cloner = GridCloner(spacing=env_spacing)
    cloner.define_base_env("/World/envs")
    envs_prim_paths = cloner.generate_paths("/World/envs/env", num_paths=num_envs)
    prim_utils.define_prim("/World/envs/env_0")

    env_origins = cloner.clone(source_prim_path="/World/envs/env_0", prim_paths=envs_prim_paths, replicate_physics=True)

    return torch.tensor(env_origins, dtype=torch.float32, device=device)


def _populate_scene(sim: SimulationContext, num_balls: int = 2048, geom_sphere: bool = False):
    """Create a scene with terrain and randomly spawned balls.

    The spawned balls are either USD Geom Spheres or are USD Meshes. We check against both these to make sure
    both USD-shape and USD-mesh collisions work as expected.
    """

    terrain_importer_cfg = terrain_gen.TerrainImporterCfg(
        prim_path="/World/ground",
        max_init_terrain_level=None,
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        num_envs=num_balls,
    )
    terrain_importer = TerrainImporter(terrain_importer_cfg)


    cloner = GridCloner(spacing=2.0)
    cloner.define_base_env("/World/envs")

    prim_utils.define_prim(prim_path="/World/envs/env_0", prim_type="Xform")



    if geom_sphere:

        _ = DynamicSphere(
            prim_path="/World/envs/env_0/ball", translation=np.array([0.0, 0.0, 5.0]), mass=0.5, radius=0.25
        )
    else:

        enable_extension("omni.kit.primitive.mesh")
        cube_prim_path = omni.kit.commands.execute("CreateMeshPrimCommand", prim_type="Sphere")[1]
        prim_utils.move_prim(cube_prim_path, "/World/envs/env_0/ball")

        SingleRigidPrim(
            prim_path="/World/envs/env_0/ball", mass=0.5, scale=(0.5, 0.5, 0.5), translation=(0.0, 0.0, 0.5)
        )
        SingleGeometryPrim(prim_path="/World/envs/env_0/ball", collision=True)


    sphere_geom = SingleGeometryPrim(prim_path="/World/envs/env_0/ball", collision=True)
    visual_material = PreviewSurface(prim_path="/World/Looks/ballColorMaterial", color=np.asarray([0.0, 0.0, 1.0]))
    physics_material = PhysicsMaterial(
        prim_path="/World/Looks/ballPhysicsMaterial",
        dynamic_friction=1.0,
        static_friction=0.2,
        restitution=0.0,
    )
    sphere_geom.set_collision_approximation("convexHull")
    sphere_geom.apply_visual_material(visual_material)
    sphere_geom.apply_physics_material(physics_material)


    cloner.define_base_env("/World/envs")
    envs_prim_paths = cloner.generate_paths("/World/envs/env", num_paths=num_balls)
    cloner.clone(
        source_prim_path="/World/envs/env_0",
        prim_paths=envs_prim_paths,
        replicate_physics=True,
    )
    physics_scene_path = sim.get_physics_context().prim_path
    cloner.filter_collisions(
        physics_scene_path, "/World/collisions", prim_paths=envs_prim_paths, global_paths=["/World/ground"]
    )



    ball_view = RigidPrim("/World/envs/env_.*/ball", reset_xform_properties=False)

    ball_initial_positions = terrain_importer.env_origins
    ball_initial_positions[:, 2] += 5.0


    ball_view.set_world_poses(positions=ball_initial_positions)
