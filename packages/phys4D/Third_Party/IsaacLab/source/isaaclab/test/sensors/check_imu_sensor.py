




"""
Visual test script for the imu sensor from the Orbit framework.
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse

from isaacsim import SimulationApp


parser = argparse.ArgumentParser(description="Imu Test Script")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--num_envs", type=int, default=128, help="Number of environments to clone.")
parser.add_argument(
    "--terrain_type",
    type=str,
    default="generator",
    choices=["generator", "usd", "plane"],
    help="Type of terrain to import. Can be 'generator' or 'usd' or 'plane'.",
)
args_cli = parser.parse_args()


config = {"headless": args_cli.headless}
simulation_app = SimulationApp(config)


"""Rest everything follows."""

import torch
import traceback

import carb
import omni
from isaacsim.core.api.simulation_context import SimulationContext
from isaacsim.core.cloner import GridCloner
from isaacsim.core.utils.viewports import set_camera_view
from pxr import PhysxSchema

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sensors.imu import Imu, ImuCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.terrains.terrain_importer import TerrainImporter
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.timer import Timer


def design_scene(sim: SimulationContext, num_envs: int = 2048) -> RigidObject:
    """Design the scene."""

    terrain_importer_cfg = terrain_gen.TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/rough_plane.usd",
        max_init_terrain_level=None,
        num_envs=1,
    )
    _ = TerrainImporter(terrain_importer_cfg)

    stage = omni.usd.get_context().get_stage()

    cloner = GridCloner(spacing=2.0)
    cloner.define_base_env("/World/envs")
    envs_prim_paths = cloner.generate_paths("/World/envs/env", num_paths=num_envs)

    stage.DefinePrim(envs_prim_paths[0], "Xform")

    cloner.clone(source_prim_path="/World/envs/env_0", prim_paths=envs_prim_paths, replicate_physics=True)


    cfg = sim_utils.DistantLightCfg(intensity=2000)
    cfg.func("/World/light", cfg)

    cfg = RigidObjectCfg(
        spawn=sim_utils.SphereCfg(
            radius=0.25,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
        prim_path="/World/envs/env_.*/ball",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 5.0)),
    )
    balls = RigidObject(cfg)


    physics_scene_prim_path = None
    for prim in stage.Traverse():
        if prim.HasAPI(PhysxSchema.PhysxSceneAPI):
            physics_scene_prim_path = prim.GetPrimPath()
            carb.log_info(f"Physics scene prim path: {physics_scene_prim_path}")
            break

    cloner.filter_collisions(
        physics_scene_prim_path,
        "/World/collisions",
        envs_prim_paths,
    )
    return balls


def main():
    """Main function."""


    sim_params = {
        "use_gpu": True,
        "use_gpu_pipeline": True,
        "use_flatcache": True,
        "use_fabric": True,
        "enable_scene_query_support": True,
    }
    sim = SimulationContext(
        physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0, sim_params=sim_params, backend="torch", device="cuda:0"
    )

    set_camera_view([0.0, 30.0, 25.0], [0.0, 0.0, -2.5])


    num_envs = args_cli.num_envs

    balls = design_scene(sim=sim, num_envs=num_envs)


    imu_cfg = ImuCfg(
        prim_path="/World/envs/env_.*/ball",
        debug_vis=not args_cli.headless,
    )

    imu_cfg.visualizer_cfg.markers["arrow"].scale = (1.0, 0.2, 0.2)
    imu = Imu(cfg=imu_cfg)


    sim.reset()


    print(imu)


    sim.step(render=not args_cli.headless)
    balls.update(sim.get_physics_dt())
    ball_initial_positions = balls.data.root_pos_w.clone()
    ball_initial_orientations = balls.data.root_quat_w.clone()


    step_count = 0

    while simulation_app.is_running():

        if sim.is_stopped():
            break

        if not sim.is_playing():
            sim.step(render=not args_cli.headless)
            continue

        if step_count % 500 == 0:

            balls.write_root_pose_to_sim(torch.cat([ball_initial_positions, ball_initial_orientations], dim=-1))
            balls.reset()

            imu.reset()

            step_count = 0

        sim.step()

        with Timer(f"Imu sensor update with {num_envs}"):
            imu.update(dt=sim.get_physics_dt(), force_recompute=True)

        step_count += 1


if __name__ == "__main__":
    try:

        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:

        simulation_app.close()
