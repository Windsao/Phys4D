




"""
This script demonstrates how to use the ray-caster sensor.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/04_sensors/run_ray_caster.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Ray Caster Test Script")

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sensors.ray_caster import RayCaster, RayCasterCfg, patterns
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.timer import Timer


def define_sensor() -> RayCaster:
    """Defines the ray-caster sensor to add to the scene."""

    ray_caster_cfg = RayCasterCfg(
        prim_path="/World/Origin.*/ball",
        mesh_prim_paths=["/World/ground"],
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(2.0, 2.0)),
        ray_alignment="yaw",
        debug_vis=not args_cli.headless,
    )
    ray_caster = RayCaster(cfg=ray_caster_cfg)

    return ray_caster


def design_scene() -> dict:
    """Design the scene."""


    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/rough_plane.usd")
    cfg.func("/World/ground", cfg)

    cfg = sim_utils.DistantLightCfg(intensity=2000)
    cfg.func("/World/light", cfg)



    origins = [[0.25, 0.25, 0.0], [-0.25, 0.25, 0.0], [0.25, -0.25, 0.0], [-0.25, -0.25, 0.0]]
    for i, origin in enumerate(origins):
        prim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=origin)

    cfg = RigidObjectCfg(
        prim_path="/World/Origin.*/ball",
        spawn=sim_utils.SphereCfg(
            radius=0.25,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
    )
    balls = RigidObject(cfg)

    ray_caster = define_sensor()


    scene_entities = {"balls": balls, "ray_caster": ray_caster}
    return scene_entities


def run_simulator(sim: sim_utils.SimulationContext, scene_entities: dict):
    """Run the simulator."""

    ray_caster: RayCaster = scene_entities["ray_caster"]
    balls: RigidObject = scene_entities["balls"]


    ball_default_state = balls.data.default_root_state.clone()
    ball_default_state[:, :3] = torch.rand_like(ball_default_state[:, :3]) * 10


    step_count = 0

    while simulation_app.is_running():

        if step_count % 250 == 0:

            balls.write_root_pose_to_sim(ball_default_state[:, :7])
            balls.write_root_velocity_to_sim(ball_default_state[:, 7:])

            ray_caster.reset()

            step_count = 0

        sim.step()

        with Timer(
            f"Ray-caster update with {4} x {ray_caster.num_rays} rays with max height of"
            f" {torch.max(ray_caster.data.pos_w).item():.2f}"
        ):
            ray_caster.update(dt=sim.get_physics_dt(), force_recompute=True)

        step_count += 1


def main():
    """Main function."""

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view([0.0, 15.0, 15.0], [0.0, 0.0, -2.5])

    scene_entities = design_scene()

    sim.reset()

    print("[INFO]: Setup complete...")

    run_simulator(sim=sim, scene_entities=scene_entities)


if __name__ == "__main__":

    main()

    simulation_app.close()
