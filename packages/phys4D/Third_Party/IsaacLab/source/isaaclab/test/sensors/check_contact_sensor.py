




"""
This script demonstrates how to use the contact sensor sensor in Isaac Lab.

.. code-block:: bash

    ./isaaclab.sh -p source/isaaclab/test/sensors/test_contact_sensor.py --num_robots 2
"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Contact Sensor Test Script")
parser.add_argument("--num_robots", type=int, default=128, help="Number of robots to spawn.")


AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""

import torch

import isaacsim.core.utils.prims as prim_utils
from isaacsim.core.api.simulation_context import SimulationContext
from isaacsim.core.cloner import GridCloner
from isaacsim.core.utils.carb import set_carb_setting
from isaacsim.core.utils.viewports import set_camera_view

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors.contact_sensor import ContactSensor, ContactSensorCfg
from isaaclab.utils.timer import Timer




from isaaclab_assets.robots.anymal import ANYMAL_C_CFG


"""
Helpers
"""


def design_scene():
    """Add prims to the scene."""

    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    cfg = sim_utils.DomeLightCfg(intensity=2000)
    cfg.func("/World/Light/DomeLight", cfg, translation=(-4.5, 3.5, 10.0))


"""
Main
"""


def main():
    """Spawns the ANYmal robot and clones it using Isaac Sim Cloner API."""


    sim = SimulationContext(physics_dt=0.005, rendering_dt=0.005, backend="torch", device="cuda:0")

    set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])



    set_carb_setting(sim._settings, "/persistent/omnihydra/useSceneGraphInstancing", True)


    cloner = GridCloner(spacing=2.0)
    cloner.define_base_env("/World/envs")

    prim_utils.define_prim("/World/envs/env_0")

    num_envs = args_cli.num_robots
    cloner.define_base_env("/World/envs")
    envs_prim_paths = cloner.generate_paths("/World/envs/env", num_paths=num_envs)
    _ = cloner.clone(source_prim_path="/World/envs/env_0", prim_paths=envs_prim_paths, replicate_physics=True)

    design_scene()

    robot_cfg = ANYMAL_C_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot_cfg.spawn.activate_contact_sensors = True
    robot = Articulation(cfg=robot_cfg)

    contact_sensor_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*_FOOT",
        track_air_time=True,
        track_contact_points=True,
        debug_vis=False,
        filter_prim_paths_expr=["/World/defaultGroundPlane/GroundPlane/CollisionPlane"],
    )
    contact_sensor = ContactSensor(cfg=contact_sensor_cfg)

    physics_scene_path = sim.get_physics_context().prim_path
    cloner.filter_collisions(
        physics_scene_path, "/World/collisions", envs_prim_paths, global_paths=["/World/defaultGroundPlane"]
    )


    sim.reset()

    print(contact_sensor)


    print("[INFO]: Setup complete...")


    decimation = 4
    physics_dt = sim.get_physics_dt()
    sim_dt = decimation * physics_dt
    sim_time = 0.0
    count = 0
    dt = []

    while simulation_app.is_running():

        if sim.is_stopped():
            break

        if not sim.is_playing():
            sim.step(render=False)
            continue

        if count % 1000 == 0 and count != 0:

            sim_time = 0.0
            count = 0
            print("=" * 80)
            print("avg dt real-time", sum(dt) / len(dt))
            print("=" * 80)


            joint_pos, joint_vel = robot.data.default_joint_pos, robot.data.default_joint_vel
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            dt = []


        for _ in range(decimation):

            robot.set_joint_position_target(robot.data.default_joint_pos)

            robot.write_data_to_sim()

            sim.step()

            robot.update(physics_dt)

        sim_time += sim_dt
        count += 1

        if sim.is_playing():
            with Timer() as timer:
                contact_sensor.update(sim_dt, force_recompute=True)
                dt.append(timer.time_elapsed)

            contact_sensor.update(sim_dt, force_recompute=True)
            if count % 100 == 0:
                print("Sim-time: ", sim_time)
                print("Number of contacts: ", torch.count_nonzero(contact_sensor.data.current_air_time == 0.0).item())
                print("-" * 80)


if __name__ == "__main__":

    main()

    simulation_app.close()
