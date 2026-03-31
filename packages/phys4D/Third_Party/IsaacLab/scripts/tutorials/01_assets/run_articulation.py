




"""This script demonstrates how to spawn a cart-pole and interact with it.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/01_assets/run_articulation.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with an articulation.")

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext




from isaaclab_assets import CARTPOLE_CFG


def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""

    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)



    origins = [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]

    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])

    prim_utils.create_prim("/World/Origin2", "Xform", translation=origins[1])


    cartpole_cfg = CARTPOLE_CFG.copy()
    cartpole_cfg.prim_path = "/World/Origin.*/Robot"
    cartpole = Articulation(cfg=cartpole_cfg)


    scene_entities = {"cartpole": cartpole}
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""



    robot = entities["cartpole"]

    sim_dt = sim.get_physics_dt()
    count = 0

    while simulation_app.is_running():

        if count % 500 == 0:

            count = 0




            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])

            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_state_to_sim(joint_pos, joint_vel)

            robot.reset()
            print("[INFO]: Resetting robot state...")


        efforts = torch.randn_like(robot.data.joint_pos) * 5.0

        robot.set_joint_effort_target(efforts)

        robot.write_data_to_sim()

        sim.step()

        count += 1

        robot.update(sim_dt)


def main():
    """Main function."""

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])

    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)

    sim.reset()

    print("[INFO]: Setup complete...")

    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":

    main()

    simulation_app.close()
