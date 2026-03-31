




"""
This script demonstrates how to simulate bipedal robots.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/bipeds.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="This script demonstrates how to simulate bipedal robots.")

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext




from isaaclab_assets.robots.cassie import CASSIE_CFG
from isaaclab_assets import H1_CFG
from isaaclab_assets import G1_CFG


def design_scene(sim: sim_utils.SimulationContext) -> tuple[list, torch.Tensor]:
    """Designs the scene."""

    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)


    origins = torch.tensor([
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]).to(device=sim.device)


    cassie = Articulation(CASSIE_CFG.replace(prim_path="/World/Cassie"))
    h1 = Articulation(H1_CFG.replace(prim_path="/World/H1"))
    g1 = Articulation(G1_CFG.replace(prim_path="/World/G1"))
    robots = [cassie, h1, g1]

    return robots, origins


def run_simulator(sim: sim_utils.SimulationContext, robots: list[Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""

    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    while simulation_app.is_running():

        if count % 200 == 0:

            sim_time = 0.0
            count = 0
            for index, robot in enumerate(robots):

                joint_pos, joint_vel = robot.data.default_joint_pos, robot.data.default_joint_vel
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                root_state = robot.data.default_root_state.clone()
                root_state[:, :3] += origins[index]
                robot.write_root_pose_to_sim(root_state[:, :7])
                robot.write_root_velocity_to_sim(root_state[:, 7:])
                robot.reset()

            print(">>>>>>>> Reset!")

        for robot in robots:
            robot.set_joint_position_target(robot.data.default_joint_pos.clone())
            robot.write_data_to_sim()

        sim.step()

        sim_time += sim_dt
        count += 1

        for robot in robots:
            robot.update(sim_dt)


def main():
    """Main function."""

    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    sim.set_camera_view(eye=[3.0, 0.0, 2.25], target=[0.0, 0.0, 1.0])


    robots, origins = design_scene(sim)


    sim.reset()


    print("[INFO]: Setup complete...")


    run_simulator(sim, robots, origins)


if __name__ == "__main__":

    main()

    simulation_app.close()
