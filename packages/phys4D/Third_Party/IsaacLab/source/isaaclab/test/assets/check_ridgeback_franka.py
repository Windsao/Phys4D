




"""
This script demonstrates how to simulate a mobile manipulator.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/isaaclab/test/assets/check_ridgeback_franka.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="This script demonstrates how to simulate a mobile manipulator with dummy joints."
)

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation




from isaaclab_assets.robots.ridgeback_franka import RIDGEBACK_FRANKA_PANDA_CFG


def design_scene():
    """Designs the scene."""

    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    return add_robots()


def add_robots() -> Articulation:
    """Adds robots to the scene."""
    robot_cfg = RIDGEBACK_FRANKA_PANDA_CFG

    robot_cfg.spawn.func("/World/Robot_1", robot_cfg.spawn, translation=(0.0, -1.0, 0.0))
    robot_cfg.spawn.func("/World/Robot_2", robot_cfg.spawn, translation=(0.0, 1.0, 0.0))

    robot = Articulation(cfg=robot_cfg.replace(prim_path="/World/Robot.*"))

    return robot


def run_simulator(sim: sim_utils.SimulationContext, robot: Articulation):
    """Runs the simulator by applying actions to the robot at every time-step"""

    actions = robot.data.default_joint_pos.clone()


    sim_dt = sim.get_physics_dt()

    sim_time = 0.0
    ep_step_count = 0

    while simulation_app.is_running():

        if ep_step_count % 1000 == 0:

            sim_time = 0.0
            ep_step_count = 0

            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)

            robot.reset()

            actions = torch.rand_like(robot.data.default_joint_pos) + robot.data.default_joint_pos

            actions[:, 0:3] = 0.0

            actions[:, -2:] = 0.04
            print("[INFO]: Resetting robots state...")

        if ep_step_count % 200 == 0:

            actions[:, -2:] = 0.0 if actions[0, -2] > 0.0 else 0.04


        if ep_step_count == 200:
            actions[:, :3] = 0.0
            actions[:, 0] = 1.0
        if ep_step_count == 300:
            actions[:, :3] = 0.0
            actions[:, 0] = -1.0

        if ep_step_count == 400:
            actions[:, :3] = 0.0
            actions[:, 1] = 1.0
        if ep_step_count == 500:
            actions[:, :3] = 0.0
            actions[:, 1] = -1.0

        if ep_step_count == 600:
            actions[:, :3] = 0.0
            actions[:, 2] = 1.0
        if ep_step_count == 700:
            actions[:, :3] = 0.0
            actions[:, 2] = -1.0
        if ep_step_count == 900:
            actions[:, :3] = 0.0
            actions[:, 2] = 1.0

        if ep_step_count % 100:
            actions[:, 3:10] = torch.rand(robot.num_instances, 7, device=robot.device)
            actions[:, 3:10] += robot.data.default_joint_pos[:, 3:10]

        robot.set_joint_velocity_target(actions[:, :3], joint_ids=[0, 1, 2])
        robot.set_joint_position_target(actions[:, 3:], joint_ids=[3, 4, 5, 6, 7, 8, 9, 10, 11])
        robot.write_data_to_sim()

        sim.step()

        sim_time += sim_dt
        ep_step_count += 1

        robot.update(sim_dt)


def main():
    """Main function."""

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg())

    sim.set_camera_view([1.5, 1.5, 1.5], [0.0, 0.0, 0.0])

    robot = design_scene()

    sim.reset()

    print("[INFO]: Setup complete...")

    run_simulator(sim, robot)


if __name__ == "__main__":

    main()

    simulation_app.close()
