




"""
This script demonstrates how to simulate a quadcopter.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/quadcopter.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="This script demonstrates how to simulate a quadcopter.")

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext




from isaaclab_assets import CRAZYFLIE_CFG


def main():
    """Main function."""

    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    sim.set_camera_view(eye=[0.5, 0.5, 1.0], target=[0.0, 0.0, 0.5])



    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)


    robot_cfg = CRAZYFLIE_CFG.replace(prim_path="/World/Crazyflie")
    robot_cfg.spawn.func("/World/Crazyflie", robot_cfg.spawn, translation=robot_cfg.init_state.pos)


    robot = Articulation(robot_cfg)


    sim.reset()


    prop_body_ids = robot.find_bodies("m.*_prop")[0]
    robot_mass = robot.root_physx_view.get_masses().sum()
    gravity = torch.tensor(sim.cfg.gravity, device=sim.device).norm()


    print("[INFO]: Setup complete...")


    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    while simulation_app.is_running():

        if count % 2000 == 0:

            sim_time = 0.0
            count = 0

            joint_pos, joint_vel = robot.data.default_joint_pos, robot.data.default_joint_vel
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.write_root_pose_to_sim(robot.data.default_root_state[:, :7])
            robot.write_root_velocity_to_sim(robot.data.default_root_state[:, 7:])
            robot.reset()

            print(">>>>>>>> Reset!")

        forces = torch.zeros(robot.num_instances, 4, 3, device=sim.device)
        torques = torch.zeros_like(forces)
        forces[..., 2] = robot_mass * gravity / 4.0
        robot.set_external_force_and_torque(forces, torques, body_ids=prop_body_ids)
        robot.write_data_to_sim()

        sim.step()

        sim_time += sim_dt
        count += 1

        robot.update(sim_dt)


if __name__ == "__main__":

    main()

    simulation_app.close()
