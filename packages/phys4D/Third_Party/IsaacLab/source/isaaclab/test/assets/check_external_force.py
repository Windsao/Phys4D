




"""
This script checks if the external force is applied correctly on the robot.

.. code-block:: bash

    # Usage to apply force on base
    ./isaaclab.sh -p source/isaaclab/test/assets/check_external_force.py --body base --force 1000
    # Usage to apply force on legs
    ./isaaclab.sh -p source/isaaclab/test/assets/check_external_force.py --body .*_SHANK --force 100
"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="This script demonstrates how to external force on a legged robot.")
parser.add_argument("--body", default="base", type=str, help="Name of the body to apply force on.")
parser.add_argument("--force", default=1000.0, type=float, help="Force to apply on the body.")

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext




from isaaclab_assets.robots.anymal import ANYMAL_C_CFG


def main():
    """Main function."""


    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.005))

    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])



    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    cfg = sim_utils.DistantLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light/greyLight", cfg)


    robot_cfg = ANYMAL_C_CFG
    robot_cfg.spawn.func("/World/Anymal_c/Robot_1", robot_cfg.spawn, translation=(0.0, -0.5, 0.65))
    robot_cfg.spawn.func("/World/Anymal_c/Robot_2", robot_cfg.spawn, translation=(0.0, 0.5, 0.65))

    robot = Articulation(robot_cfg.replace(prim_path="/World/Anymal_c/Robot.*"))


    sim.reset()


    body_ids, body_names = robot.find_bodies(args_cli.body)

    external_wrench_b = torch.zeros(robot.num_instances, len(body_ids), 6, device=sim.device)
    external_wrench_b[..., 1] = args_cli.force


    print("[INFO]: Setup complete...")
    print("[INFO]: Applying force on the robot: ", args_cli.body, " -> ", body_names)


    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    while simulation_app.is_running():

        if count % 100 == 0:

            sim_time = 0.0
            count = 0

            root_state = robot.data.default_root_state.clone()
            root_state[0, :2] = torch.tensor([0.0, -0.5], device=sim.device)
            root_state[1, :2] = torch.tensor([0.0, 0.5], device=sim.device)
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])

            joint_pos, joint_vel = robot.data.default_joint_pos, robot.data.default_joint_vel
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()

            robot.set_external_force_and_torque(
                external_wrench_b[..., :3], external_wrench_b[..., 3:], body_ids=body_ids
            )

            print(">>>>>>>> Reset!")

        robot.set_joint_position_target(robot.data.default_joint_pos.clone())
        robot.write_data_to_sim()

        sim.step()

        sim_time += sim_dt
        count += 1

        robot.update(sim_dt)


if __name__ == "__main__":

    main()

    simulation_app.close()
