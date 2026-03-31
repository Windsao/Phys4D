




"""This script demonstrates how to spawn a pick-and-place robot equipped with a surface gripper and interact with it.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/01_assets/run_surface_gripper.py --device=cpu

When running this script make sure the --device flag is set to cpu. This is because the surface gripper is
currently only supported on the CPU.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with a Surface Gripper.")

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, SurfaceGripper, SurfaceGripperCfg
from isaaclab.sim import SimulationContext




from isaaclab_assets import PICK_AND_PLACE_CFG


def design_scene():
    """Designs the scene."""

    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)



    origins = [[2.75, 0.0, 0.0], [-2.75, 0.0, 0.0]]

    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])

    prim_utils.create_prim("/World/Origin2", "Xform", translation=origins[1])


    pick_and_place_robot_cfg = PICK_AND_PLACE_CFG.copy()
    pick_and_place_robot_cfg.prim_path = "/World/Origin.*/Robot"
    pick_and_place_robot = Articulation(cfg=pick_and_place_robot_cfg)


    surface_gripper_cfg = SurfaceGripperCfg()

    surface_gripper_cfg.prim_path = "/World/Origin.*/Robot/picker_head/SurfaceGripper"


    surface_gripper_cfg.max_grip_distance = 0.1
    surface_gripper_cfg.shear_force_limit = 500.0
    surface_gripper_cfg.coaxial_force_limit = 500.0
    surface_gripper_cfg.retry_interval = 0.1

    surface_gripper = SurfaceGripper(cfg=surface_gripper_cfg)


    scene_entities = {"pick_and_place_robot": pick_and_place_robot, "surface_gripper": surface_gripper}
    return scene_entities, origins


def run_simulator(
    sim: sim_utils.SimulationContext, entities: dict[str, Articulation | SurfaceGripper], origins: torch.Tensor
):
    """Runs the simulation loop."""

    robot: Articulation = entities["pick_and_place_robot"]
    surface_gripper: SurfaceGripper = entities["surface_gripper"]


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

            surface_gripper.reset()
            print("[INFO]: Resetting gripper state...")


        gripper_commands = torch.rand(surface_gripper.num_instances) * 2.0 - 1.0




        print(f"[INFO]: Gripper commands: {gripper_commands}")
        mapped_commands = [
            "Opening" if command < -0.3 else "Closing" if command > 0.3 else "Idle" for command in gripper_commands
        ]
        print(f"[INFO]: Mapped commands: {mapped_commands}")

        surface_gripper.set_grippers_command(gripper_commands)

        surface_gripper.write_data_to_sim()

        sim.step()

        count += 1

        surface_gripper.update(sim_dt)

        surface_gripper_state = surface_gripper.state





        print(f"[INFO]: Gripper state: {surface_gripper_state}")
        mapped_commands = [
            "Open" if state == -1 else "Closing" if state == 0 else "Closed" for state in surface_gripper_state.tolist()
        ]
        print(f"[INFO]: Mapped commands: {mapped_commands}")


def main():
    """Main function."""

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    sim.set_camera_view([2.75, 7.5, 10.0], [2.75, 0.0, 0.0])

    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)

    sim.reset()

    print("[INFO]: Setup complete...")

    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":

    main()

    simulation_app.close()
