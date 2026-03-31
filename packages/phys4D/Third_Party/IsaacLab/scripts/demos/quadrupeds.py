




"""
This script demonstrates different legged robots.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/quadrupeds.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="This script demonstrates different legged robots.")

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation




from isaaclab_assets.robots.anymal import ANYMAL_B_CFG, ANYMAL_C_CFG, ANYMAL_D_CFG
from isaaclab_assets.robots.spot import SPOT_CFG
from isaaclab_assets.robots.unitree import UNITREE_A1_CFG, UNITREE_GO1_CFG, UNITREE_GO2_CFG


def define_origins(num_origins: int, spacing: float) -> list[list[float]]:
    """Defines the origins of the scene."""

    env_origins = torch.zeros(num_origins, 3)

    num_cols = np.floor(np.sqrt(num_origins))
    num_rows = np.ceil(num_origins / num_cols)
    xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="xy")
    env_origins[:, 0] = spacing * xx.flatten()[:num_origins] - spacing * (num_rows - 1) / 2
    env_origins[:, 1] = spacing * yy.flatten()[:num_origins] - spacing * (num_cols - 1) / 2
    env_origins[:, 2] = 0.0

    return env_origins.tolist()


def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""

    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)



    origins = define_origins(num_origins=7, spacing=1.25)


    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])

    anymal_b = Articulation(ANYMAL_B_CFG.replace(prim_path="/World/Origin1/Robot"))


    prim_utils.create_prim("/World/Origin2", "Xform", translation=origins[1])

    anymal_c = Articulation(ANYMAL_C_CFG.replace(prim_path="/World/Origin2/Robot"))


    prim_utils.create_prim("/World/Origin3", "Xform", translation=origins[2])

    anymal_d = Articulation(ANYMAL_D_CFG.replace(prim_path="/World/Origin3/Robot"))


    prim_utils.create_prim("/World/Origin4", "Xform", translation=origins[3])

    unitree_a1 = Articulation(UNITREE_A1_CFG.replace(prim_path="/World/Origin4/Robot"))


    prim_utils.create_prim("/World/Origin5", "Xform", translation=origins[4])

    unitree_go1 = Articulation(UNITREE_GO1_CFG.replace(prim_path="/World/Origin5/Robot"))


    prim_utils.create_prim("/World/Origin6", "Xform", translation=origins[5])

    unitree_go2 = Articulation(UNITREE_GO2_CFG.replace(prim_path="/World/Origin6/Robot"))


    prim_utils.create_prim("/World/Origin7", "Xform", translation=origins[6])

    spot = Articulation(SPOT_CFG.replace(prim_path="/World/Origin7/Robot"))


    scene_entities = {
        "anymal_b": anymal_b,
        "anymal_c": anymal_c,
        "anymal_d": anymal_d,
        "unitree_a1": unitree_a1,
        "unitree_go1": unitree_go1,
        "unitree_go2": unitree_go2,
        "spot": spot,
    }
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""

    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    while simulation_app.is_running():

        if count % 200 == 0:

            sim_time = 0.0
            count = 0

            for index, robot in enumerate(entities.values()):

                root_state = robot.data.default_root_state.clone()
                root_state[:, :3] += origins[index]
                robot.write_root_pose_to_sim(root_state[:, :7])
                robot.write_root_velocity_to_sim(root_state[:, 7:])

                joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
                robot.write_joint_state_to_sim(joint_pos, joint_vel)

                robot.reset()
            print("[INFO]: Resetting robots state...")

        for robot in entities.values():

            joint_pos_target = robot.data.default_joint_pos + torch.randn_like(robot.data.joint_pos) * 0.1

            robot.set_joint_position_target(joint_pos_target)

            robot.write_data_to_sim()

        sim.step()

        sim_time += sim_dt
        count += 1

        for robot in entities.values():
            robot.update(sim_dt)


def main():
    """Main function."""


    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01))

    sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.0])

    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)

    sim.reset()

    print("[INFO]: Setup complete...")

    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":

    main()

    simulation_app.close()
