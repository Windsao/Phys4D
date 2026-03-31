




"""
This script demonstrates different dexterous hands.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/hands.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="This script demonstrates different dexterous hands.")

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




from isaaclab_assets.robots.allegro import ALLEGRO_HAND_CFG
from isaaclab_assets.robots.shadow_hand import SHADOW_HAND_CFG


def define_origins(num_origins: int, spacing: float) -> list[list[float]]:
    """Defines the origins of the the scene."""

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



    origins = define_origins(num_origins=2, spacing=0.5)


    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])

    allegro = Articulation(ALLEGRO_HAND_CFG.replace(prim_path="/World/Origin1/Robot"))


    prim_utils.create_prim("/World/Origin2", "Xform", translation=origins[1])

    shadow_hand = Articulation(SHADOW_HAND_CFG.replace(prim_path="/World/Origin2/Robot"))


    scene_entities = {
        "allegro": allegro,
        "shadow_hand": shadow_hand,
    }
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""

    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    grasp_mode = 0

    while simulation_app.is_running():

        if count % 1000 == 0:

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

        if count % 100 == 0:
            grasp_mode = 1 - grasp_mode

        for robot in entities.values():

            joint_pos_target = robot.data.soft_joint_pos_limits[..., grasp_mode]

            robot.set_joint_position_target(joint_pos_target)

            robot.write_data_to_sim()

        sim.step()

        sim_time += sim_dt
        count += 1

        for robot in entities.values():
            robot.update(sim_dt)


def main():
    """Main function."""

    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view(eye=[0.0, -0.5, 1.5], target=[0.0, -0.2, 0.5])

    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)

    sim.reset()

    print("[INFO]: Setup complete...")

    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":

    main()

    simulation_app.close()
