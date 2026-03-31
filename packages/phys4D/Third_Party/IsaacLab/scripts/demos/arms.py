




"""
This script demonstrates different single-arm manipulators.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/arms.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="This script demonstrates different single-arm manipulators.")

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
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR





from isaaclab_assets import (
    FRANKA_PANDA_CFG,
    UR10_CFG,
    KINOVA_JACO2_N7S300_CFG,
    KINOVA_JACO2_N6S300_CFG,
    KINOVA_GEN3_N7_CFG,
    SAWYER_CFG,
)




def define_origins(num_origins: int, spacing: float) -> list[list[float]]:
    """Defines the origins of the the scene."""

    env_origins = torch.zeros(num_origins, 3)

    num_rows = np.floor(np.sqrt(num_origins))
    num_cols = np.ceil(num_origins / num_rows)
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



    origins = define_origins(num_origins=6, spacing=2.0)


    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])

    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
    cfg.func("/World/Origin1/Table", cfg, translation=(0.55, 0.0, 1.05))

    franka_arm_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/Origin1/Robot")
    franka_arm_cfg.init_state.pos = (0.0, 0.0, 1.05)
    franka_panda = Articulation(cfg=franka_arm_cfg)


    prim_utils.create_prim("/World/Origin2", "Xform", translation=origins[1])

    cfg = sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
    )
    cfg.func("/World/Origin2/Table", cfg, translation=(0.0, 0.0, 1.03))

    ur10_cfg = UR10_CFG.replace(prim_path="/World/Origin2/Robot")
    ur10_cfg.init_state.pos = (0.0, 0.0, 1.03)
    ur10 = Articulation(cfg=ur10_cfg)


    prim_utils.create_prim("/World/Origin3", "Xform", translation=origins[2])

    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd")
    cfg.func("/World/Origin3/Table", cfg, translation=(0.0, 0.0, 0.8))

    kinova_arm_cfg = KINOVA_JACO2_N7S300_CFG.replace(prim_path="/World/Origin3/Robot")
    kinova_arm_cfg.init_state.pos = (0.0, 0.0, 0.8)
    kinova_j2n7s300 = Articulation(cfg=kinova_arm_cfg)


    prim_utils.create_prim("/World/Origin4", "Xform", translation=origins[3])

    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd")
    cfg.func("/World/Origin4/Table", cfg, translation=(0.0, 0.0, 0.8))

    kinova_arm_cfg = KINOVA_JACO2_N6S300_CFG.replace(prim_path="/World/Origin4/Robot")
    kinova_arm_cfg.init_state.pos = (0.0, 0.0, 0.8)
    kinova_j2n6s300 = Articulation(cfg=kinova_arm_cfg)


    prim_utils.create_prim("/World/Origin5", "Xform", translation=origins[4])

    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
    cfg.func("/World/Origin5/Table", cfg, translation=(0.55, 0.0, 1.05))

    kinova_arm_cfg = KINOVA_GEN3_N7_CFG.replace(prim_path="/World/Origin5/Robot")
    kinova_arm_cfg.init_state.pos = (0.0, 0.0, 1.05)
    kinova_gen3n7 = Articulation(cfg=kinova_arm_cfg)


    prim_utils.create_prim("/World/Origin6", "Xform", translation=origins[5])

    cfg = sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
    )
    cfg.func("/World/Origin6/Table", cfg, translation=(0.0, 0.0, 1.03))

    sawyer_arm_cfg = SAWYER_CFG.replace(prim_path="/World/Origin6/Robot")
    sawyer_arm_cfg.init_state.pos = (0.0, 0.0, 1.03)
    sawyer = Articulation(cfg=sawyer_arm_cfg)


    scene_entities = {
        "franka_panda": franka_panda,
        "ur10": ur10,
        "kinova_j2n7s300": kinova_j2n7s300,
        "kinova_j2n6s300": kinova_j2n6s300,
        "kinova_gen3n7": kinova_gen3n7,
        "sawyer": sawyer,
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
            joint_pos_target = joint_pos_target.clamp_(
                robot.data.soft_joint_pos_limits[..., 0], robot.data.soft_joint_pos_limits[..., 1]
            )

            robot.set_joint_position_target(joint_pos_target)

            robot.write_data_to_sim()

        sim.step()

        sim_time += sim_dt
        count += 1

        for robot in entities.values():
            robot.update(sim_dt)


def main():
    """Main function."""

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])

    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)

    sim.reset()

    print("[INFO]: Setup complete...")

    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":

    main()

    simulation_app.close()
