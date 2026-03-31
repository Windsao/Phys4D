




"""
This script demonstrates the FrameTransformer sensor by visualizing the frames that it creates.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/04_sensors/run_frame_transformer.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="This script checks the FrameTransformer sensor by visualizing the frames that it creates."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import torch

import isaacsim.util.debug_draw._debug_draw as omni_debug_draw

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg, OffsetCfg
from isaaclab.sim import SimulationContext




from isaaclab_assets.robots.anymal import ANYMAL_C_CFG


def define_sensor() -> FrameTransformer:
    """Defines the FrameTransformer sensor to add to the scene."""

    rot_offset = math_utils.quat_from_euler_xyz(torch.zeros(1), torch.zeros(1), torch.tensor(-math.pi / 2))
    pos_offset = math_utils.quat_apply(rot_offset, torch.tensor([0.08795, 0.01305, -0.33797]))


    frame_transformer_cfg = FrameTransformerCfg(
        prim_path="/World/Robot/base",
        target_frames=[
            FrameTransformerCfg.FrameCfg(prim_path="/World/Robot/.*"),
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/Robot/LF_SHANK",
                name="LF_FOOT_USER",
                offset=OffsetCfg(pos=tuple(pos_offset.tolist()), rot=tuple(rot_offset[0].tolist())),
            ),
        ],
        debug_vis=False,
    )
    frame_transformer = FrameTransformer(frame_transformer_cfg)

    return frame_transformer


def design_scene() -> dict:
    """Design the scene."""


    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    robot = Articulation(ANYMAL_C_CFG.replace(prim_path="/World/Robot"))

    frame_transformer = define_sensor()


    scene_entities = {"robot": robot, "frame_transformer": frame_transformer}
    return scene_entities


def run_simulator(sim: sim_utils.SimulationContext, scene_entities: dict):
    """Run the simulator."""

    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0


    robot: Articulation = scene_entities["robot"]
    frame_transformer: FrameTransformer = scene_entities["frame_transformer"]




    if not args_cli.headless:
        cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/FrameVisualizerFromScript")
        cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        transform_visualizer = VisualizationMarkers(cfg)

        draw_interface = omni_debug_draw.acquire_debug_draw_interface()
    else:
        transform_visualizer = None
        draw_interface = None

    frame_index = 0

    while simulation_app.is_running():

        robot.set_joint_position_target(robot.data.default_joint_pos.clone())
        robot.write_data_to_sim()

        sim.step()

        sim_time += sim_dt
        count += 1

        robot.update(sim_dt)
        frame_transformer.update(dt=sim_dt)



        if not args_cli.headless:
            if count % 50 == 0:

                frame_names = frame_transformer.data.target_frame_names

                frame_index += 1
                frame_index = frame_index % len(frame_names)
                print(f"Displaying Frame ID {frame_index}: {frame_names[frame_index]}")


            source_pos = frame_transformer.data.source_pos_w
            source_quat = frame_transformer.data.source_quat_w
            target_pos = frame_transformer.data.target_pos_w[:, frame_index]
            target_quat = frame_transformer.data.target_quat_w[:, frame_index]

            transform_visualizer.visualize(
                torch.cat([source_pos, target_pos], dim=0), torch.cat([source_quat, target_quat], dim=0)
            )

            draw_interface.clear_lines()

            lines_colors = [[1.0, 1.0, 0.0, 1.0]] * source_pos.shape[0]
            line_thicknesses = [5.0] * source_pos.shape[0]
            draw_interface.draw_lines(source_pos.tolist(), target_pos.tolist(), lines_colors, line_thicknesses)


def main():
    """Main function."""

    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.0])

    scene_entities = design_scene()

    sim.reset()

    print("[INFO]: Setup complete...")

    run_simulator(sim, scene_entities)


if __name__ == "__main__":

    main()

    simulation_app.close()
