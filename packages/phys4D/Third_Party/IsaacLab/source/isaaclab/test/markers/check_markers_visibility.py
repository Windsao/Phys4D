




"""
This script checks if the debug markers are visible from the camera.

To check if the markers are visible on different rendering modalities, you can switch them by going
through the synthetic data generation tool in the Isaac Sim UI. For more information,
please check: https://www.youtube.com/watch?v=vLk-f9LWj48&ab_channel=NVIDIAOmniverse

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/isaaclab/test/markers/check_markers_visibility.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Check if the debug markers are visible from the camera.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass




from isaaclab_assets.robots.anymal import ANYMAL_C_CFG


@configclass
class SensorsSceneCfg(InteractiveSceneCfg):
    """Design the scene with sensors on the robot."""


    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())


    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )


    robot: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        update_period=0.02,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=True,
        mesh_prim_paths=["/World/defaultGroundPlane"],
    )


def run_simulator(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
):
    """Run the simulator."""


    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0


    while simulation_app.is_running():

        if count % 500 == 0:

            count = 0


            root_state = scene["robot"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            scene["robot"].write_root_pose_to_sim(root_state[:, :7])
            scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])

            joint_pos, joint_vel = (
                scene["robot"].data.default_joint_pos.clone(),
                scene["robot"].data.default_joint_vel.clone(),
            )
            scene["robot"].write_joint_state_to_sim(joint_pos, joint_vel)

            scene.reset()
            print("[INFO]: Resetting robot state...")


        targets = scene["robot"].data.default_joint_pos

        scene["robot"].set_joint_position_target(targets)

        scene.write_data_to_sim()

        sim.step()

        sim_time += sim_dt
        count += 1

        scene.update(sim_dt)


def main():
    """Main function."""


    sim_cfg = sim_utils.SimulationCfg(dt=0.005)
    sim = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])

    scene_cfg = SensorsSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()

    print("[INFO]: Setup complete...")

    run_simulator(sim, scene)


if __name__ == "__main__":

    main()

    simulation_app.close()
