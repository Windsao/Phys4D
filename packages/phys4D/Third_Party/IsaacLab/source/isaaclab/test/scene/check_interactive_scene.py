




"""
This script demonstrates how to use the scene interface to quickly setup a scene with multiple
articulated robots and sensors.
"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="This script demonstrates how to use the scene interface.")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
args_cli = parser.parse_args()


app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.ray_caster import RayCasterCfg, patterns
from isaaclab.sim import SimulationContext
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.timer import Timer




from isaaclab_assets.robots.anymal import ANYMAL_C_CFG


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Example scene configuration."""


    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
    )


    robot_1 = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_1")

    robot_2 = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_2")
    robot_2.init_state.pos = (0.0, 1.0, 0.6)


    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot_1/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )


    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 500.0)),
    )


def main():
    """Main function."""


    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.005))

    sim.set_camera_view(eye=[5, 5, 5], target=[0.0, 0.0, 0.0])


    with Timer("Setup scene"):
        scene = InteractiveScene(MySceneCfg(num_envs=args_cli.num_envs, env_spacing=5.0, lazy_sensor_update=False))


    assert len(scene.env_prim_paths) == args_cli.num_envs, "Number of environments does not match."
    assert scene.terrain is not None, "Terrain not found."
    assert len(scene.articulations) == 2, "Number of robots does not match."
    assert len(scene.sensors) == 1, "Number of sensors does not match."
    assert len(scene.extras) == 1, "Number of extras does not match."


    with Timer("Time taken to play the simulator"):
        sim.reset()


    print("[INFO]: Setup complete...")


    robot_1_actions = scene.articulations["robot_1"].data.default_joint_pos.clone()
    robot_2_actions = scene.articulations["robot_2"].data.default_joint_pos.clone()

    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    while simulation_app.is_running():

        if sim.is_stopped():
            break

        if not sim.is_playing():
            sim.step()
            continue

        if count % 50 == 0:

            sim_time = 0.0
            count = 0

            root_state = scene.articulations["robot_1"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            joint_pos = scene.articulations["robot_1"].data.default_joint_pos
            joint_vel = scene.articulations["robot_1"].data.default_joint_vel


            scene.articulations["robot_1"].write_root_pose_to_sim(root_state[:, :7])
            scene.articulations["robot_1"].write_root_velocity_to_sim(root_state[:, 7:])
            scene.articulations["robot_1"].write_joint_state_to_sim(joint_pos, joint_vel)

            root_state[:, 1] += 1.0
            scene.articulations["robot_2"].write_root_pose_to_sim(root_state[:, :7])
            scene.articulations["robot_2"].write_root_velocity_to_sim(root_state[:, 7:])
            scene.articulations["robot_2"].write_joint_state_to_sim(joint_pos, joint_vel)

            scene.reset()
            print(">>>>>>>> Reset!")

        for _ in range(4):

            scene.articulations["robot_1"].set_joint_position_target(robot_1_actions)
            scene.articulations["robot_2"].set_joint_position_target(robot_2_actions)

            scene.write_data_to_sim()

            sim.step()

            scene.update(sim_dt)

        sim_time += sim_dt * 4
        count += 1


if __name__ == "__main__":

    main()

    simulation_app.close()
