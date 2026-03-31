




import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Example on using the raycaster sensor.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.ray_caster import RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR




from isaaclab_assets.robots.anymal import ANYMAL_C_CFG


@configclass
class RaycasterSensorSceneCfg(InteractiveSceneCfg):
    """Design the scene with sensors on the robot."""


    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/rough_plane.usd",
            scale=(1, 1, 1),
        ),
    )


    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )


    robot = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    ray_caster = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/lidar_cage",
        update_period=1 / 60,
        offset=RayCasterCfg.OffsetCfg(pos=(0, 0, 0.5)),
        mesh_prim_paths=["/World/Ground"],
        ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=100, vertical_fov_range=[-90, 90], horizontal_fov_range=[-90, 90], horizontal_res=1.0
        ),
        debug_vis=not args_cli.headless,
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulator."""

    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    triggered = True
    countdown = 42


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
            joint_pos += torch.rand_like(joint_pos) * 0.1
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


        print("-------------------------------")
        print(scene["ray_caster"])
        print("Ray cast hit results: ", scene["ray_caster"].data.ray_hits_w)

        if not triggered:
            if countdown > 0:
                countdown -= 1
                continue
            data = scene["ray_caster"].data.ray_hits_w.cpu().numpy()
            np.save("cast_data.npy", data)
            triggered = True
        else:
            continue


def main():
    """Main function."""


    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])

    scene_cfg = RaycasterSensorSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()

    print("[INFO]: Setup complete...")

    run_simulator(sim, scene)


if __name__ == "__main__":

    main()

    simulation_app.close()
