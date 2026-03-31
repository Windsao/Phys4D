




"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Example on using the IMU sensor.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ImuCfg
from isaaclab.utils import configclass




from isaaclab_assets.robots.anymal import ANYMAL_C_CFG


@configclass
class ImuSensorSceneCfg(InteractiveSceneCfg):
    """Design the scene with sensors on the robot."""


    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())


    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )


    robot = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    imu_RF = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/LF_FOOT", debug_vis=True)

    imu_LF = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/RF_FOOT", gravity_bias=(0, 0, 0), debug_vis=True)


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulator."""

    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0


    while simulation_app.is_running():

        if count % 500 == 0:

            count = 0




            root_state = scene["robot"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            scene["robot"].write_root_link_pose_to_sim(root_state[:, :7])
            scene["robot"].write_root_com_velocity_to_sim(root_state[:, 7:])

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
        print(scene["imu_LF"])
        print("Received linear velocity: ", scene["imu_LF"].data.lin_vel_b)
        print("Received angular velocity: ", scene["imu_LF"].data.ang_vel_b)
        print("Received linear acceleration: ", scene["imu_LF"].data.lin_acc_b)
        print("Received angular acceleration: ", scene["imu_LF"].data.ang_acc_b)
        print("-------------------------------")
        print(scene["imu_RF"])
        print("Received linear velocity: ", scene["imu_RF"].data.lin_vel_b)
        print("Received angular velocity: ", scene["imu_RF"].data.ang_vel_b)
        print("Received linear acceleration: ", scene["imu_RF"].data.lin_acc_b)
        print("Received angular acceleration: ", scene["imu_RF"].data.ang_acc_b)


def main():
    """Main function."""


    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])

    scene_cfg = ImuSensorSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()

    print("[INFO]: Setup complete...")

    run_simulator(sim, scene)


if __name__ == "__main__":

    main()

    simulation_app.close()
