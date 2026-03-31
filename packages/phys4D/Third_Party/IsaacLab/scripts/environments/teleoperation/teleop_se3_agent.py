




"""Script to run a keyboard teleoperation with Isaac Lab manipulation environments."""

"""Launch Isaac Sim Simulator first."""

import argparse
from collections.abc import Callable

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Keyboard teleoperation for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    help=(
        "Teleop device. Set here (legacy) or via the environment config. If using the environment config, pass the"
        " device key/name defined under 'teleop_devices' (it can be a custom name, not necessarily 'handtracking')."
        " Built-ins: keyboard, spacemouse, gamepad. Not all tasks support all built-ins."
    ),
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

if args_cli.enable_pinocchio:



    import pinocchio
if "handtracking" in args_cli.teleop_device.lower():
    app_launcher_args["xr"] = True


app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import torch

import omni.log

from isaaclab.devices import Se3Gamepad, Se3GamepadCfg, Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.managers import TerminationTermCfg as DoneTerm

import isaaclab_tasks
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.utils import parse_env_cfg

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.locomanipulation.pick_place
    import isaaclab_tasks.manager_based.manipulation.pick_place


def main() -> None:
    """
    Run keyboard teleoperation with Isaac Lab manipulation environment.

    Creates the environment, sets up teleoperation interfaces and callbacks,
    and runs the main simulation loop until the application is closed.

    Returns:
        None
    """

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task

    env_cfg.terminations.time_out = None
    if "Lift" in args_cli.task:

        env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)

        env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)

    if args_cli.xr:


        env_cfg = remove_camera_configs(env_cfg)
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    try:

        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

        if "Reach" in args_cli.task:
            omni.log.warn(
                f"The environment '{args_cli.task}' does not support gripper control. The device command will be"
                " ignored."
            )
    except Exception as e:
        omni.log.error(f"Failed to create environment: {e}")
        simulation_app.close()
        return


    should_reset_recording_instance = False
    teleoperation_active = True


    def reset_recording_instance() -> None:
        """
        Reset the environment to its initial state.

        Sets a flag to reset the environment on the next simulation step.

        Returns:
            None
        """
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True
        print("Reset triggered - Environment will reset on next step")

    def start_teleoperation() -> None:
        """
        Activate teleoperation control of the robot.

        Enables the application of teleoperation commands to the environment.

        Returns:
            None
        """
        nonlocal teleoperation_active
        teleoperation_active = True
        print("Teleoperation activated")

    def stop_teleoperation() -> None:
        """
        Deactivate teleoperation control of the robot.

        Disables the application of teleoperation commands to the environment.

        Returns:
            None
        """
        nonlocal teleoperation_active
        teleoperation_active = False
        print("Teleoperation deactivated")


    teleoperation_callbacks: dict[str, Callable[[], None]] = {
        "R": reset_recording_instance,
        "START": start_teleoperation,
        "STOP": stop_teleoperation,
        "RESET": reset_recording_instance,
    }


    if args_cli.xr:

        teleoperation_active = False
    else:

        teleoperation_active = True


    teleop_interface = None
    try:
        if hasattr(env_cfg, "teleop_devices") and args_cli.teleop_device in env_cfg.teleop_devices.devices:
            teleop_interface = create_teleop_device(
                args_cli.teleop_device, env_cfg.teleop_devices.devices, teleoperation_callbacks
            )
        else:
            omni.log.warn(f"No teleop device '{args_cli.teleop_device}' found in environment config. Creating default.")

            sensitivity = args_cli.sensitivity
            if args_cli.teleop_device.lower() == "keyboard":
                teleop_interface = Se3Keyboard(
                    Se3KeyboardCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity)
                )
            elif args_cli.teleop_device.lower() == "spacemouse":
                teleop_interface = Se3SpaceMouse(
                    Se3SpaceMouseCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity)
                )
            elif args_cli.teleop_device.lower() == "gamepad":
                teleop_interface = Se3Gamepad(
                    Se3GamepadCfg(pos_sensitivity=0.1 * sensitivity, rot_sensitivity=0.1 * sensitivity)
                )
            else:
                omni.log.error(f"Unsupported teleop device: {args_cli.teleop_device}")
                omni.log.error("Supported devices: keyboard, spacemouse, gamepad, handtracking")
                env.close()
                simulation_app.close()
                return


            for key, callback in teleoperation_callbacks.items():
                try:
                    teleop_interface.add_callback(key, callback)
                except (ValueError, TypeError) as e:
                    omni.log.warn(f"Failed to add callback for key {key}: {e}")
    except Exception as e:
        omni.log.error(f"Failed to create teleop device: {e}")
        env.close()
        simulation_app.close()
        return

    if teleop_interface is None:
        omni.log.error("Failed to create teleop interface")
        env.close()
        simulation_app.close()
        return

    print(f"Using teleop device: {teleop_interface}")


    env.reset()
    teleop_interface.reset()

    print("Teleoperation started. Press 'R' to reset the environment.")


    while simulation_app.is_running():
        try:

            with torch.inference_mode():

                action = teleop_interface.advance()


                if teleoperation_active:

                    actions = action.repeat(env.num_envs, 1)

                    env.step(actions)
                else:
                    env.sim.render()

                if should_reset_recording_instance:
                    env.reset()
                    should_reset_recording_instance = False
                    print("Environment reset complete")
        except Exception as e:
            omni.log.error(f"Error during simulation step: {e}")
            break


    env.close()
    print("Environment closed")


if __name__ == "__main__":

    main()

    simulation_app.close()
