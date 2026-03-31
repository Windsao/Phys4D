




"""
This script shows how to use a teleoperation device with Isaac Sim.

The teleoperation device is a keyboard device that allows the user to control the robot.
It is possible to add additional callbacks to it for user-defined operations.
"""

"""Launch Isaac Sim Simulator first."""


from isaaclab.app import AppLauncher


app_launcher = AppLauncher()
simulation_app = app_launcher.app

"""Rest everything follows."""

import ctypes

from isaacsim.core.api.simulation_context import SimulationContext

from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg


def print_cb():
    """Dummy callback function executed when the key 'L' is pressed."""
    print("Print callback")


def quit_cb():
    """Dummy callback function executed when the key 'ESC' is pressed."""
    print("Quit callback")
    simulation_app.close()


def main():

    sim = SimulationContext(physics_dt=0.01, rendering_dt=0.01)


    teleop_interface = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.1, rot_sensitivity=0.1))


    teleop_interface.add_callback("L", print_cb)
    teleop_interface.add_callback("ESCAPE", quit_cb)

    print("Press 'L' to print a message. Press 'ESC' to quit.")


    if ctypes.c_long.from_address(id(teleop_interface)).value != 1:
        raise RuntimeError("Teleoperation interface is not bounded to a single instance.")


    teleop_interface.reset()


    sim.reset()


    while simulation_app.is_running():

        if sim.is_stopped():
            break

        if not sim.is_playing():
            sim.step()
            continue

        delta_pose, gripper_command = teleop_interface.advance()

        if gripper_command:
            print(f"Gripper command: {gripper_command}")

        sim.step()

        if sim.is_stopped():
            break


if __name__ == "__main__":

    main()

    simulation_app.close()
