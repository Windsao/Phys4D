




"""
This script demonstrates how to generate log outputs while the simulation plays.
It accompanies the tutorial on docker usage.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/00_sim/log_time.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse
import os

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Tutorial on creating logs from within the docker container.")

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from isaaclab.sim import SimulationCfg, SimulationContext


def main():
    """Main function."""

    log_dir_path = os.path.join("logs")
    if not os.path.isdir(log_dir_path):
        os.mkdir(log_dir_path)




    log_dir_path = os.path.abspath(os.path.join(log_dir_path, "docker_tutorial"))
    if not os.path.isdir(log_dir_path):
        os.mkdir(log_dir_path)
    print(f"[INFO] Logging experiment to directory: {log_dir_path}")


    sim_cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)

    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])


    sim.reset()

    print("[INFO]: Setup complete...")


    sim_dt = sim.get_physics_dt()
    sim_time = 0.0


    with open(os.path.join(log_dir_path, "log.txt"), "w") as log_file:

        while simulation_app.is_running():
            log_file.write(f"{sim_time}" + "\n")

            sim.step()
            sim_time += sim_dt


if __name__ == "__main__":

    main()

    simulation_app.close()
