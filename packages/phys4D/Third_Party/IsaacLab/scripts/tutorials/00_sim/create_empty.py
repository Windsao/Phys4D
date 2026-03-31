




"""This script demonstrates how to create a simple stage in Isaac Sim.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from isaaclab.sim import SimulationCfg, SimulationContext


def main():
    """Main function."""


    sim_cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)

    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])


    sim.reset()

    print("[INFO]: Setup complete...")


    while simulation_app.is_running():

        sim.step()


if __name__ == "__main__":

    main()

    simulation_app.close()
