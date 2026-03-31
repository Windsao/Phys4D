




"""This script demonstrates how to spawn prims into the scene.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/00_sim/set_rendering_mode.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Tutorial on viewing a warehouse scene with a given rendering mode preset."
)

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


def main():
    """Main function."""



    rendering_mode = "performance"


    carb_settings = {"rtx.reflections.enabled": True}


    render_cfg = sim_utils.RenderCfg(
        rendering_mode=rendering_mode,
        carb_settings=carb_settings,
    )


    sim_cfg = sim_utils.SimulationCfg(render=render_cfg)
    sim = sim_utils.SimulationContext(sim_cfg)


    sim.set_camera_view([-11, -0.5, 2], [0, 0, 0.5])


    hospital_usd_path = f"{ISAAC_NUCLEUS_DIR}/Environments/Hospital/hospital.usd"
    cfg = sim_utils.UsdFileCfg(usd_path=hospital_usd_path)
    cfg.func("/Scene", cfg)


    sim.reset()


    print("[INFO]: Setup complete...")


    while simulation_app.is_running():
        sim.step()


if __name__ == "__main__":

    main()

    simulation_app.close()
