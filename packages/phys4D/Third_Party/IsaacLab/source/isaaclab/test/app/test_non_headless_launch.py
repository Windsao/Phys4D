




"""
This script checks if the app can be launched with non-headless app and start the simulation.
"""

"""Launch Isaac Sim Simulator first."""


import pytest

from isaaclab.app import AppLauncher


app_launcher = AppLauncher(experience="isaaclab.python.kit", headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass


@configclass
class SensorsSceneCfg(InteractiveSceneCfg):
    """Design the scene with sensors on the robot."""


    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())


def run_simulator(
    sim: sim_utils.SimulationContext,
):
    """Run the simulator."""

    count = 0


    while simulation_app.is_running() and count < 100:

        sim.step()
        count += 1


@pytest.mark.isaacsim_ci
def test_non_headless_launch():

    sim_cfg = sim_utils.SimulationCfg(dt=0.005)
    sim = sim_utils.SimulationContext(sim_cfg)

    scene_cfg = SensorsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    print(scene)

    sim.reset()

    print("[INFO]: Setup complete...")

    run_simulator(sim)
