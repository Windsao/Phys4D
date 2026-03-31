




"""
This script demonstrates how to use the cloner API from Isaac Sim.

Reference: https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/tutorial_gym_cloner.html
"""

"""Launch Isaac Sim Simulator first."""


import argparse
import contextlib

with contextlib.suppress(ModuleNotFoundError):
    import isaacsim

from isaacsim import SimulationApp


parser = argparse.ArgumentParser(
    description="This script shows the issue in Isaac Sim with GPU simulation of floating robots."
)
parser.add_argument("--num_robots", type=int, default=128, help="Number of robots to spawn.")
parser.add_argument(
    "--asset",
    type=str,
    default="isaaclab",
    help="The asset source location for the robot. Can be: isaaclab, oige, custom asset path.",
)
parser.add_argument("--headless", action="store_true", help="Run in headless mode.")

args_cli = parser.parse_args()


simulation_app = SimulationApp({"headless": args_cli.headless})

"""Rest everything follows."""

import os
import torch

import omni.log

try:
    import isaacsim.storage.native as nucleus_utils
except ModuleNotFoundError:
    import isaacsim.core.utils.nucleus as nucleus_utils

import isaacsim.core.utils.prims as prim_utils
from isaacsim.core.api.world import World
from isaacsim.core.cloner import GridCloner
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.carb import set_carb_setting
from isaacsim.core.utils.viewports import set_camera_view


if nucleus_utils.get_assets_root_path() is None:
    msg = (
        "Unable to perform Nucleus login on Omniverse. Assets root path is not set.\n"
        "\tPlease check: https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html#omniverse-nucleus"
    )
    omni.log.error(msg)
    raise RuntimeError(msg)


ISAAC_NUCLEUS_DIR = f"{nucleus_utils.get_assets_root_path()}/Isaac"
"""Path to the `Isaac` directory on the NVIDIA Nucleus Server."""

ISAACLAB_NUCLEUS_DIR = f"{ISAAC_NUCLEUS_DIR}/IsaacLab"
"""Path to the `Isaac/IsaacLab` directory on the NVIDIA Nucleus Server."""


"""
Main
"""


def main():
    """Spawns the ANYmal robot and clones it using Isaac Sim Cloner API."""


    world = World(physics_dt=0.005, rendering_dt=0.005, backend="torch", device="cuda:0")

    set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])



    set_carb_setting(world._settings, "/persistent/omnihydra/useSceneGraphInstancing", True)


    cloner = GridCloner(spacing=2.0)
    cloner.define_base_env("/World/envs")

    prim_utils.define_prim("/World/envs/env_0")



    world.scene.add_default_ground_plane(prim_path="/World/defaultGroundPlane", z_position=0.0)

    prim_utils.create_prim("/World/Light/GreySphere", "SphereLight", translation=(4.5, 3.5, 10.0))

    prim_utils.create_prim("/World/Light/WhiteSphere", "SphereLight", translation=(-4.5, 3.5, 10.0))


    if args_cli.asset == "isaaclab":
        usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-C/anymal_c.usd"
        root_prim_path = "/World/envs/env_.*/Robot/base"
    elif args_cli.asset == "oige":
        usd_path = f"{ISAAC_NUCLEUS_DIR}/Robots/ANYbotics/anymal_c/anymal_c.usd"
        root_prim_path = "/World/envs/env_.*/Robot"
    elif os.path.exists(args_cli.asset):
        usd_path = args_cli.asset
    else:
        raise ValueError(f"Invalid asset: {args_cli.asset}. Must be one of: isaaclab, oige.")

    print("Loading robot from: ", usd_path)
    prim_utils.create_prim(
        "/World/envs/env_0/Robot",
        usd_path=usd_path,
        translation=(0.0, 0.0, 0.6),
    )


    num_envs = args_cli.num_robots
    cloner.define_base_env("/World/envs")
    envs_prim_paths = cloner.generate_paths("/World/envs/env", num_paths=num_envs)
    envs_positions = cloner.clone(
        source_prim_path="/World/envs/env_0", prim_paths=envs_prim_paths, replicate_physics=True
    )

    envs_positions = torch.tensor(envs_positions, dtype=torch.float, device=world.device)

    physics_scene_path = world.get_physics_context().prim_path
    cloner.filter_collisions(
        physics_scene_path, "/World/collisions", envs_prim_paths, global_paths=["/World/defaultGroundPlane"]
    )


    if args_cli.asset == "isaaclab":
        root_prim_path = "/World/envs/env_.*/Robot/base"
    elif args_cli.asset == "oige":
        root_prim_path = "/World/envs/env_.*/Robot"
    elif os.path.exists(args_cli.asset):
        usd_path = args_cli.asset
        root_prim_path = "/World/envs/env_.*/Robot"
    else:
        raise ValueError(f"Invalid asset: {args_cli.asset}. Must be one of: isaaclab, oige.")

    robot_view = Articulation(root_prim_path, name="ANYMAL")
    world.scene.add(robot_view)

    world.reset()


    print("[INFO]: Setup complete...")





    sim_dt = world.get_physics_dt()

    sim_time = 0.0

    while simulation_app.is_running():

        if world.is_stopped():
            break

        if not world.is_playing():
            world.step(render=False)
            continue

        world.step()

        sim_time += sim_dt


if __name__ == "__main__":

    main()

    simulation_app.close()
