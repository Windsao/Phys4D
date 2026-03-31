




"""
This script shows how to use replicator to randomly change the textures of a USD scene.

Note:
    Currently this script fails since cloner does not support changing textures of cloned
    USD prims. This is because the prims are cloned using `Sdf.ChangeBlock` which does not
    allow individual texture changes.

Usage:

.. code-block:: bash

    ./isaaclab.sh -p source/isaaclab/test/deps/isaacsim/check_rep_texture_randomizer.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse


from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="This script shows how to use replicator to randomly change the textures of a USD scene."
)

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""

import numpy as np
import torch

import isaacsim.core.utils.prims as prim_utils
import omni.replicator.core as rep
from isaacsim.core.api.simulation_context import SimulationContext
from isaacsim.core.cloner import GridCloner
from isaacsim.core.objects import DynamicSphere
from isaacsim.core.prims import RigidPrim
from isaacsim.core.utils.viewports import set_camera_view


def main():
    """Spawn a bunch of balls and randomly change their textures."""


    sim_params = {
        "use_gpu": True,
        "use_gpu_pipeline": True,
        "use_flatcache": True,
        "use_fabric": True,
        "enable_scene_query_support": True,
    }
    sim = SimulationContext(
        physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0, sim_params=sim_params, backend="torch", device="cuda:0"
    )

    set_camera_view([0.0, 30.0, 25.0], [0.0, 0.0, -2.5])


    num_balls = 128


    cloner = GridCloner(spacing=2.0)
    cloner.define_base_env("/World/envs")

    prim_utils.define_prim("/World/envs/env_0")



    DynamicSphere(prim_path="/World/envs/env_0/ball", translation=np.array([0.0, 0.0, 5.0]), mass=0.5, radius=0.25)


    cloner.define_base_env("/World/envs")
    envs_prim_paths = cloner.generate_paths("/World/envs/env", num_paths=num_balls)
    env_positions = cloner.clone(
        source_prim_path="/World/envs/env_0", prim_paths=envs_prim_paths, replicate_physics=True, copy_from_source=True
    )
    physics_scene_path = sim.get_physics_context().prim_path
    cloner.filter_collisions(
        physics_scene_path, "/World/collisions", prim_paths=envs_prim_paths, global_paths=["/World/ground"]
    )


    with rep.new_layer():

        def get_shapes():
            shapes = rep.get.prims(path_pattern="/World/envs/env_.*/ball")
            with shapes:
                rep.randomizer.color(colors=rep.distribution.uniform((0, 0, 0), (1, 1, 1)))
            return shapes.node


        rep.randomizer.register(get_shapes)

        with rep.trigger.on_frame():
            rep.randomizer.get_shapes()



    ball_view = RigidPrim("/World/envs/env_.*/ball", reset_xform_properties=False)

    ball_initial_positions = torch.tensor(env_positions, dtype=torch.float, device=sim.device)
    ball_initial_positions[:, 2] += 5.0


    ball_view.set_world_poses(positions=ball_initial_positions)


    sim.reset()

    rep.orchestrator.step(pause_timeline=False)

    rep.orchestrator.stop()

    sim.pause()


    ball_view.initialize()
    ball_initial_velocities = ball_view.get_velocities()


    step_count = 0

    while simulation_app.is_running():

        if sim.is_stopped():
            break

        if not sim.is_playing():
            sim.step()
            continue

        if step_count % 500 == 0:

            ball_view.set_world_poses(positions=ball_initial_positions)
            ball_view.set_velocities(ball_initial_velocities)

            step_count = 0

        sim.step()

        step_count += 1


if __name__ == "__main__":

    main()

    simulation_app.close()
