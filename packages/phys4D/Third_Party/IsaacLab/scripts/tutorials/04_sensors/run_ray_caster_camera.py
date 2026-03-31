




"""
This script shows how to use the ray-cast camera sensor from the Isaac Lab framework.

The camera sensor is based on using Warp kernels which do ray-casting against static meshes.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/04_sensors/run_ray_caster_camera.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="This script demonstrates how to use the ray-cast camera sensor.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to generate.")
parser.add_argument("--save", action="store_true", default=False, help="Save the obtained data to disk.")

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import torch

import isaacsim.core.utils.prims as prim_utils
import omni.replicator.core as rep

import isaaclab.sim as sim_utils
from isaaclab.sensors.ray_caster import RayCasterCamera, RayCasterCameraCfg, patterns
from isaaclab.utils import convert_dict_to_backend
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import project_points, unproject_depth


def define_sensor() -> RayCasterCamera:
    """Defines the ray-cast camera sensor to add to the scene."""



    prim_utils.create_prim("/World/Origin_00/CameraSensor", "Xform")
    prim_utils.create_prim("/World/Origin_01/CameraSensor", "Xform")


    camera_cfg = RayCasterCameraCfg(
        prim_path="/World/Origin_.*/CameraSensor",
        mesh_prim_paths=["/World/ground"],
        update_period=0.1,
        offset=RayCasterCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        data_types=["distance_to_image_plane", "normals", "distance_to_camera"],
        debug_vis=True,
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
            height=480,
            width=640,
        ),
    )

    camera = RayCasterCamera(cfg=camera_cfg)

    return camera


def design_scene():


    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/rough_plane.usd")
    cfg.func("/World/ground", cfg)

    cfg = sim_utils.DistantLightCfg(intensity=600.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    camera = define_sensor()


    scene_entities = {"camera": camera}
    return scene_entities


def run_simulator(sim: sim_utils.SimulationContext, scene_entities: dict):
    """Run the simulator."""

    camera: RayCasterCamera = scene_entities["camera"]


    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "ray_caster_camera")
    rep_writer = rep.BasicWriter(output_dir=output_dir, frame_padding=3)



    eyes = torch.tensor([[2.5, 2.5, 2.5], [-2.5, -2.5, 2.5]], device=sim.device)
    targets = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=sim.device)
    camera.set_world_poses_from_view(eyes, targets)






    while simulation_app.is_running():

        sim.step()

        camera.update(dt=sim.get_physics_dt())


        print(camera)
        print("Received shape of depth image: ", camera.data.output["distance_to_image_plane"].shape)
        print("-------------------------------")


        if args_cli.save:

            camera_index = 0

            single_cam_data = convert_dict_to_backend(
                {k: v[camera_index] for k, v in camera.data.output.items()}, backend="numpy"
            )

            single_cam_info = camera.data.info[camera_index]


            rep_output = {"annotators": {}}
            for key, data, info in zip(single_cam_data.keys(), single_cam_data.values(), single_cam_info.values()):
                if info is not None:
                    rep_output["annotators"][key] = {"render_product": {"data": data, **info}}
                else:
                    rep_output["annotators"][key] = {"render_product": {"data": data}}

            rep_output["trigger_outputs"] = {"on_time": camera.frame[camera_index]}
            rep_writer.write(rep_output)


            points_3d_cam = unproject_depth(
                camera.data.output["distance_to_image_plane"], camera.data.intrinsic_matrices
            )


            im_height, im_width = camera.image_shape

            reproj_points = project_points(points_3d_cam, camera.data.intrinsic_matrices)
            reproj_depths = reproj_points[..., -1].view(-1, im_width, im_height).transpose_(1, 2)
            sim_depths = camera.data.output["distance_to_image_plane"].squeeze(-1)
            torch.testing.assert_close(reproj_depths, sim_depths)


def main():
    """Main function."""

    sim_cfg = sim_utils.SimulationCfg()
    sim = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view([2.5, 2.5, 3.5], [0.0, 0.0, 0.0])

    scene_entities = design_scene()

    sim.reset()

    print("[INFO]: Setup complete...")

    run_simulator(sim=sim, scene_entities=scene_entities)


if __name__ == "__main__":

    main()

    simulation_app.close()
