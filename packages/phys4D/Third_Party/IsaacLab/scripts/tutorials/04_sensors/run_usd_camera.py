




"""
This script shows how to use the camera sensor from the Isaac Lab framework.

The camera sensor is created and interfaced through the Omniverse Replicator API. However, instead of using
the simulator or OpenGL convention for the camera, we use the robotics or ROS convention.

.. code-block:: bash

    # Usage with GUI
    ./isaaclab.sh -p scripts/tutorials/04_sensors/run_usd_camera.py --enable_cameras

    # Usage with headless
    ./isaaclab.sh -p scripts/tutorials/04_sensors/run_usd_camera.py --headless --enable_cameras

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="This script demonstrates how to use the camera sensor.")
parser.add_argument(
    "--draw",
    action="store_true",
    default=False,
    help="Draw the pointcloud from camera at index specified by ``--camera_id``.",
)
parser.add_argument(
    "--save",
    action="store_true",
    default=False,
    help="Save the data from camera at index specified by ``--camera_id``.",
)
parser.add_argument(
    "--camera_id",
    type=int,
    choices={0, 1},
    default=0,
    help=(
        "The camera ID to use for displaying points or saving the camera data. Default is 0."
        " The viewport will always initialize with the perspective of camera 0."
    ),
)

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import os
import random
import torch

import isaacsim.core.utils.prims as prim_utils
import omni.replicator.core as rep

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sensors.camera.utils import create_pointcloud_from_depth
from isaaclab.utils import convert_dict_to_backend


def define_sensor() -> Camera:
    """Defines the camera sensor to add to the scene."""



    prim_utils.create_prim("/World/Origin_00", "Xform")
    prim_utils.create_prim("/World/Origin_01", "Xform")
    camera_cfg = CameraCfg(
        prim_path="/World/Origin_.*/CameraSensor",
        update_period=0,
        height=480,
        width=640,
        data_types=[
            "rgb",
            "distance_to_image_plane",
            "normals",
            "semantic_segmentation",
            "instance_segmentation_fast",
            "instance_id_segmentation_fast",
        ],
        colorize_semantic_segmentation=True,
        colorize_instance_id_segmentation=True,
        colorize_instance_segmentation=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
    )

    camera = Camera(cfg=camera_cfg)

    return camera


def design_scene() -> dict:
    """Design the scene."""


    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)


    scene_entities = {}


    prim_utils.create_prim("/World/Objects", "Xform")

    for i in range(8):

        position = np.random.rand(3) - np.asarray([0.05, 0.05, -1.0])
        position *= np.asarray([1.5, 1.5, 0.5])

        color = (random.random(), random.random(), random.random())

        prim_type = random.choice(["Cube", "Cone", "Cylinder"])
        common_properties = {
            "rigid_props": sim_utils.RigidBodyPropertiesCfg(),
            "mass_props": sim_utils.MassPropertiesCfg(mass=5.0),
            "collision_props": sim_utils.CollisionPropertiesCfg(),
            "visual_material": sim_utils.PreviewSurfaceCfg(diffuse_color=color, metallic=0.5),
            "semantic_tags": [("class", prim_type)],
        }
        if prim_type == "Cube":
            shape_cfg = sim_utils.CuboidCfg(size=(0.25, 0.25, 0.25), **common_properties)
        elif prim_type == "Cone":
            shape_cfg = sim_utils.ConeCfg(radius=0.1, height=0.25, **common_properties)
        elif prim_type == "Cylinder":
            shape_cfg = sim_utils.CylinderCfg(radius=0.25, height=0.25, **common_properties)

        obj_cfg = RigidObjectCfg(
            prim_path=f"/World/Objects/Obj_{i:02d}",
            spawn=shape_cfg,
            init_state=RigidObjectCfg.InitialStateCfg(pos=position),
        )
        scene_entities[f"rigid_object{i}"] = RigidObject(cfg=obj_cfg)


    camera = define_sensor()


    scene_entities["camera"] = camera
    return scene_entities


def run_simulator(sim: sim_utils.SimulationContext, scene_entities: dict):
    """Run the simulator."""

    camera: Camera = scene_entities["camera"]


    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "camera")
    rep_writer = rep.BasicWriter(
        output_dir=output_dir,
        frame_padding=0,
        colorize_instance_id_segmentation=camera.cfg.colorize_instance_id_segmentation,
        colorize_instance_segmentation=camera.cfg.colorize_instance_segmentation,
        colorize_semantic_segmentation=camera.cfg.colorize_semantic_segmentation,
    )


    camera_positions = torch.tensor([[2.5, 2.5, 2.5], [-2.5, -2.5, 2.5]], device=sim.device)
    camera_targets = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=sim.device)

    camera_orientations = torch.tensor(
        [[-0.1759, 0.3399, 0.8205, -0.4247], [-0.4247, 0.8205, -0.3399, 0.1759]], device=sim.device
    )



    camera.set_world_poses_from_view(camera_positions, camera_targets)




    camera_index = args_cli.camera_id


    if sim.has_gui() and args_cli.draw:
        cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/CameraPointCloud")
        cfg.markers["hit"].radius = 0.002
        pc_markers = VisualizationMarkers(cfg)


    while simulation_app.is_running():

        sim.step()

        camera.update(dt=sim.get_physics_dt())


        print(camera)
        if "rgb" in camera.data.output.keys():
            print("Received shape of rgb image        : ", camera.data.output["rgb"].shape)
        if "distance_to_image_plane" in camera.data.output.keys():
            print("Received shape of depth image      : ", camera.data.output["distance_to_image_plane"].shape)
        if "normals" in camera.data.output.keys():
            print("Received shape of normals          : ", camera.data.output["normals"].shape)
        if "semantic_segmentation" in camera.data.output.keys():
            print("Received shape of semantic segm.   : ", camera.data.output["semantic_segmentation"].shape)
        if "instance_segmentation_fast" in camera.data.output.keys():
            print("Received shape of instance segm.   : ", camera.data.output["instance_segmentation_fast"].shape)
        if "instance_id_segmentation_fast" in camera.data.output.keys():
            print("Received shape of instance id segm.: ", camera.data.output["instance_id_segmentation_fast"].shape)
        print("-------------------------------")


        if args_cli.save:


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


        if sim.has_gui() and args_cli.draw and "distance_to_image_plane" in camera.data.output.keys():

            pointcloud = create_pointcloud_from_depth(
                intrinsic_matrix=camera.data.intrinsic_matrices[camera_index],
                depth=camera.data.output["distance_to_image_plane"][camera_index],
                position=camera.data.pos_w[camera_index],
                orientation=camera.data.quat_w_ros[camera_index],
                device=sim.device,
            )




            if pointcloud.size()[0] > 0:
                pc_markers.visualize(translations=pointcloud)


def main():
    """Main function."""

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    scene_entities = design_scene()

    sim.reset()

    print("[INFO]: Setup complete...")

    run_simulator(sim, scene_entities)


if __name__ == "__main__":

    main()

    simulation_app.close()
