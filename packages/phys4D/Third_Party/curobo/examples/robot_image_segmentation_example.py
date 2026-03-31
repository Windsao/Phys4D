









"""This example shows how to use cuRobo's kinematics to generate a mask."""



import time


import imageio
import numpy as np
import torch
import torch.autograd.profiler as profiler
from mesh_dataset import MeshDataset
from torch.profiler import ProfilerActivity, profile, record_function


from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.geom.types import PointCloud, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.camera import CameraObservation
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.types.state import JointState
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.model.robot_segmenter import RobotSegmenter

torch.manual_seed(30)

torch.backends.cudnn.benchmark = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def create_render_dataset(
    robot_file,
    save_debug_data: bool = False,
    fov_deg: float = 60,
    n_frames: int = 20,
    retract_delta: float = 0.0,
):

    robot_dict = load_yaml(join_path(get_robot_configs_path(), robot_file))
    robot_dict["robot_cfg"]["kinematics"]["load_link_names_with_mesh"] = True
    robot_dict["robot_cfg"]["kinematics"]["load_meshes"] = True

    robot_cfg = RobotConfig.from_dict(robot_dict["robot_cfg"])

    kin_model = CudaRobotModel(robot_cfg.kinematics)

    q = kin_model.retract_config

    q += retract_delta

    meshes = kin_model.get_robot_as_mesh(q)

    world = WorldConfig(mesh=meshes[:])
    world_table = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_test.yml"))
    )
    world_table.cuboid[0].dims = [0.5, 0.5, 0.1]
    world.add_obstacle(world_table.objects[0])
    world.add_obstacle(world_table.objects[1])
    if save_debug_data:
        world.save_world_as_mesh("scene.stl", process_color=False)
    robot_mesh = (
        WorldConfig.create_merged_mesh_world(world, process_color=False).mesh[0].get_trimesh_mesh()
    )

    mesh_dataset = MeshDataset(
        None,
        n_frames=n_frames,
        image_size=1920,
        save_data_dir=None,
        trimesh_mesh=robot_mesh,
        fov_deg=fov_deg,
    )
    q_js = JointState(position=q, joint_names=kin_model.joint_names)

    return mesh_dataset, q_js


def mask_image(robot_file="ur5e.yml"):
    save_debug_data = False
    write_pointcloud = False

    tensor_args = TensorDeviceType()

    curobo_segmenter = RobotSegmenter.from_robot_file(
        robot_file,
        collision_sphere_buffer=0.01,
        distance_threshold=0.05,
        use_cuda_graph=True,
        ops_dtype=torch.float16,
    )

    mesh_dataset, q_js = create_render_dataset(robot_file, write_pointcloud, n_frames=30)

    if save_debug_data:
        visualize_scale = 10.0
        data = mesh_dataset[0]
        cam_obs = CameraObservation(
            depth_image=tensor_args.to_device(data["depth"]).unsqueeze(0) * 1000,
            intrinsics=data["intrinsics"],
            pose=Pose.from_matrix(data["pose"].to(device=tensor_args.device)),
        )

        imageio.imwrite(
            "camera_depth.png",
            (cam_obs.depth_image * visualize_scale)
            .squeeze()
            .detach()
            .cpu()
            .numpy()
            .astype(np.uint16),
        )


        robot_kinematics = curobo_segmenter._robot_world.kinematics
        if write_pointcloud:
            sph = robot_kinematics.get_robot_as_spheres(q_js.position)
            WorldConfig(sphere=sph[0]).save_world_as_mesh("robot_spheres.stl")



            pc = cam_obs.get_pointcloud()
            pc_obs = PointCloud("world", pose=cam_obs.pose.to_list(), points=pc)
            pc_obs.save_as_mesh("camera_pointcloud.stl", transform_with_pose=True)


        depth_mask, filtered_image = curobo_segmenter.get_robot_mask_from_active_js(
            cam_obs,
            q_js,
        )


        robot_mask = cam_obs.clone()
        robot_mask.depth_image[~depth_mask] = 0.0

        if write_pointcloud:
            robot_mesh = PointCloud(
                "world", pose=robot_mask.pose.to_list(), points=robot_mask.get_pointcloud()
            )
            robot_mesh.save_as_mesh("robot_segmented.stl", transform_with_pose=True)

        imageio.imwrite(
            "robot_depth.png",
            (robot_mask.depth_image * visualize_scale)
            .detach()
            .squeeze()
            .cpu()
            .numpy()
            .astype(np.uint16),
        )



        world_mask = cam_obs.clone()
        world_mask.depth_image[depth_mask] = 0.0
        if write_pointcloud:
            world_mesh = PointCloud(
                "world", pose=world_mask.pose.to_list(), points=world_mask.get_pointcloud()
            )
            world_mesh.save_as_mesh("world_segmented.stl", transform_with_pose=True)

        imageio.imwrite(
            "world_depth.png",
            (world_mask.depth_image * visualize_scale)
            .detach()
            .squeeze()
            .cpu()
            .numpy()
            .astype(np.uint16),
        )

    dt_list = []
    for j in range(10):

        for i in range(len(mesh_dataset)):
            data = mesh_dataset[i]
            cam_obs = CameraObservation(
                depth_image=tensor_args.to_device(data["depth"]).unsqueeze(0) * 1000,
                intrinsics=data["intrinsics"],
                pose=Pose.from_matrix(data["pose"].to(device=tensor_args.device)),
            )
            if not curobo_segmenter.ready:
                curobo_segmenter.update_camera_projection(cam_obs)
            st_time = time.time()

            depth_mask, filtered_image = curobo_segmenter.get_robot_mask_from_active_js(
                cam_obs,
                q_js,
            )

            torch.cuda.synchronize()
            dt_list.append(time.time() - st_time)

        print(
            "Segmentation Time (ms), (hz)",
            np.mean(dt_list[5:]) * 1000.0,
            1.0 / np.mean(dt_list[5:]),
        )


def batch_mask_image(robot_file="ur5e.yml"):
    """Mask images from different camera views using batched query.

    Note: This only works for a single joint configuration across camera views.

    Args:
        robot_file: robot to use for example.
    """
    save_debug_data = True

    tensor_args = TensorDeviceType()

    curobo_segmenter = RobotSegmenter.from_robot_file(
        robot_file,
        collision_sphere_buffer=0.01,
        distance_threshold=0.05,
        use_cuda_graph=True,
        ops_dtype=torch.float16,
    )

    mesh_dataset, q_js = create_render_dataset(robot_file, save_debug_data, fov_deg=60)

    mesh_dataset_zoom, q_js = create_render_dataset(
        robot_file, save_debug_data, fov_deg=40, n_frames=30
    )

    if save_debug_data:
        visualize_scale = 10.0
        data = mesh_dataset[0]
        cam_obs = CameraObservation(
            depth_image=tensor_args.to_device(data["depth"]).unsqueeze(0) * 1000,
            intrinsics=data["intrinsics"],
            pose=Pose.from_matrix(data["pose"].to(device=tensor_args.device)),
        )
        data_zoom = mesh_dataset_zoom[1]
        cam_obs = CameraObservation(
            depth_image=tensor_args.to_device(data["depth"]) * 1000,
            intrinsics=data["intrinsics"],
            pose=Pose.from_matrix(data["pose"].to(device=tensor_args.device)),
        )
        cam_obs_zoom = CameraObservation(
            depth_image=tensor_args.to_device(data_zoom["depth"]) * 1000,
            intrinsics=data_zoom["intrinsics"],
            pose=Pose.from_matrix(data_zoom["pose"].to(device=tensor_args.device)),
        )
        cam_obs = cam_obs.stack(cam_obs_zoom)

        for i in range(cam_obs.depth_image.shape[0]):

            imageio.imwrite(
                "camera_depth_" + str(i) + ".png",
                (cam_obs.depth_image[i] * visualize_scale)
                .squeeze()
                .detach()
                .cpu()
                .numpy()
                .astype(np.uint16),
            )


        depth_mask, filtered_image = curobo_segmenter.get_robot_mask_from_active_js(
            cam_obs,
            q_js,
        )


        robot_mask = cam_obs.clone()
        robot_mask.depth_image[~depth_mask] = 0.0
        for i in range(cam_obs.depth_image.shape[0]):

            imageio.imwrite(
                "robot_depth_" + str(i) + ".png",
                (robot_mask.depth_image[i] * visualize_scale)
                .detach()
                .squeeze()
                .cpu()
                .numpy()
                .astype(np.uint16),
            )



            imageio.imwrite(
                "world_depth_" + str(i) + ".png",
                (filtered_image[i] * visualize_scale)
                .detach()
                .squeeze()
                .cpu()
                .numpy()
                .astype(np.uint16),
            )

    dt_list = []

    for i in range(len(mesh_dataset)):

        data = mesh_dataset[i]
        data_zoom = mesh_dataset_zoom[i + 1]
        cam_obs = CameraObservation(
            depth_image=tensor_args.to_device(data["depth"]) * 1000,
            intrinsics=data["intrinsics"],
            pose=Pose.from_matrix(data["pose"].to(device=tensor_args.device)),
        )
        cam_obs_zoom = CameraObservation(
            depth_image=tensor_args.to_device(data_zoom["depth"]) * 1000,
            intrinsics=data_zoom["intrinsics"],
            pose=Pose.from_matrix(data_zoom["pose"].to(device=tensor_args.device)),
        )
        cam_obs = cam_obs.stack(cam_obs_zoom)
        if not curobo_segmenter.ready:
            curobo_segmenter.update_camera_projection(cam_obs)
        st_time = time.time()

        depth_mask, filtered_image = curobo_segmenter.get_robot_mask_from_active_js(
            cam_obs,
            q_js,
        )

        torch.cuda.synchronize()
        dt_list.append(time.time() - st_time)

    print("Segmentation Time (ms), (hz)", np.mean(dt_list[5:]) * 1000.0, 1.0 / np.mean(dt_list[5:]))


def batch_robot_mask_image(robot_file="ur5e.yml"):
    """Mask images from different camera views using batched query.

    Note: This example treats each image to have different robot joint configuration

    Args:
        robot_file: robot to use for example.
    """
    save_debug_data = True

    tensor_args = TensorDeviceType()

    curobo_segmenter = RobotSegmenter.from_robot_file(
        robot_file, collision_sphere_buffer=0.01, distance_threshold=0.05, use_cuda_graph=True
    )

    mesh_dataset, q_js = create_render_dataset(robot_file, save_debug_data, fov_deg=60)

    mesh_dataset_zoom, q_js_zoom = create_render_dataset(
        robot_file, save_debug_data, fov_deg=60, retract_delta=-0.5
    )
    q_js = q_js.unsqueeze(0)
    q_js = q_js.stack(q_js_zoom.unsqueeze(0))

    if save_debug_data:
        visualize_scale = 10.0
        data = mesh_dataset[0]
        cam_obs = CameraObservation(
            depth_image=tensor_args.to_device(data["depth"]).unsqueeze(0) * 1000,
            intrinsics=data["intrinsics"],
            pose=Pose.from_matrix(data["pose"].to(device=tensor_args.device)),
        )
        data_zoom = mesh_dataset_zoom[0]
        cam_obs = CameraObservation(
            depth_image=tensor_args.to_device(data["depth"]) * 1000,
            intrinsics=data["intrinsics"],
            pose=Pose.from_matrix(data["pose"].to(device=tensor_args.device)),
        )
        cam_obs_zoom = CameraObservation(
            depth_image=tensor_args.to_device(data_zoom["depth"]) * 1000,
            intrinsics=data_zoom["intrinsics"],
            pose=Pose.from_matrix(data_zoom["pose"].to(device=tensor_args.device)),
        )
        cam_obs = cam_obs.stack(cam_obs_zoom)

        for i in range(cam_obs.depth_image.shape[0]):

            imageio.imwrite(
                "camera_depth_" + str(i) + ".png",
                (cam_obs.depth_image[i] * visualize_scale)
                .squeeze()
                .detach()
                .cpu()
                .numpy()
                .astype(np.uint16),
            )


        depth_mask, filtered_image = curobo_segmenter.get_robot_mask_from_active_js(
            cam_obs,
            q_js,
        )


        robot_mask = cam_obs.clone()
        robot_mask.depth_image[~depth_mask] = 0.0
        for i in range(cam_obs.depth_image.shape[0]):

            imageio.imwrite(
                "robot_depth_" + str(i) + ".png",
                (robot_mask.depth_image[i] * visualize_scale)
                .detach()
                .squeeze()
                .cpu()
                .numpy()
                .astype(np.uint16),
            )



            imageio.imwrite(
                "world_depth_" + str(i) + ".png",
                (filtered_image[i] * visualize_scale)
                .detach()
                .squeeze()
                .cpu()
                .numpy()
                .astype(np.uint16),
            )

    dt_list = []

    for i in range(len(mesh_dataset)):

        data = mesh_dataset[i]
        data_zoom = mesh_dataset_zoom[i]
        cam_obs = CameraObservation(
            depth_image=tensor_args.to_device(data["depth"]) * 1000,
            intrinsics=data["intrinsics"],
            pose=Pose.from_matrix(data["pose"].to(device=tensor_args.device)),
        )
        cam_obs_zoom = CameraObservation(
            depth_image=tensor_args.to_device(data_zoom["depth"]) * 1000,
            intrinsics=data_zoom["intrinsics"],
            pose=Pose.from_matrix(data_zoom["pose"].to(device=tensor_args.device)),
        )
        cam_obs = cam_obs.stack(cam_obs_zoom)
        if not curobo_segmenter.ready:
            curobo_segmenter.update_camera_projection(cam_obs)
        st_time = time.time()

        depth_mask, filtered_image = curobo_segmenter.get_robot_mask_from_active_js(
            cam_obs,
            q_js,
        )

        torch.cuda.synchronize()
        dt_list.append(time.time() - st_time)

    print("Segmentation Time (ms), (hz)", np.mean(dt_list[5:]) * 1000.0, 1.0 / np.mean(dt_list[5:]))


def profile_mask_image(robot_file="ur5e.yml"):

    tensor_args = TensorDeviceType()

    curobo_segmenter = RobotSegmenter.from_robot_file(
        robot_file, collision_sphere_buffer=0.0, distance_threshold=0.05, use_cuda_graph=False
    )

    mesh_dataset, q_js = create_render_dataset(robot_file)

    dt_list = []
    data = mesh_dataset[0]
    cam_obs = CameraObservation(
        depth_image=tensor_args.to_device(data["depth"]).unsqueeze(0) * 1000,
        intrinsics=data["intrinsics"],
        pose=Pose.from_matrix(data["pose"].to(device=tensor_args.device)),
    )
    if not curobo_segmenter.ready:
        curobo_segmenter.update_camera_projection(cam_obs)
    depth_mask, filtered_image = curobo_segmenter.get_robot_mask_from_active_js(cam_obs, q_js)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:

        for i in range(len(mesh_dataset)):
            with profiler.record_function("get_data"):

                data = mesh_dataset[i]
                cam_obs = CameraObservation(
                    depth_image=tensor_args.to_device(data["depth"]).unsqueeze(0) * 1000,
                    intrinsics=data["intrinsics"],
                    pose=Pose.from_matrix(data["pose"].to(device=tensor_args.device)),
                )
            st_time = time.time()
            with profiler.record_function("segmentation"):

                depth_mask, filtered_image = curobo_segmenter.get_robot_mask_from_active_js(
                    cam_obs, q_js
                )

    print("Exporting the trace..")
    prof.export_chrome_trace("segmentation.json")


if __name__ == "__main__":
    robot_file = "quad_ur10e.yml"


    batch_mask_image(robot_file)

