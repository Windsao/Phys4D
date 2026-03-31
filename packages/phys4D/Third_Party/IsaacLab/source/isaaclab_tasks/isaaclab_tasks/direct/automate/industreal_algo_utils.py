
































"""IndustReal: algorithms module.

Contains functions that implement Simulation-Aware Policy Update (SAPU), SDF-Based Reward, and Sampling-Based Curriculum (SBC).

Not intended to be executed as a standalone script.
"""


import gc
import numpy as np
import os


import torch
import trimesh
from trimesh.exchange.load import load


import warp as wp

from isaaclab.utils.assets import retrieve_file_path

"""
Simulation-Aware Policy Update (SAPU)
"""


def load_asset_mesh_in_warp(held_asset_obj, fixed_asset_obj, num_samples, device):
    """Create mesh objects in Warp for all environments."""
    retrieve_file_path(held_asset_obj, download_dir="./")
    plug_trimesh = load(os.path.basename(held_asset_obj))

    retrieve_file_path(fixed_asset_obj, download_dir="./")
    socket_trimesh = load(os.path.basename(fixed_asset_obj))


    plug_wp_mesh = wp.Mesh(
        points=wp.array(plug_trimesh.vertices, dtype=wp.vec3, device=device),
        indices=wp.array(plug_trimesh.faces.flatten(), dtype=wp.int32, device=device),
    )


    sampled_points, _ = trimesh.sample.sample_surface_even(plug_trimesh, num_samples)
    wp_mesh_sampled_points = wp.array(sampled_points, dtype=wp.vec3, device=device)

    socket_wp_mesh = wp.Mesh(
        points=wp.array(socket_trimesh.vertices, dtype=wp.vec3, device=device),
        indices=wp.array(socket_trimesh.faces.flatten(), dtype=wp.int32, device=device),
    )

    return plug_wp_mesh, wp_mesh_sampled_points, socket_wp_mesh


"""
SDF-Based Reward
"""


def get_sdf_reward(
    wp_plug_mesh,
    wp_plug_mesh_sampled_points,
    plug_pos,
    plug_quat,
    socket_pos,
    socket_quat,
    wp_device,
    device,
):
    """Calculate SDF-based reward."""

    num_envs = len(plug_pos)
    sdf_reward = torch.zeros((num_envs,), dtype=torch.float32, device=device)

    for i in range(num_envs):


        mesh_points = wp.clone(wp_plug_mesh.points)
        mesh_indices = wp.clone(wp_plug_mesh.indices)
        mesh_copy = wp.Mesh(points=mesh_points, indices=mesh_indices)




        goal_transform = wp.transform(socket_pos[i], socket_quat[i])
        wp.launch(
            kernel=transform_points,
            dim=len(mesh_copy.points),
            inputs=[mesh_copy.points, mesh_copy.points, goal_transform],
            device=wp_device,
        )


        mesh_copy.refit()


        sampled_points = wp.clone(wp_plug_mesh_sampled_points)


        curr_transform = wp.transform(plug_pos[i], plug_quat[i])
        wp.launch(
            kernel=transform_points,
            dim=len(sampled_points),
            inputs=[sampled_points, sampled_points, curr_transform],
            device=wp_device,
        )


        sdf_dist = wp.zeros((len(sampled_points),), dtype=wp.float32, device=wp_device)
        wp.launch(
            kernel=get_batch_sdf,
            dim=len(sampled_points),
            inputs=[mesh_copy.id, sampled_points, sdf_dist],
            device=wp_device,
        )
        sdf_dist = wp.to_torch(sdf_dist)


        sdf_dist = torch.where(sdf_dist < 0.0, 0.0, sdf_dist)

        sdf_reward[i] = torch.mean(sdf_dist)

        del mesh_copy
        del mesh_points
        del mesh_indices
        del sampled_points

    sdf_reward = -torch.log(sdf_reward)

    gc.collect()
    return sdf_reward


"""
Sampling-Based Curriculum (SBC)
"""


def get_curriculum_reward_scale(cfg_task, curr_max_disp):
    """Compute reward scale for SBC."""




    curr_stage_diff = cfg_task.curriculum_height_bound[1] - curr_max_disp



    final_stage_diff = cfg_task.curriculum_height_bound[1] - cfg_task.curriculum_height_bound[0]


    reward_scale = curr_stage_diff / final_stage_diff + 1.0

    return reward_scale


def get_new_max_disp(curr_success, cfg_task, curr_max_disp):
    """Update max downward displacement of plug at beginning of episode, based on success rate."""

    if curr_success > cfg_task.curriculum_success_thresh:


        new_max_disp = max(
            curr_max_disp + cfg_task.curriculum_height_step[0],
            cfg_task.curriculum_height_bound[0],
        )

    elif curr_success < cfg_task.curriculum_failure_thresh:


        new_max_disp = min(
            curr_max_disp + cfg_task.curriculum_height_step[1],
            cfg_task.curriculum_height_bound[1],
        )

    else:

        new_max_disp = curr_max_disp

    return new_max_disp


"""
Bonus and Success Checking
"""


def get_keypoint_offsets(num_keypoints, device):
    """Get uniformly-spaced keypoints along a line of unit length, centered at 0."""

    keypoint_offsets = torch.zeros((num_keypoints, 3), device=device)
    keypoint_offsets[:, -1] = torch.linspace(0.0, 1.0, num_keypoints, device=device) - 0.5

    return keypoint_offsets


def check_plug_close_to_socket(keypoints_plug, keypoints_socket, dist_threshold, progress_buf):
    """Check if plug is close to socket."""


    keypoint_dist = torch.norm(keypoints_socket - keypoints_plug, p=2, dim=-1)


    is_plug_close_to_socket = torch.where(
        torch.sum(keypoint_dist, dim=-1) < dist_threshold,
        torch.ones_like(progress_buf),
        torch.zeros_like(progress_buf),
    )

    return is_plug_close_to_socket


def check_plug_inserted_in_socket(
    plug_pos, socket_pos, keypoints_plug, keypoints_socket, success_height_thresh, close_error_thresh, progress_buf
):
    """Check if plug is inserted in socket."""


    is_plug_below_insertion_height = plug_pos[:, 2] < socket_pos[:, 2] + success_height_thresh




    is_plug_close_to_socket = check_plug_close_to_socket(
        keypoints_plug=keypoints_plug,
        keypoints_socket=keypoints_socket,
        dist_threshold=close_error_thresh,
        progress_buf=progress_buf,
    )


    is_plug_inserted_in_socket = torch.logical_and(is_plug_below_insertion_height, is_plug_close_to_socket)

    return is_plug_inserted_in_socket


def get_engagement_reward_scale(plug_pos, socket_pos, is_plug_engaged_w_socket, success_height_thresh, device):
    """Compute scale on reward. If plug is not engaged with socket, scale is zero.
    If plug is engaged, scale is proportional to distance between plug and bottom of socket."""


    num_envs = len(plug_pos)
    reward_scale = torch.zeros((num_envs,), dtype=torch.float32, device=device)


    engaged_idx = np.argwhere(is_plug_engaged_w_socket.cpu().numpy().copy()).squeeze()
    height_dist = plug_pos[engaged_idx, 2] - socket_pos[engaged_idx, 2]


    reward_scale[engaged_idx] = 1.0 / ((height_dist - success_height_thresh) + 0.1)

    return reward_scale


"""
Warp Functions
"""


@wp.func
def mesh_sdf(mesh: wp.uint64, point: wp.vec3, max_dist: float):
    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)
    res = wp.mesh_query_point(mesh, point, max_dist, sign, face_index, face_u, face_v)
    if res:
        closest = wp.mesh_eval_position(mesh, face_index, face_u, face_v)
        return wp.length(point - closest) * sign
    return max_dist


"""
Warp Kernels
"""


@wp.kernel
def get_batch_sdf(
    mesh: wp.uint64,
    queries: wp.array(dtype=wp.vec3),
    sdf_dist: wp.array(dtype=wp.float32),
):
    tid = wp.tid()

    q = queries[tid]
    max_dist = 1.5



    sdf_dist[tid] = mesh_sdf(mesh, q, max_dist)



@wp.kernel
def transform_points(src: wp.array(dtype=wp.vec3), dest: wp.array(dtype=wp.vec3), xform: wp.transform):
    tid = wp.tid()

    p = src[tid]
    m = wp.transform_point(xform, p)

    dest[tid] = m




@wp.kernel
def get_interpen_dist(
    queries: wp.array(dtype=wp.vec3),
    mesh: wp.uint64,
    interpen_dists: wp.array(dtype=wp.float32),
):
    tid = wp.tid()


    q = queries[tid]
    max_dist = 1.5


    sign = float(
        0.0
    )
    face_idx = int(0)
    face_u = float(0.0)
    face_v = float(0.0)


    closest_mesh_point_exists = wp.mesh_query_point(mesh, q, max_dist, sign, face_idx, face_u, face_v)


    if closest_mesh_point_exists:

        p = wp.mesh_eval_position(mesh, face_idx, face_u, face_v)


        delta = q - p
        signed_dist = sign * wp.length(delta)


        if signed_dist < 0.0:

            interpen_dists[tid] = signed_dist
