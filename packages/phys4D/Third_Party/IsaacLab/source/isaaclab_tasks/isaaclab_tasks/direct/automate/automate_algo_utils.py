




import os
import re
import subprocess
import sys
import torch
import trimesh

import warp as wp

print("Python Executable:", sys.executable)
print("Python Path:", sys.path)

base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "."))
sys.path.append(base_dir)

from isaaclab.utils.assets import retrieve_file_path

"""
Util Functions
"""


def parse_cuda_version(version_string):
    """
       Parse CUDA version string into comparable tuple of (major, minor, patch).

       Args:
           version_string: Version string like "12.8.9" or "11.2"

       Returns:
           Tuple of (major, minor, patch) as integers, where patch defaults to 0 iff
    not present.

       Example:
           "12.8.9" -> (12, 8, 9)
           "11.2" -> (11, 2, 0)
    """
    parts = version_string.split(".")
    major = int(parts[0])
    minor = int(parts[1]) if len(parts) > 1 else 0
    patch = int(parts[2]) if len(parts) > 2 else 0
    return (major, minor, patch)


def get_cuda_version():
    try:

        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, check=True)
        output = result.stdout


        match = re.search(r"V(\d+\.\d+(\.\d+)?)", output)
        if match:
            return parse_cuda_version(match.group(1))
        else:
            print("CUDA version not found in output.")
            return None
    except FileNotFoundError:
        print("nvcc command not found. Is CUDA installed and in your PATH?")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error executing nvcc: {e.stderr}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def get_gripper_open_width(obj_filepath):

    retrieve_file_path(obj_filepath, download_dir="./")
    obj_mesh = trimesh.load_mesh(os.path.basename(obj_filepath))

    aabb = obj_mesh.bounds

    return min(0.04, (aabb[1][1] - aabb[0][1]) / 1.25)


"""
Imitation Reward
"""


def get_closest_state_idx(ref_traj, curr_ee_pos):
    """Find the index of the closest state in reference trajectory."""


    traj_len = ref_traj.shape[1]
    num_envs = curr_ee_pos.shape[0]


    dist_from_all_state = torch.cdist(ref_traj.unsqueeze(0), curr_ee_pos.reshape(-1, 1, 1, 3), p=2)


    dist_from_all_state_flatten = dist_from_all_state.reshape(num_envs, -1)


    min_dist_per_env = torch.amin(dist_from_all_state_flatten, dim=-1)


    min_dist_idx = torch.argmin(dist_from_all_state_flatten, dim=-1)



    min_dist_traj_idx = min_dist_idx // traj_len
    min_dist_step_idx = min_dist_idx % traj_len

    return min_dist_traj_idx, min_dist_step_idx, min_dist_per_env


def get_reward_mask(ref_traj, curr_ee_pos, tolerance):

    _, min_dist_step_idx, _ = get_closest_state_idx(ref_traj, curr_ee_pos)
    selected_steps = torch.index_select(
        ref_traj, dim=1, index=min_dist_step_idx
    )

    x_min = torch.amin(selected_steps[:, :, 0], dim=0) - tolerance
    x_max = torch.amax(selected_steps[:, :, 0], dim=0) + tolerance
    y_min = torch.amin(selected_steps[:, :, 1], dim=0) - tolerance
    y_max = torch.amax(selected_steps[:, :, 1], dim=0) + tolerance

    x_in_range = torch.logical_and(torch.lt(curr_ee_pos[:, 0], x_max), torch.gt(curr_ee_pos[:, 0], x_min))
    y_in_range = torch.logical_and(torch.lt(curr_ee_pos[:, 1], y_max), torch.gt(curr_ee_pos[:, 1], y_min))
    pos_in_range = torch.logical_and(x_in_range, y_in_range).int()

    return pos_in_range


def get_imitation_reward_from_dtw(ref_traj, curr_ee_pos, prev_ee_traj, criterion, device):
    """Get imitation reward based on dynamic time warping."""

    soft_dtw = torch.zeros((curr_ee_pos.shape[0]), device=device)
    prev_ee_pos = prev_ee_traj[:, 0, :]
    min_dist_traj_idx, min_dist_step_idx, min_dist_per_env = get_closest_state_idx(ref_traj, prev_ee_pos)

    for i in range(curr_ee_pos.shape[0]):
        traj_idx = min_dist_traj_idx[i]
        step_idx = min_dist_step_idx[i]
        curr_ee_pos_i = curr_ee_pos[i].reshape(1, 3)


        traj = ref_traj[traj_idx, step_idx:, :].reshape((1, -1, 3))

        _, curr_step_idx, _ = get_closest_state_idx(traj, curr_ee_pos_i)

        if curr_step_idx == 0:
            selected_pos = ref_traj[traj_idx, step_idx, :].reshape((1, 1, 3))
            selected_traj = torch.cat([selected_pos, selected_pos], dim=1)
        else:
            selected_traj = ref_traj[traj_idx, step_idx : (curr_step_idx + step_idx), :].reshape((1, -1, 3))
        eef_traj = torch.cat((prev_ee_traj[i, 1:, :], curr_ee_pos_i)).reshape((1, -1, 3))
        soft_dtw[i] = criterion(eef_traj, selected_traj)

    w_task_progress = 1 - (min_dist_step_idx / ref_traj.shape[1])


    imitation_rwd = 1 - torch.tanh(soft_dtw)

    return imitation_rwd * w_task_progress


"""
Sampling-Based Curriculum (SBC)
"""


def get_new_max_disp(curr_success, cfg_task, curriculum_height_bound, curriculum_height_step, curr_max_disp):
    """Update max downward displacement of plug at beginning of episode, based on success rate."""

    if curr_success > cfg_task.curriculum_success_thresh:

        new_max_disp = torch.where(
            curr_max_disp + curriculum_height_step[:, 0] < curriculum_height_bound[:, 1],
            curr_max_disp + curriculum_height_step[:, 0],
            curriculum_height_bound[:, 1],
        )
    elif curr_success < cfg_task.curriculum_failure_thresh:

        new_max_disp = torch.where(
            curr_max_disp + curriculum_height_step[:, 1] > curriculum_height_bound[:, 0],
            curr_max_disp + curriculum_height_step[:, 1],
            curriculum_height_bound[:, 0],
        )
    else:

        new_max_disp = curr_max_disp

    return new_max_disp


"""
Bonus and Success Checking
"""


def check_plug_close_to_socket(keypoints_plug, keypoints_socket, dist_threshold, progress_buf):
    """Check if plug is close to socket."""


    keypoint_dist = torch.norm(keypoints_socket - keypoints_plug, p=2, dim=-1)


    is_plug_close_to_socket = torch.where(
        torch.mean(keypoint_dist, dim=-1) < dist_threshold,
        torch.ones_like(progress_buf),
        torch.zeros_like(progress_buf),
    )

    return is_plug_close_to_socket


def check_plug_inserted_in_socket(
    plug_pos, socket_pos, disassembly_dist, keypoints_plug, keypoints_socket, close_error_thresh, progress_buf
):
    """Check if plug is inserted in socket."""


    is_plug_below_insertion_height = plug_pos[:, 2] < socket_pos[:, 2] + disassembly_dist
    is_plug_above_table_height = plug_pos[:, 2] > socket_pos[:, 2]

    is_plug_height_success = torch.logical_and(is_plug_below_insertion_height, is_plug_above_table_height)




    is_plug_close_to_socket = check_plug_close_to_socket(
        keypoints_plug=keypoints_plug,
        keypoints_socket=keypoints_socket,
        dist_threshold=close_error_thresh,
        progress_buf=progress_buf,
    )


    is_plug_inserted_in_socket = torch.logical_and(is_plug_height_success, is_plug_close_to_socket)

    return is_plug_inserted_in_socket


def get_curriculum_reward_scale(curr_max_disp, curriculum_height_bound):
    """Compute reward scale for SBC."""




    curr_stage_diff = curr_max_disp - curriculum_height_bound[:, 0]



    final_stage_diff = curriculum_height_bound[:, 1] - curriculum_height_bound[:, 0]


    reward_scale = curr_stage_diff / final_stage_diff + 1.0

    return reward_scale.mean()


"""
Warp Kernels
"""



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
