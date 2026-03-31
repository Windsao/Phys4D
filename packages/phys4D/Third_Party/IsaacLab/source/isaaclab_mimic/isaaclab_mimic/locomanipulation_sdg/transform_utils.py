




import torch

import isaaclab.utils.math as math_utils


def transform_mul(transform_a: torch.Tensor, transform_b: torch.Tensor) -> torch.Tensor:
    """Multiply two translation, quaternion pose representations by converting to matrices first."""

    pos_a, quat_a = transform_a[..., :3], transform_a[..., 3:]
    pos_b, quat_b = transform_b[..., :3], transform_b[..., 3:]


    rot_a = math_utils.matrix_from_quat(quat_a)
    rot_b = math_utils.matrix_from_quat(quat_b)


    pose_a = math_utils.make_pose(pos_a, rot_a)
    pose_b = math_utils.make_pose(pos_b, rot_b)


    result_pose = torch.matmul(pose_a, pose_b)


    result_pos, result_rot = math_utils.unmake_pose(result_pose)


    result_quat = math_utils.quat_from_matrix(result_rot)

    return torch.cat([result_pos, result_quat], dim=-1)


def transform_inv(transform: torch.Tensor) -> torch.Tensor:
    """Invert a translation, quaternion format transformation using math_utils."""
    pos, quat = transform[..., :3], transform[..., 3:]
    quat_inv = math_utils.quat_inv(quat)
    pos_inv = math_utils.quat_apply(quat_inv, -pos)
    return torch.cat([pos_inv, quat_inv], dim=-1)


def transform_relative_pose(world_pose: torch.Tensor, src_frame_pose: torch.Tensor, dst_frame_pose: torch.Tensor):
    """Compute the relative pose with respect to a source frame, and apply this relative pose to a destination frame."""
    pose = transform_mul(dst_frame_pose, transform_mul(transform_inv(src_frame_pose), world_pose))
    return pose
