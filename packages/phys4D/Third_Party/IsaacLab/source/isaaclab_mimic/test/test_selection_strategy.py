




from isaaclab.app import AppLauncher


simulation_app = AppLauncher(headless=True).app

import numpy as np
import torch

import pytest

import isaaclab.utils.math as PoseUtils

from isaaclab_mimic.datagen.datagen_info import DatagenInfo


from isaaclab_mimic.datagen.selection_strategy import (
    NearestNeighborObjectStrategy,
    NearestNeighborRobotDistanceStrategy,
)


NUM_ITERS = 1000


@pytest.fixture
def nearest_neighbor_object_strategy():
    """Fixture for NearestNeighborObjectStrategy."""
    return NearestNeighborObjectStrategy()


@pytest.fixture
def nearest_neighbor_robot_distance_strategy():
    """Fixture for NearestNeighborRobotDistanceStrategy."""
    return NearestNeighborRobotDistanceStrategy()


def test_select_source_demo_identity_orientations_object_strategy(nearest_neighbor_object_strategy):
    """Test the selection of source demonstrations using two distinct object_pose clusters.

    This method generates two clusters of object poses and randomly adjusts the current object pose within
    specified deviations. It then simulates multiple selections to verify that when the current pose is close
    to cluster 1, all selected indices correspond to that cluster, and that the same holds true for cluster 2.
    """


    cluster_1_range_min = 0
    cluster_1_range_max = 4
    cluster_2_range_min = 25
    cluster_2_range_max = 35


    src_object_poses_in_world_cluster_1 = [
        torch.eye(4) + torch.tensor([[0.0, 0.0, 0.0, i], [0.0, 0.0, 0.0, i], [0.0, 0.0, 0.0, i], [0.0, 0.0, 0.0, -1.0]])
        for i in range(cluster_1_range_min, cluster_1_range_max)
    ]


    src_object_poses_in_world_cluster_2 = [
        torch.eye(4) + torch.tensor([[0.0, 0.0, 0.0, i], [0.0, 0.0, 0.0, i], [0.0, 0.0, 0.0, i], [0.0, 0.0, 0.0, -1.0]])
        for i in range(cluster_2_range_min, cluster_2_range_max)
    ]


    src_object_poses_in_world = src_object_poses_in_world_cluster_1 + src_object_poses_in_world_cluster_2


    src_subtask_datagen_infos = [
        DatagenInfo(object_poses={0: object_pose.unsqueeze(0)}) for object_pose in src_object_poses_in_world
    ]


    eef_pose = torch.eye(4)




    max_deviation = 3

    random_index_cluster_1 = np.random.randint(0, len(src_object_poses_in_world_cluster_1))
    cluster_1_curr_object_pose = src_object_poses_in_world_cluster_1[
        random_index_cluster_1
    ].clone()

    cluster_1_curr_object_pose[0, 3] += torch.rand(1).item() * max_deviation
    cluster_1_curr_object_pose[1, 3] += torch.rand(1).item() * max_deviation
    cluster_1_curr_object_pose[2, 3] += torch.rand(1).item() * max_deviation


    selected_indices = [
        nearest_neighbor_object_strategy.select_source_demo(
            eef_pose,
            cluster_1_curr_object_pose,
            src_subtask_datagen_infos,
            pos_weight=1.0,
            rot_weight=1.0,
            nn_k=3,
        )
        for _ in range(NUM_ITERS)
    ]


    assert np.all(
        np.array(selected_indices) < len(src_object_poses_in_world_cluster_1)
    ), "Some selected indices are not part of cluster 1."




    max_deviation = 5

    random_index_cluster_2 = np.random.randint(0, len(src_object_poses_in_world_cluster_2))
    cluster_2_curr_object_pose = src_object_poses_in_world_cluster_2[
        random_index_cluster_2
    ].clone()

    cluster_2_curr_object_pose[0, 3] += torch.rand(1).item() * max_deviation
    cluster_2_curr_object_pose[1, 3] += torch.rand(1).item() * max_deviation
    cluster_2_curr_object_pose[2, 3] += torch.rand(1).item() * max_deviation


    selected_indices = [
        nearest_neighbor_object_strategy.select_source_demo(
            eef_pose,
            cluster_2_curr_object_pose,
            src_subtask_datagen_infos,
            pos_weight=1.0,
            rot_weight=1.0,
            nn_k=6,
        )
        for _ in range(20)
    ]


    assert np.all(
        np.array(selected_indices) < len(src_object_poses_in_world)
    ), "Some selected indices are not part of cluster 2."
    assert np.all(
        np.array(selected_indices) > (len(src_object_poses_in_world_cluster_1) - 1)
    ), "Some selected indices are not part of cluster 2."


def test_select_source_demo_identity_orientations_robot_distance_strategy(nearest_neighbor_robot_distance_strategy):
    """Test the selection of source demonstrations based on identity-oriented poses with varying positions.

    This method generates two clusters of object poses and randomly adjusts the current object pose within
    specified deviations. It then simulates multiple selections to verify that when the current pose is close
    to cluster 1, all selected indices correspond to that cluster, and that the same holds true for cluster 2.
    """


    cluster_1_range_min = 0
    cluster_1_range_max = 4
    cluster_2_range_min = 25
    cluster_2_range_max = 35



    transformed_eef_pose_cluster_1 = [
        torch.eye(4) + torch.tensor([[0, 0, 0, i], [0, 0, 0, i], [0, 0, 0, i], [0, 0, 0, -1]])
        for i in range(cluster_1_range_min, cluster_1_range_max)
    ]


    transformed_eef_pose_cluster_2 = [
        torch.eye(4) + torch.tensor([[0, 0, 0, i], [0, 0, 0, i], [0, 0, 0, i], [0, 0, 0, -1]])
        for i in range(cluster_2_range_min, cluster_2_range_max)
    ]



    transformed_eef_in_world_poses_tensor = torch.stack(transformed_eef_pose_cluster_1 + transformed_eef_pose_cluster_2)


    src_obj_in_world_poses = torch.stack([
        PoseUtils.generate_random_transformation_matrix(pos_boundary=10, rot_boundary=(2 * np.pi))
        for _ in range(transformed_eef_in_world_poses_tensor.shape[0])
    ])




    curr_object_in_world_pose = PoseUtils.generate_random_transformation_matrix(
        pos_boundary=10, rot_boundary=(2 * np.pi)
    )
    world_in_curr_obj_pose = PoseUtils.pose_inv(curr_object_in_world_pose)

    src_eef_in_src_obj_poses = PoseUtils.pose_in_A_to_pose_in_B(
        pose_in_A=transformed_eef_in_world_poses_tensor,
        pose_A_in_B=world_in_curr_obj_pose,
    )

    src_eef_in_world_poses = PoseUtils.pose_in_A_to_pose_in_B(
        pose_in_A=src_eef_in_src_obj_poses,
        pose_A_in_B=src_obj_in_world_poses,
    )


    assert src_obj_in_world_poses.shape[0] == src_eef_in_world_poses.shape[0], (
        "Source object poses and end effector poses does not have the same length. "
        "This is a bug in the test code and not the source code."
    )


    src_subtask_datagen_infos = [
        DatagenInfo(eef_pose=src_eef_in_world_pose.unsqueeze(0), object_poses={0: src_obj_in_world_pose.unsqueeze(0)})
        for src_obj_in_world_pose, src_eef_in_world_pose in zip(src_obj_in_world_poses, src_eef_in_world_poses)
    ]


    max_deviation = 3


    random_index_cluster_1 = np.random.randint(0, len(transformed_eef_pose_cluster_1))
    curr_eef_in_world_pose = transformed_eef_pose_cluster_1[
        random_index_cluster_1
    ].clone()

    curr_eef_in_world_pose[0, 3] += torch.rand(1).item() * max_deviation
    curr_eef_in_world_pose[1, 3] += torch.rand(1).item() * max_deviation
    curr_eef_in_world_pose[2, 3] += torch.rand(1).item() * max_deviation


    selected_indices = [
        nearest_neighbor_robot_distance_strategy.select_source_demo(
            curr_eef_in_world_pose,
            curr_object_in_world_pose,
            src_subtask_datagen_infos,
            pos_weight=1.0,
            rot_weight=1.0,
            nn_k=3,
        )
        for _ in range(20)
    ]


    assert np.all(
        np.array(selected_indices) < len(transformed_eef_pose_cluster_1)
    ), "Some selected indices are not part of cluster 1."


    max_deviation = 3


    random_index_cluster_2 = np.random.randint(0, len(transformed_eef_pose_cluster_2))
    curr_eef_in_world_pose = transformed_eef_pose_cluster_2[
        random_index_cluster_2
    ].clone()

    curr_eef_in_world_pose[0, 3] += torch.rand(1).item() * max_deviation
    curr_eef_in_world_pose[1, 3] += torch.rand(1).item() * max_deviation
    curr_eef_in_world_pose[2, 3] += torch.rand(1).item() * max_deviation


    selected_indices = [
        nearest_neighbor_robot_distance_strategy.select_source_demo(
            curr_eef_in_world_pose,
            curr_object_in_world_pose,
            src_subtask_datagen_infos,
            pos_weight=1.0,
            rot_weight=1.0,
            nn_k=3,
        )
        for _ in range(20)
    ]


    assert np.all(
        np.array(selected_indices) < transformed_eef_in_world_poses_tensor.shape[0]
    ), "Some selected indices are not part of cluster 2."
    assert np.all(
        np.array(selected_indices) > (len(transformed_eef_pose_cluster_1) - 1)
    ), "Some selected indices are not part of cluster 2."
