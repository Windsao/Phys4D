




"""
Selection strategies used by Isaac Lab Mimic to select subtask segments from
source human demonstrations.
"""
import abc
import torch

import isaaclab.utils.math as PoseUtils


REGISTERED_SELECTION_STRATEGIES = {}


def make_selection_strategy(name, *args, **kwargs):
    """
    Creates an instance of a selection strategy class, specified by @name,
    which is used to look it up in the registry.
    """
    assert_selection_strategy_exists(name)
    return REGISTERED_SELECTION_STRATEGIES[name](*args, **kwargs)


def register_selection_strategy(cls):
    """
    Register selection strategy class into global registry.
    """
    ignore_classes = ["SelectionStrategy"]
    if cls.__name__ not in ignore_classes:
        REGISTERED_SELECTION_STRATEGIES[cls.NAME] = cls


def assert_selection_strategy_exists(name):
    """
    Allow easy way to check if selection strategy exists.
    """
    if name not in REGISTERED_SELECTION_STRATEGIES:
        raise Exception(
            "assert_selection_strategy_exists: name {} not found. Make sure it is a registered selection strategy"
            " among {}".format(name, ", ".join(REGISTERED_SELECTION_STRATEGIES))
        )


class SelectionStrategyMeta(type):
    """
    This metaclass adds selection strategy classes into the global registry.
    """

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        register_selection_strategy(cls)
        return cls


class SelectionStrategy(metaclass=SelectionStrategyMeta):
    """
    Defines methods and functions for selection strategies to implement.
    """

    def __init__(self):
        pass

    @property
    @classmethod
    def NAME(self):
        """
        This name (str) will be used to register the selection strategy class in the global
        registry.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def select_source_demo(
        self,
        eef_pose,
        object_pose,
        src_subtask_datagen_infos,
    ):
        """
        Selects source demonstration index using the current robot pose, relevant object pose
        for the current subtask, and relevant information from the source demonstrations for the
        current subtask.

        Args:
            eef_pose (torch.Tensor): current 4x4 eef pose
            object_pose (torch.Tensor): current 4x4 object pose, for the object in this subtask
            src_subtask_datagen_infos (list): DatagenInfo instance for the relevant subtask segment
                in the source demonstrations

        Returns:
            source_demo_ind (int): index of source demonstration - indicates which source subtask segment to use
        """
        raise NotImplementedError


class RandomStrategy(SelectionStrategy):
    """
    Pick source demonstration randomly.
    """


    NAME = "random"

    def select_source_demo(
        self,
        eef_pose,
        object_pose,
        src_subtask_datagen_infos,
    ):
        """
        Selects source demonstration index using the current robot pose, relevant object pose
        for the current subtask, and relevant information from the source demonstrations for the
        current subtask.

        Args:
            eef_pose (torch.Tensor): current 4x4 eef pose
            object_pose (torch.Tensor): current 4x4 object pose, for the object in this subtask
            src_subtask_datagen_infos (list): DatagenInfo instance for the relevant subtask segment
                in the source demonstrations

        Returns:
            source_demo_ind (int): index of source demonstration - indicates which source subtask segment to use
        """


        n_src_demo = len(src_subtask_datagen_infos)
        return torch.randint(0, n_src_demo, (1,)).item()


class NearestNeighborObjectStrategy(SelectionStrategy):
    """
    Pick source demonstration to be the one with the closest object pose to the object
    in the current scene.
    """


    NAME = "nearest_neighbor_object"

    def select_source_demo(
        self,
        eef_pose,
        object_pose,
        src_subtask_datagen_infos,
        pos_weight=1.0,
        rot_weight=1.0,
        nn_k=3,
    ):
        """
        Selects source demonstration index using the current robot pose, relevant object pose
        for the current subtask, and relevant information from the source demonstrations for the
        current subtask.

        Args:
            eef_pose (torch.Tensor): current 4x4 eef pose
            object_pose (torch.Tensor): current 4x4 object pose, for the object in this subtask
            src_subtask_datagen_infos (list): DatagenInfo instance for the relevant subtask segment
                in the source demonstrations
            pos_weight (float): weight on position for minimizing pose distance
            rot_weight (float): weight on rotation for minimizing pose distance
            nn_k (int): pick source demo index uniformly at randomly from the top @nn_k nearest neighbors

        Returns:
            source_demo_ind (int): index of source demonstration - indicates which source subtask segment to use
        """


        src_object_poses = []
        for di in src_subtask_datagen_infos:
            src_obj_pose = list(di.object_poses.values())
            assert len(src_obj_pose) == 1

            src_object_poses.append(src_obj_pose[0][0])
        src_object_poses = torch.stack(src_object_poses)


        all_src_obj_pos, all_src_obj_rot = PoseUtils.unmake_pose(src_object_poses)
        obj_pos, obj_rot = PoseUtils.unmake_pose(object_pose)


        obj_pos = obj_pos.view(-1, 3)
        obj_rot_T = obj_rot.transpose(0, 1).view(-1, 3, 3)


        pos_dists = torch.sqrt(((all_src_obj_pos - obj_pos) ** 2).sum(dim=-1))





        delta_R = torch.matmul(all_src_obj_rot, obj_rot_T)
        arc_cos_in = (torch.diagonal(delta_R, dim1=-2, dim2=-1).sum(dim=-1) - 1.0) / 2.0
        arc_cos_in = torch.clamp(arc_cos_in, -1.0, 1.0)
        rot_dists = torch.acos(arc_cos_in)


        dists_to_minimize = pos_weight * pos_dists + rot_weight * rot_dists


        nn_k = min(nn_k, len(dists_to_minimize))


        rand_k = torch.randint(0, nn_k, (1,)).item()
        top_k_neighbors_in_order = torch.argsort(dists_to_minimize)[:nn_k]
        return top_k_neighbors_in_order[rand_k]


class NearestNeighborRobotDistanceStrategy(SelectionStrategy):
    """
    Pick source demonstration to be the one that minimizes the distance the robot
    end effector will need to travel from the current pose to the first pose
    in the transformed segment.
    """


    NAME = "nearest_neighbor_robot_distance"

    def select_source_demo(
        self,
        eef_pose,
        object_pose,
        src_subtask_datagen_infos,
        pos_weight=1.0,
        rot_weight=1.0,
        nn_k=3,
    ):
        """
        Selects source demonstration index using the current robot pose, relevant object pose
        for the current subtask, and relevant information from the source demonstrations for the
        current subtask.

        Args:
            eef_pose (torch.Tensor): current 4x4 eef pose
            object_pose (torch.Tensor): current 4x4 object pose, for the object in this subtask
            src_subtask_datagen_infos (list): DatagenInfo instance for the relevant subtask segment
                in the source demonstrations
            pos_weight (float): weight on position for minimizing pose distance
            rot_weight (float): weight on rotation for minimizing pose distance
            nn_k (int): pick source demo index uniformly at randomly from the top @nn_k nearest neighbors

        Returns:
            source_demo_ind (int): index of source demonstration - indicates which source subtask segment to use
        """


        src_eef_poses = []
        src_object_poses = []
        for di in src_subtask_datagen_infos:

            src_eef_poses.append(di.eef_pose[0])

            src_obj_pose = list(di.object_poses.values())
            assert len(src_obj_pose) == 1
            src_object_poses.append(src_obj_pose[0][0])
        src_eef_poses = torch.stack(src_eef_poses)
        src_object_poses = torch.stack(src_object_poses)



        src_object_poses_inv = PoseUtils.pose_inv(src_object_poses)
        src_eef_poses_in_obj = PoseUtils.pose_in_A_to_pose_in_B(
            pose_in_A=src_eef_poses,
            pose_A_in_B=src_object_poses_inv,
        )



        transformed_eef_poses = PoseUtils.pose_in_A_to_pose_in_B(
            pose_in_A=src_eef_poses_in_obj,
            pose_A_in_B=object_pose,
        )


        all_transformed_eef_pos, all_transformed_eef_rot = PoseUtils.unmake_pose(transformed_eef_poses)
        eef_pos, eef_rot = PoseUtils.unmake_pose(eef_pose)





        eef_pos = eef_pos.view(-1, 3)
        eef_rot_T = eef_rot.transpose(0, 1).view(-1, 3, 3)


        pos_dists = torch.sqrt(((all_transformed_eef_pos - eef_pos) ** 2).sum(dim=-1))





        delta_R = torch.matmul(all_transformed_eef_rot, eef_rot_T)
        arc_cos_in = (torch.diagonal(delta_R, dim1=-2, dim2=-1).sum(dim=-1) - 1.0) / 2.0
        arc_cos_in = torch.clamp(arc_cos_in, -1.0, 1.0)
        rot_dists = torch.acos(arc_cos_in)


        dists_to_minimize = pos_weight * pos_dists + rot_weight * rot_dists


        nn_k = min(nn_k, len(dists_to_minimize))


        rand_k = torch.randint(0, nn_k, (1,)).item()
        top_k_neighbors_in_order = torch.argsort(dists_to_minimize)[:nn_k]
        return top_k_neighbors_in_order[rand_k]
