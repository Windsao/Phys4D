









"""Module contains custom types and dataclasses used across reacher solvers."""
from __future__ import annotations


from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union


import torch


from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.types.tensor import T_BDOF


class ReacherSolveType(Enum):
    """Enum for different types of problems solved with reacher solvers."""

    SINGLE = 0
    GOALSET = 1
    BATCH = 2
    BATCH_GOALSET = 3
    BATCH_ENV = 4
    BATCH_ENV_GOALSET = 5


@dataclass
class ReacherSolveState:
    """Dataclass for storing the current problem type of a reacher solver."""


    solve_type: ReacherSolveType


    batch_size: int


    n_envs: int


    n_goalset: int = 1


    batch_env: bool = False


    batch_retract: bool = False


    batch_mode: bool = False


    num_seeds: Optional[int] = None


    num_ik_seeds: Optional[int] = None


    num_graph_seeds: Optional[int] = None


    num_trajopt_seeds: Optional[int] = None


    num_mpc_seeds: Optional[int] = None


    env_query_idx: Optional[torch.Tensor] = None

    def __post_init__(self):
        """Post init method to set default flags based on input values."""
        if self.n_envs == 1:
            self.batch_env = False
        else:
            self.batch_env = True
        if self.batch_size > 1:
            self.batch_mode = True
        if self.num_seeds is None:
            self.num_seeds = self.num_ik_seeds
        if self.num_seeds is None:
            self.num_seeds = self.num_trajopt_seeds
        if self.num_seeds is None:
            self.num_seeds = self.num_graph_seeds
        if self.num_seeds is None:
            self.num_seeds = self.num_mpc_seeds

    def __eq__(self, other):
        """Custom equality comparison that safely handles tensor fields."""
        if not isinstance(other, ReacherSolveState):
            return False


        if (self.solve_type != other.solve_type or
            self.batch_size != other.batch_size or
            self.n_envs != other.n_envs or
            self.n_goalset != other.n_goalset or
            self.batch_env != other.batch_env or
            self.batch_retract != other.batch_retract or
            self.batch_mode != other.batch_mode or
            self.num_seeds != other.num_seeds or
            self.num_ik_seeds != other.num_ik_seeds or
            self.num_graph_seeds != other.num_graph_seeds or
            self.num_trajopt_seeds != other.num_trajopt_seeds or
            self.num_mpc_seeds != other.num_mpc_seeds):
            return False


        if self.env_query_idx is None and other.env_query_idx is None:
            return True
        if self.env_query_idx is None or other.env_query_idx is None:
            return False

        try:
            return torch.equal(self.env_query_idx, other.env_query_idx)
        except RuntimeError:

            return False

    def clone(self) -> ReacherSolveState:
        """Method to create a deep copy of the current reacher solve state."""
        return ReacherSolveState(
            solve_type=self.solve_type,
            n_envs=self.n_envs,
            batch_size=self.batch_size,
            n_goalset=self.n_goalset,
            batch_env=self.batch_env,
            batch_retract=self.batch_retract,
            batch_mode=self.batch_mode,
            num_seeds=self.num_seeds,
            num_ik_seeds=self.num_ik_seeds,
            num_graph_seeds=self.num_graph_seeds,
            num_trajopt_seeds=self.num_trajopt_seeds,
            num_mpc_seeds=self.num_mpc_seeds,
            env_query_idx=self.env_query_idx,
        )

    def get_batch_size(self) -> int:
        """Method to get total number of optimization problems in the batch including seeds."""
        return self.num_seeds * self.batch_size

    def get_ik_batch_size(self) -> int:
        """Method to get total number of IK problems in the batch including seeds."""
        return self.num_ik_seeds * self.batch_size

    def create_goal_buffer(
        self,
        goal_pose: Pose,
        goal_state: Optional[JointState] = None,
        retract_config: Optional[T_BDOF] = None,
        link_poses: Optional[Dict[str, Pose]] = None,
        tensor_args: TensorDeviceType = TensorDeviceType(),
        env_query_idx: Optional[torch.Tensor] = None,
    ) -> Goal:
        """Method to create a goal buffer from goal pose and other problem targets.

        Args:
            goal_pose: Pose to reach with the end effector.
            goal_state: Joint configuration to reach. If None, the goal is to reach the pose.
            retract_config: Joint configuration to use for L2 regularization. If None,
                `retract_config` from robot configuration file is used. An alternative value is to
                use the start state as the retract configuration.
            link_poses: Dictionary of link poses to reach. This is only required for multi-link
                pose reaching, where the goal is to reach multiple poses with different links.
            tensor_args: Device and floating precision.
            env_query_idx: Optional tensor of shape [batch_size] or [batch_size, 1] specifying
                which env_id each batch element should map to. Only used when batch_env=True.

        Returns:
            Goal buffer with the goal pose, goal state, retract state, and link poses.
        """


        batch_retract = True
        if retract_config is None or retract_config.shape[0] != goal_pose.batch:
            batch_retract = False
        goal_buffer = Goal.create_idx(
            pose_batch_size=self.batch_size,
            batch_env=self.batch_env,
            batch_retract=batch_retract,
            num_seeds=self.num_seeds,
            tensor_args=tensor_args,
            env_query_idx=env_query_idx,
        )
        goal_buffer.goal_pose = goal_pose
        goal_buffer.retract_state = retract_config
        goal_buffer.goal_state = goal_state
        goal_buffer.links_goal_pose = link_poses
        if goal_buffer.links_goal_pose is not None:
            for k in goal_buffer.links_goal_pose.keys():
                goal_buffer.links_goal_pose[k] = goal_buffer.links_goal_pose[k].contiguous()
        return goal_buffer

    def update_goal_buffer(
        self,
        goal_pose: Pose,
        goal_state: Optional[JointState] = None,
        retract_config: Optional[T_BDOF] = None,
        link_poses: Optional[List[Pose]] = None,
        current_solve_state: Optional[ReacherSolveState] = None,
        current_goal_buffer: Optional[Goal] = None,
        tensor_args: TensorDeviceType = TensorDeviceType(),
        env_query_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[ReacherSolveState, Goal, bool]:
        """Method to update the goal buffer with new goal pose and other problem targets.

        Args:
            goal_pose: Pose to reach with the end effector.
            goal_state: Joint configuration to reach. If None, the goal is to reach the pose.
            retract_config: Joint configuration to use for L2 regularization. If None,
                `retract_config` from robot configuration file is used. An alternative value is to
                use the start state as the retract configuration.
            link_poses: Dictionary of link poses to reach. This is only required for multi-link
                pose reaching, where the goal is to reach multiple poses with different links. To
                use this,
                :attr:`curobo.cuda_robot_model.cuda_robot_model.CudaRobotModelConfig.link_names`
                should have the link names to reach.
            current_solve_state: Current reacher solve state.
            current_goal_buffer: Current goal buffer.
            tensor_args: Device and floating precision.

        Returns:
            Tuple of updated reacher solve state, goal buffer, and a flag indicating if the goal
            buffer reference has changed which is useful to break existing CUDA Graphs.
        """
        solve_state = self

        update_reference = False
        if (
            current_solve_state is None
            or current_goal_buffer is None
            or (current_goal_buffer.retract_state is None and retract_config is not None)
            or (current_goal_buffer.goal_state is None and goal_state is not None)
            or (current_goal_buffer.links_goal_pose is None and link_poses is not None)
        ):
            update_reference = True

        elif current_solve_state != solve_state:
            new_goal_pose = get_padded_goalset(
                solve_state, current_solve_state, current_goal_buffer, goal_pose
            )
            if new_goal_pose is not None:
                goal_pose = new_goal_pose
            else:
                update_reference = True

        if update_reference:
            current_solve_state = solve_state
            current_goal_buffer = solve_state.create_goal_buffer(
                goal_pose, goal_state, retract_config, link_poses, tensor_args, env_query_idx
            )
        else:
            current_goal_buffer.goal_pose.copy_(goal_pose)
            if retract_config is not None:
                current_goal_buffer.retract_state.copy_(retract_config)
            if goal_state is not None:
                current_goal_buffer.goal_state.copy_(goal_state)
            if link_poses is not None:
                for k in link_poses.keys():
                    current_goal_buffer.links_goal_pose[k].copy_(link_poses[k].contiguous())

        return current_solve_state, current_goal_buffer, update_reference

    def update_goal(
        self,
        goal: Goal,
        current_solve_state: Optional[ReacherSolveState] = None,
        current_goal_buffer: Optional[Goal] = None,
        tensor_args: TensorDeviceType = TensorDeviceType(),
        env_query_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[ReacherSolveState, Goal, bool]:
        """Method to update the goal buffer with values from new Rollout goal.

        Args:
            goal: Rollout goal to update the goal buffer.
            current_solve_state: Current reacher solve state.
            current_goal_buffer: Current goal buffer.
            tensor_args: Device and floating precision.
            env_query_idx: Optional tensor specifying which env_id each batch element should map to.

        Returns:
            Tuple of updated reacher solve state, goal buffer, and a flag indicating if the goal
            buffer reference has changed which is useful to break existing CUDA Graphs.
        """
        solve_state = self
        update_reference = False
        if (
            current_solve_state is None
            or current_goal_buffer is None
            or (current_goal_buffer.goal_state is None and goal.goal_state is not None)
            or (current_goal_buffer.goal_state is not None and goal.goal_state is None)
        ):
            update_reference = True
        elif current_solve_state != solve_state:
            new_goal_pose = get_padded_goalset(
                solve_state, current_solve_state, current_goal_buffer, goal.goal_pose
            )
            if new_goal_pose is not None:
                goal = goal.clone()
                goal.goal_pose = new_goal_pose

            else:
                update_reference = True

        if update_reference:
            current_solve_state = solve_state




            if (solve_state.num_seeds == 1 and goal.batch == solve_state.batch_size
                and goal.batch_pose_idx is not None and goal.batch_pose_idx.shape[0] == goal.batch):

                current_goal_buffer = goal.clone()
            else:


                if goal.batch <= 0:
                    if goal.goal_pose is not None and goal.goal_pose.batch > 0:
                        goal.batch = goal.goal_pose.batch
                    elif goal.current_state is not None and goal.current_state.position.shape[0] > 0:
                        goal.batch = goal.current_state.position.shape[0]
                    elif goal.goal_state is not None and goal.goal_state.position.shape[0] > 0:
                        goal.batch = goal.goal_state.position.shape[0]

                batch_size_to_use = solve_state.batch_size if solve_state.batch_size > 0 else goal.batch
                if batch_size_to_use <= 0:
                    raise ValueError(f"Invalid batch_size: {batch_size_to_use}, goal.batch: {goal.batch}, solve_state.batch_size: {solve_state.batch_size}, goal_pose.batch: {goal.goal_pose.batch if goal.goal_pose is not None else None}")
                current_goal_buffer = goal.create_index_buffers(
                    batch_size_to_use,
                    solve_state.batch_env,
                    solve_state.batch_retract,
                    solve_state.num_seeds,
                    tensor_args,
                    env_query_idx=env_query_idx,
                )
        else:
            current_goal_buffer.copy_(goal, update_idx_buffers=False)
        return current_solve_state, current_goal_buffer, update_reference


@dataclass
class MotionGenSolverState:
    """Dataclass for storing the current state of a motion generation solver."""

    solve_type: ReacherSolveType
    ik_solve_state: ReacherSolveState
    trajopt_solve_state: ReacherSolveState


def get_padded_goalset(
    solve_state: ReacherSolveState,
    current_solve_state: ReacherSolveState,
    current_goal_buffer: Goal,
    new_goal_pose: Pose,
) -> Union[Pose, None]:
    """Method to pad number of goals in goalset to match the cached goal buffer.

    This allows for creating a goalset problem with large number of goals during the first call,
    and subsequent calls can have fewer goals. This function will pad the new goalset with the
    first goal to match the cached goal buffer's shape.

    Args:
        solve_state: New problem's solve state.
        current_solve_state: Current solve state.
        current_goal_buffer: Current goal buffer.
        new_goal_pose: Padded goal pose to match the cached goal buffer's shape.

    Returns:
        Padded goal pose to match the cached goal buffer's shape. If the new goal can't be padded,
        returns None.
    """
    if (
        current_solve_state.solve_type == ReacherSolveType.GOALSET
        and solve_state.solve_type == ReacherSolveType.SINGLE
    ):




        goal_pose = current_goal_buffer.goal_pose.clone()
        goal_pose.position[:] = new_goal_pose.position
        goal_pose.quaternion[:] = new_goal_pose.quaternion
        return goal_pose

    elif (
        current_solve_state.solve_type == ReacherSolveType.BATCH_GOALSET
        and solve_state.solve_type == ReacherSolveType.BATCH
        and new_goal_pose.n_goalset <= current_solve_state.n_goalset
        and new_goal_pose.batch == current_solve_state.batch_size
    ):
        goal_pose = current_goal_buffer.goal_pose.clone()
        if len(new_goal_pose.position.shape) == 2:
            new_goal_pose = new_goal_pose.unsqueeze(1)
        goal_pose.position[..., :, :] = new_goal_pose.position
        goal_pose.quaternion[..., :, :] = new_goal_pose.quaternion
        return goal_pose
    elif (
        current_solve_state.solve_type == ReacherSolveType.BATCH_ENV_GOALSET
        and solve_state.solve_type == ReacherSolveType.BATCH_ENV
        and new_goal_pose.n_goalset <= current_solve_state.n_goalset
        and new_goal_pose.batch == current_solve_state.batch_size
    ):
        goal_pose = current_goal_buffer.goal_pose.clone()
        if len(new_goal_pose.position.shape) == 2:
            new_goal_pose = new_goal_pose.unsqueeze(1)
        goal_pose.position[..., :, :] = new_goal_pose.position
        goal_pose.quaternion[..., :, :] = new_goal_pose.quaternion
        return goal_pose

    elif (
        current_solve_state.solve_type == ReacherSolveType.GOALSET
        and solve_state.solve_type == ReacherSolveType.GOALSET
        and new_goal_pose.n_goalset <= current_solve_state.n_goalset
    ):
        goal_pose = current_goal_buffer.goal_pose.clone()
        goal_pose.position[..., : new_goal_pose.n_goalset, :] = new_goal_pose.position
        goal_pose.quaternion[..., : new_goal_pose.n_goalset, :] = new_goal_pose.quaternion
        goal_pose.position[..., new_goal_pose.n_goalset :, :] = new_goal_pose.position[..., :1, :]
        goal_pose.quaternion[..., new_goal_pose.n_goalset :, :] = new_goal_pose.quaternion[
            ..., :1, :
        ]

        return goal_pose
    elif (
        current_solve_state.solve_type == ReacherSolveType.BATCH_GOALSET
        and solve_state.solve_type == ReacherSolveType.BATCH_GOALSET
        and new_goal_pose.n_goalset <= current_solve_state.n_goalset
        and new_goal_pose.batch == current_solve_state.batch_size
    ):
        goal_pose = current_goal_buffer.goal_pose.clone()
        goal_pose.position[..., : new_goal_pose.n_goalset, :] = new_goal_pose.position
        goal_pose.quaternion[..., : new_goal_pose.n_goalset, :] = new_goal_pose.quaternion
        goal_pose.position[..., new_goal_pose.n_goalset :, :] = new_goal_pose.position[..., :1, :]
        goal_pose.quaternion[..., new_goal_pose.n_goalset :, :] = new_goal_pose.quaternion[
            ..., :1, :
        ]
        return goal_pose
    elif (
        current_solve_state.solve_type == ReacherSolveType.BATCH_ENV_GOALSET
        and solve_state.solve_type == ReacherSolveType.BATCH_ENV_GOALSET
        and new_goal_pose.n_goalset <= current_solve_state.n_goalset
        and new_goal_pose.batch == current_solve_state.batch_size
    ):
        goal_pose = current_goal_buffer.goal_pose.clone()
        goal_pose.position[..., : new_goal_pose.n_goalset, :] = new_goal_pose.position
        goal_pose.quaternion[..., : new_goal_pose.n_goalset, :] = new_goal_pose.quaternion
        goal_pose.position[..., new_goal_pose.n_goalset :, :] = new_goal_pose.position[..., :1, :]
        goal_pose.quaternion[..., new_goal_pose.n_goalset :, :] = new_goal_pose.quaternion[
            ..., :1, :
        ]
        return goal_pose
    return None
