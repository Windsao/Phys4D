from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from magicsim.Env.Robot.mdp.action_manager import ActionTerm

if TYPE_CHECKING:
    from magicsim.Env.Environment.Isaac.IsaacRLEnv import IsaacRLEnv

    from magicsim.Env.Robot.mdp import actions_cfg

from .differential_ik import DifferentialIKController

from gymnasium import spaces


class JointPositionToLimitsAction(ActionTerm):
    """Joint position action term that scales the input actions to the joint limits and applies them to the
    articulation's joints.

    This class is similar to the :class:`JointPositionAction` class. However, it performs additional
    re-scaling of input actions to the actuator joint position limits.

    While processing the actions, it performs the following operations:

    1. Apply scaling to the raw actions based on :attr:`actions_cfg.JointPositionToLimitsActionCfg.scale`.
    2. Clip the scaled actions to the range [-1, 1] and re-scale them to the joint limits if
       :attr:`actions_cfg.JointPositionToLimitsActionCfg.rescale_to_limits` is set to True.

    The processed actions are then sent as position commands to the articulation's joints.
    """

    cfg: actions_cfg.JointPositionToLimitsActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor | float
    """The scaling factor applied to the input action."""
    _clip: torch.Tensor
    """The clip applied to the input action."""
    _action_space: spaces.Box
    """The action space of the action term."""

    def __init__(
        self, cfg: actions_cfg.JointPositionToLimitsActionCfg, env: IsaacRLEnv
    ):
        super().__init__(cfg, env)

        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names
        )
        self._num_joints = len(self._joint_ids)

        omni.log.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )

        self._raw_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )
        self._processed_actions = torch.zeros_like(self._raw_actions)

        if isinstance(cfg.scale, (float, int)):
            self._scale = float(cfg.scale)
        elif isinstance(cfg.scale, dict):
            self._scale = torch.ones(self.num_envs, self.action_dim, device=self.device)

            index_list, _, value_list = string_utils.resolve_matching_names_values(
                self.cfg.scale, self._joint_names
            )
            self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(
                f"Unsupported scale type: {type(cfg.scale)}. Supported types are float and dict."
            )

        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor(
                    [[-float("inf"), float("inf")]], device=self.device
                ).repeat(self.num_envs, self.action_dim, 1)
                index_list, _, value_list = string_utils.resolve_matching_names_values(
                    self.cfg.clip, self._joint_names
                )
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(
                    f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict."
                )

        if self.cfg.rescale_to_limits:
            self._action_space_raw = torch.stack(
                [
                    torch.tensor([-1.0] * self.action_dim),
                    torch.tensor([1.0] * self.action_dim),
                ],
                dim=1,
            )
            if self.cfg.clip is not None:
                self._action_space_raw = torch.clamp(
                    self._action_space_raw,
                    min=self.cfg.clip[:, 0],
                    max=self.cfg.clip[:, 1],
                )
            self._action_space = spaces.Box(
                low=self._action_space_raw[0].cpu().numpy(),
                high=self._action_space_raw[1].cpu().numpy(),
                dtype=torch.float32,
            )
        else:
            self._action_space_raw = torch.clone(
                self._asset.data.soft_joint_pos_limits[0, self._joint_ids, :]
            )
            if self.cfg.clip is not None:
                self._action_space_raw = torch.clamp(
                    self._action_space_raw,
                    min=self.cfg.clip[:, 0],
                    max=self.cfg.clip[:, 1],
                )

            self._action_space_raw = self._action_space_raw.T
            self._action_space = spaces.Box(
                low=self._action_space_raw[0].cpu().numpy(),
                high=self._action_space_raw[1].cpu().numpy(),
            )

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def env_ids(self) -> torch.Tensor:
        return self._env_ids

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    """
    Operations.
    """

    def process_actions(
        self, actions: torch.Tensor, env_ids: Sequence[int] | None = None
    ):
        if env_ids is None:
            self._env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            if not isinstance(env_ids, torch.Tensor):
                self._env_ids = torch.tensor(
                    env_ids, device=self.device, dtype=torch.int32
                )
            else:
                self._env_ids = env_ids
        assert actions.shape[0] == len(env_ids), (
            f"Expected actions shape[0] to be {self.num_envs}, but got {actions.shape[0]}"
        )

        self._raw_actions[self.env_ids] = actions

        self._processed_actions = self._raw_actions[self.env_ids] * self._scale
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[self.env_ids, :, 0],
                max=self._clip[self.env_ids, :, 1],
            )

        if self.cfg.rescale_to_limits:
            actions = self._processed_actions.clamp(-1.0, 1.0)

            actions = math_utils.unscale_transform(
                actions,
                self._asset.data.soft_joint_pos_limits[
                    self.env_ids.unsqueeze(1), self._joint_ids, 0
                ],
                self._asset.data.soft_joint_pos_limits[
                    self.env_ids.unsqueeze(1), self._joint_ids, 1
                ],
            )
            self._processed_actions[:] = actions[:]
        else:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._asset.data.soft_joint_pos_limits[
                    self.env_ids.unsqueeze(1), self._joint_ids, 0
                ],
                max=self._asset.data.soft_joint_pos_limits[
                    self.env_ids.unsqueeze(1), self._joint_ids, 1
                ],
            )

    def apply_actions(self):
        assert self.processed_actions.shape[0] == len(self.env_ids), (
            f"Expected processed actions shape[0] to be {self.num_envs}, but got {self.processed_actions.shape[0]}"
        )
        self._asset.set_joint_position_target(
            self.processed_actions, joint_ids=self._joint_ids, env_ids=self.env_ids
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        pass
        self._raw_actions[env_ids] = 0.0


class BinaryJointAction(ActionTerm):
    """Base class for binary joint actions.

    This action term maps a binary action to the *open* or *close* joint configurations. These configurations are
    specified through the :class:`BinaryJointActionCfg` object. If the input action is a float vector, the action
    is considered binary based on the sign of the action values.

    Based on above, we follow the following convention for the binary action:

    1. Open action: 1 (bool) or positive values (float).
    2. Close action: 0 (bool) or negative values (float).

    The action term can mostly be used for gripper actions, where the gripper is either open or closed. This
    helps in devising a mimicking mechanism for the gripper, since in simulation it is often not possible to
    add such constraints to the gripper.
    """

    cfg: actions_cfg.BinaryJointActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _clip: torch.Tensor
    """The clip applied to the input action."""
    _action_space: spaces.Box
    """The action space of the action term."""

    def __init__(self, cfg: actions_cfg.BinaryJointActionCfg, env: IsaacRLEnv) -> None:
        super().__init__(cfg, env)

        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names
        )
        self._num_joints = len(self._joint_ids)

        omni.log.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )

        self._raw_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )
        self._processed_actions = torch.zeros_like(self._raw_actions)

        self._open_command = torch.zeros(self._num_joints, device=self.device)
        index_list, name_list, value_list = string_utils.resolve_matching_names_values(
            self.cfg.open_command_expr, self._joint_names
        )
        if len(index_list) != self._num_joints:
            raise ValueError(
                f"Could not resolve all joints for the action term. Missing: {set(self._joint_names) - set(name_list)}"
            )
        self._open_command[index_list] = torch.tensor(value_list, device=self.device)

        self._close_command = torch.zeros_like(self._open_command)
        index_list, name_list, value_list = string_utils.resolve_matching_names_values(
            self.cfg.close_command_expr, self._joint_names
        )
        if len(index_list) != self._num_joints:
            raise ValueError(
                f"Could not resolve all joints for the action term. Missing: {set(self._joint_names) - set(name_list)}"
            )
        self._close_command[index_list] = torch.tensor(value_list, device=self.device)

        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor(
                    [[-float("inf"), float("inf")]], device=self.device
                ).repeat(self.num_envs, self.action_dim, 1)
                index_list, _, value_list = string_utils.resolve_matching_names_values(
                    self.cfg.clip, self._joint_names
                )
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(
                    f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict."
                )

        self._action_space = spaces.Discrete(2)

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def env_ids(self) -> torch.Tensor:
        return self._env_ids

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    """
    Operations.
    """

    def process_actions(
        self, actions: torch.Tensor, env_ids: Sequence[int] | None = None
    ):
        if env_ids is None:
            self._env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            if not isinstance(env_ids, torch.Tensor):
                self._env_ids = torch.tensor(
                    self.env_ids, device=self.device, dtype=torch.int32
                )
            else:
                self._env_ids = env_ids
        assert actions.shape[0] == self._env_ids.shape[0], (
            f"Expected actions shape[0] to be {self.num_envs}, but got {actions.shape[0]}"
        )

        self._raw_actions[self.env_ids] = actions

        if actions.dtype == torch.bool:
            binary_mask = actions == 1
        else:
            binary_mask = actions >= 1

        self._processed_actions = torch.where(
            binary_mask, self._close_command, self._open_command
        )
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[self.env_ids, :, 0],
                max=self._clip[self.env_ids, :, 1],
            )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0


class BinaryJointPositionAction(BinaryJointAction):
    """Binary joint action that sets the binary action into joint position targets."""

    cfg: actions_cfg.BinaryJointPositionActionCfg
    """The configuration of the action term."""

    def apply_actions(self):
        assert self.processed_actions.shape[0] == len(self.env_ids), (
            f"Expected processed actions shape[0] to be {self.num_envs}, but got {self.processed_actions.shape[0]}"
        )
        self._asset.set_joint_position_target(
            self.processed_actions, joint_ids=self._joint_ids, env_ids=self.env_ids
        )


class DifferentialInverseKinematicsAction(ActionTerm):
    r"""Inverse Kinematics action term.

    This action term performs pre-processing of the raw actions using scaling transformation.

    .. math::
        \text{action} = \text{scaling} \times \text{input action}
        \text{joint position} = J^{-} \times \text{action}

    where :math:`\text{scaling}` is the scaling applied to the input action, and :math:`\text{input action}`
    is the input action from the user, :math:`J` is the Jacobian over the articulation's actuated joints,
    and \text{joint position} is the desired joint position command for the articulation's joints.
    """

    cfg: actions_cfg.DifferentialInverseKinematicsActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor
    """The scaling factor applied to the input action. Shape is (1, action_dim)."""
    _clip: torch.Tensor
    """The clip applied to the input action."""
    _action_space: spaces.Box
    """The action space of the action term."""

    def __init__(
        self, cfg: actions_cfg.DifferentialInverseKinematicsActionCfg, env: IsaacRLEnv
    ):
        super().__init__(cfg, env)

        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names
        )
        self._num_joints = len(self._joint_ids)

        body_ids, body_names = self._asset.find_bodies(self.cfg.body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected one match for the body name: {self.cfg.body_name}. Found {len(body_ids)}: {body_names}."
            )

        self._body_idx = body_ids[0]
        self._body_name = body_names[0]

        if self._asset.is_fixed_base:
            self._jacobi_body_idx = self._body_idx - 1
            self._jacobi_joint_ids = self._joint_ids
        else:
            self._jacobi_body_idx = self._body_idx
            self._jacobi_joint_ids = [i + 6 for i in self._joint_ids]

        omni.log.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )
        omni.log.info(
            f"Resolved body name for the action term {self.__class__.__name__}: {self._body_name} [{self._body_idx}]"
        )

        self._ik_controller = DifferentialIKController(
            cfg=self.cfg.controller, num_envs=self.num_envs, device=self.device
        )

        self._raw_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )
        self._processed_actions = torch.zeros_like(self.raw_actions)

        self._scale = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        self._scale[:] = torch.tensor(self.cfg.scale, device=self.device)

        if self.cfg.body_offset is not None:
            self._offset_pos = torch.tensor(
                self.cfg.body_offset.pos, device=self.device
            ).repeat(self.num_envs, 1)
            self._offset_rot = torch.tensor(
                self.cfg.body_offset.rot, device=self.device
            ).repeat(self.num_envs, 1)
        else:
            self._offset_pos, self._offset_rot = None, None

        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor(
                    [[-float("inf"), float("inf")]], device=self.device
                ).repeat(self.num_envs, self.action_dim, 1)
                index_list, _, value_list = string_utils.resolve_matching_names_values(
                    self.cfg.clip, self._joint_names
                )
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(
                    f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict."
                )

        assert self.cfg.action_space.shape[0] == 2, (
            "Expected action space to be of shape (2, action_dim)."
        )
        assert self.cfg.action_space.shape[1] == self.action_dim, (
            f"Expected action space to be of shape (2, {self.action_dim})."
        )

        if self.cfg.clip is not None:
            self.cfg.action_space = torch.clamp(
                self.cfg.action_space, min=self.cfg.clip[:, 0], max=self.cfg.clip[:, 1]
            )
        self._action_space = spaces.Box(
            low=self.cfg.action_space[0].cpu().numpy(),
            high=self.cfg.action_space[1].cpu().numpy(),
        )

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._ik_controller.action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def jacobian_w(self) -> torch.Tensor:
        return self._asset.root_physx_view.get_jacobians()[
            :, self._jacobi_body_idx, :, self._jacobi_joint_ids
        ]

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    @property
    def env_ids(self) -> torch.Tensor:
        return self._env_ids

    @property
    def jacobian_b(self) -> torch.Tensor:
        jacobian = self.jacobian_w
        base_rot = self._asset.data.root_quat_w
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot))
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
        return jacobian

    """
    Operations.
    """

    def process_actions(
        self, actions: torch.Tensor, env_ids: Sequence[int] | None = None
    ):
        if env_ids is None:
            self._env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            if not isinstance(env_ids, torch.Tensor):
                self._env_ids = torch.tensor(
                    env_ids, device=self.device, dtype=torch.int32
                )
            else:
                self._env_ids = env_ids
        assert actions.shape[0] == len(env_ids), (
            f"Expected actions shape[0] to be {self.num_envs}, but got {actions.shape[0]}"
        )

        self._raw_actions[self.env_ids] = actions

        self._processed_actions = self._raw_actions[self.env_ids]

        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )

        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()

        self._ik_controller.set_command(
            self._processed_actions, ee_pos_curr, ee_quat_curr
        )

    def apply_actions(self):
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        joint_pos = self._asset.data.joint_pos[
            self.env_ids.unsqueeze(1), self._joint_ids
        ]

        if ee_quat_curr.norm() != 0:
            jacobian = self._compute_frame_jacobian()[self.env_ids]
            joint_pos_des = self._ik_controller.compute(
                ee_pos_curr, ee_quat_curr, jacobian, joint_pos
            )
        else:
            joint_pos_des = joint_pos.clone()

        self._proccessed_actions = joint_pos_des
        self._asset.set_joint_position_target(
            joint_pos_des, self._joint_ids, self.env_ids
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0

    """
    Helper functions.
    """

    def _compute_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the pose of the target frame in the root frame.

        Returns:
            A tuple of the body's position and orientation in the root frame.
        """

        ee_pos_w = self._asset.data.body_pos_w[self.env_ids, self._body_idx]
        ee_quat_w = self._asset.data.body_quat_w[self.env_ids, self._body_idx]
        root_pos_w = self._asset.data.root_pos_w[self.env_ids]
        root_quat_w = self._asset.data.root_quat_w[self.env_ids]

        ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(
            root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
        )

        if self.cfg.body_offset is not None:
            ee_pose_b, ee_quat_b = math_utils.combine_frame_transforms(
                ee_pose_b,
                ee_quat_b,
                self._offset_pos[self.env_ids],
                self._offset_rot[self.env_ids],
            )

        return ee_pose_b, ee_quat_b

    def _compute_frame_jacobian(self):
        """Computes the geometric Jacobian of the target frame in the root frame.

        This function accounts for the target frame offset and applies the necessary transformations to obtain
        the right Jacobian from the parent body Jacobian.
        """

        jacobian = self.jacobian_b

        if self.cfg.body_offset is not None:
            jacobian[:, 0:3, :] += torch.bmm(
                -math_utils.skew_symmetric_matrix(self._offset_pos), jacobian[:, 3:, :]
            )

            jacobian[:, 3:, :] = torch.bmm(
                math_utils.matrix_from_quat(self._offset_rot), jacobian[:, 3:, :]
            )

        return jacobian


class HolomonicAction(ActionTerm):
    """Applies a differential controller to compute left/right wheel speeds from (v, ω)."""

    cfg: actions_cfg.HolomonicActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _clip: torch.Tensor
    """The clip applied to the input action."""
    _action_space: spaces.Box
    """The action space of the action term."""

    def __init__(self, cfg: actions_cfg.HolomonicActionCfg, env: IsaacRLEnv):
        super().__init__(cfg, env)

        self._raw_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )
        self._processed_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )

        self._scale = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        self._scale[:] = torch.tensor(self.cfg.scale, device=self.device)

        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor(
                    [[-float("inf"), float("inf")]], device=self.device
                ).repeat(self.num_envs, self.action_dim, 1)
                index_list, _, value_list = string_utils.resolve_matching_names_values(
                    self.cfg.clip, self._joint_names
                )
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(
                    f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict."
                )

        assert self.cfg.action_space.shape[0] == 2, (
            "Expected action space to be of shape (2, action_dim)."
        )
        assert self.cfg.action_space.shape[1] == self.action_dim, (
            f"Expected action space to be of shape (2, {self.action_dim})."
        )

        if self.cfg.clip is not None:
            self.cfg.action_space = torch.clamp(
                self.cfg.action_space, min=self.cfg.clip[:, 0], max=self.cfg.clip[:, 1]
            )
        self._action_space = spaces.Box(
            low=self.cfg.action_space[0].cpu().numpy(),
            high=self.cfg.action_space[1].cpu().numpy(),
        )

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def env_ids(self) -> torch.Tensor:
        return self._env_ids

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    def process_actions(self, actions, env_ids=None):
        """Convert [v, ω] commands to left/right wheel speeds."""
        if env_ids is None:
            self._env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            if not isinstance(env_ids, torch.Tensor):
                self._env_ids = torch.tensor(
                    env_ids, device=self.device, dtype=torch.int32
                )
            else:
                self._env_ids = env_ids
        assert actions.shape[0] == len(env_ids), (
            f"Expected actions shape[0] to be {self.num_envs}, but got {actions.shape[0]}"
        )

        self._raw_actions[self.env_ids] = actions

        self._processed_actions = self._raw_actions[self.env_ids]

        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )

    def apply_actions(self):
        """Send wheel speeds to the articulation."""

        root_velocity = self._asset.data.root_com_vel_w[self._env_ids].clone()

        root_velocity[:, 0] = self._processed_actions[:, 0]
        root_velocity[:, 1] = self._processed_actions[:, 1]
        root_velocity[:, 5] = self._processed_actions[:, 2]

        self._processed_actions = root_velocity

        self._asset.write_root_velocity_to_sim(
            root_velocity=root_velocity, env_ids=self._env_ids
        )


class HolomonicUVAction(ActionTerm):
    """Applies a differential controller to compute left/right wheel speeds from (v, ω)."""

    cfg: actions_cfg.HolomonicUVActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _clip: torch.Tensor
    """The clip applied to the input action."""
    _action_space: spaces.Box
    """The action space of the action term."""

    def __init__(self, cfg: actions_cfg.HolomonicUVActionCfg, env: IsaacRLEnv):
        super().__init__(cfg, env)

        self._raw_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )
        self._processed_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )

        self._scale = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        self._scale[:] = torch.tensor(self.cfg.scale, device=self.device)

        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor(
                    [[-float("inf"), float("inf")]], device=self.device
                ).repeat(self.num_envs, self.action_dim, 1)
                index_list, _, value_list = string_utils.resolve_matching_names_values(
                    self.cfg.clip, self._joint_names
                )
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(
                    f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict."
                )

        assert self.cfg.action_space.shape[0] == 2, (
            "Expected action space to be of shape (2, action_dim)."
        )
        assert self.cfg.action_space.shape[1] == self.action_dim, (
            f"Expected action space to be of shape (2, {self.action_dim})."
        )

        if self.cfg.clip is not None:
            self.cfg.action_space = torch.clamp(
                self.cfg.action_space, min=self.cfg.clip[:, 0], max=self.cfg.clip[:, 1]
            )
        self._action_space = spaces.Box(
            low=self.cfg.action_space[0].cpu().numpy(),
            high=self.cfg.action_space[1].cpu().numpy(),
        )

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def env_ids(self) -> torch.Tensor:
        return self._env_ids

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    def process_actions(self, actions, env_ids=None):
        """Convert [v, ω] commands to left/right wheel speeds."""
        if env_ids is None:
            self._env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            if not isinstance(env_ids, torch.Tensor):
                self._env_ids = torch.tensor(
                    env_ids, device=self.device, dtype=torch.int32
                )
            else:
                self._env_ids = env_ids
        assert actions.shape[0] == len(env_ids), (
            f"Expected actions shape[0] to be {self.num_envs}, but got {actions.shape[0]}"
        )

        self._raw_actions[self.env_ids] = actions

        self._processed_actions = self._raw_actions[self.env_ids]

        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )

    def quat_to_yaw(self, quat: torch.Tensor) -> torch.Tensor:
        """Convert quaternion to yaw angle.

        Args:
            quat (torch.Tensor): Quaternion tensor of shape (num_envs, 4) in (x, y, z, w) format.

        Returns:
            torch.Tensor: Yaw angles of shape (num_envs,).
        """

        x = quat[:, 0]
        y = quat[:, 1]
        z = quat[:, 2]
        w = quat[:, 3]

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)

        return yaw

    def apply_actions(self):
        """Applies linear and angular velocity (v, ω) to the robot base."""

        root_pose = self._asset.root_physx_view.get_root_transforms()[
            self._env_ids, 3:7
        ]
        yaw = self.quat_to_yaw(root_pose)

        v = self._processed_actions[:, 0]
        w = self._processed_actions[:, 1]

        vx = v * torch.cos(yaw)
        vy = v * torch.sin(yaw)

        root_velocity = self._asset.data.root_com_vel_w[self._env_ids].clone()

        root_velocity[:, 0] = vx
        root_velocity[:, 1] = vy
        root_velocity[:, 5] = w

        self._processed_actions = root_velocity

        self._asset.write_root_velocity_to_sim(
            root_velocity=root_velocity, env_ids=self._env_ids
        )


class DifferentialAction(ActionTerm):
    """Applies a differential controller to compute left/right wheel speeds from (v, ω)."""

    cfg: actions_cfg.DifferentialActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _clip: torch.Tensor
    """The clip applied to the input action."""
    _action_space: spaces.Box
    """The action space of the action term."""

    def __init__(self, cfg: actions_cfg.DifferentialActionCfg, env: IsaacRLEnv):
        super().__init__(cfg, env)

        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names
        )

        self._num_joints = len(self._joint_ids)
        self._raw_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )
        self._processed_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )

        self._scale = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        self._scale[:] = torch.tensor(self.cfg.scale, device=self.device)

        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor(
                    [[-float("inf"), float("inf")]], device=self.device
                ).repeat(self.num_envs, self.action_dim, 1)
                index_list, _, value_list = string_utils.resolve_matching_names_values(
                    self.cfg.clip, self._joint_names
                )
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(
                    f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict."
                )

        assert self.cfg.action_space.shape[0] == 2, (
            "Expected action space to be of shape (2, action_dim)."
        )
        assert self.cfg.action_space.shape[1] == self.action_dim, (
            f"Expected action space to be of shape (2, {self.action_dim})."
        )

        if self.cfg.clip is not None:
            self.cfg.action_space = torch.clamp(
                self.cfg.action_space, min=self.cfg.clip[:, 0], max=self.cfg.clip[:, 1]
            )
        self._action_space = spaces.Box(
            low=self.cfg.action_space[0].cpu().numpy(),
            high=self.cfg.action_space[1].cpu().numpy(),
        )

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def env_ids(self) -> torch.Tensor:
        return self._env_ids

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    def process_actions(self, actions, env_ids=None):
        """Convert [v, ω] commands to left/right wheel speeds."""
        if env_ids is None:
            self._env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            if not isinstance(env_ids, torch.Tensor):
                self._env_ids = torch.tensor(
                    env_ids, device=self.device, dtype=torch.int32
                )
            else:
                self._env_ids = env_ids
        assert actions.shape[0] == len(env_ids), (
            f"Expected actions shape[0] to be {self.num_envs}, but got {actions.shape[0]}"
        )

        self._raw_actions[self.env_ids] = actions

        self._processed_actions = self._raw_actions[self.env_ids]

        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )

    def apply_actions(self):
        """
        [v, ω]
        """

        v = self._processed_actions[:, 0]
        w = self._processed_actions[:, 1]

        v_left = v - (w * self.cfg.wheel_base / 2.0)
        v_right = v + (w * self.cfg.wheel_base / 2.0)

        omega_left = v_left / self.cfg.wheel_radius
        omega_right = v_right / self.cfg.wheel_radius

        joint_vel_target = torch.stack([omega_left, omega_right], dim=1)

        self._processed_actions = joint_vel_target

        self._asset.set_joint_velocity_target(
            target=joint_vel_target,
            joint_ids=self._joint_ids,
            env_ids=self._env_ids,
        )


class AckermannSteeringAction(ActionTerm):
    """
    Ackermann

    2
    - actions[:, 0]:  [-1, 1]
        -
        -
        - 0
    - actions[:, 1]:  [-1, 1]
        -
        -
        - 0


    -
    -  Ackermann
    """

    cfg: actions_cfg.AckermannSteeringActionCfg

    _asset: Articulation

    _clip: torch.Tensor
    """The clip applied to the input action."""

    _action_space: spaces.Box
    """The action space of the action term."""

    def __init__(self, cfg: actions_cfg.AckermannSteeringActionCfg, env: IsaacRLEnv):
        super().__init__(cfg, env)

        self._wheel_joint_ids, self._wheel_joint_names = self._asset.find_joints(
            self.cfg.wheel_joint_names
        )

        self._steering_joint_ids, self._steering_joint_names = self._asset.find_joints(
            self.cfg.steering_joint_names
        )

        assert len(self._wheel_joint_ids) == 4, (
            f"Expected 4 wheel joints, got {len(self._wheel_joint_ids)}"
        )
        assert len(self._steering_joint_ids) == 2, (
            f"Expected 2 steering joints, got {len(self._steering_joint_ids)}"
        )

        self._raw_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )
        self._processed_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )

        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor(
                    [[-float("inf"), float("inf")]], device=self.device
                ).repeat(self.num_envs, self.action_dim, 1)
                index_list, _, value_list = string_utils.resolve_matching_names_values(
                    self.cfg.clip, self._joint_names
                )
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(
                    f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict."
                )

        assert self.cfg.action_space.shape[0] == 2, (
            "Expected action space to be of shape (2, action_dim)."
        )
        assert self.cfg.action_space.shape[1] == self.action_dim, (
            f"Expected action space to be of shape (2, {self.action_dim})."
        )

        if self.cfg.clip is not None:
            self.cfg.action_space = torch.clamp(
                self.cfg.action_space, min=self.cfg.clip[:, 0], max=self.cfg.clip[:, 1]
            )
        self._action_space = spaces.Box(
            low=self.cfg.action_space[0].cpu().numpy(),
            high=self.cfg.action_space[1].cpu().numpy(),
        )

    @property
    def action_dim(self) -> int:
        """Ackermann  2"""
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def env_ids(self) -> torch.Tensor:
        return self._env_ids

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    def process_actions(self, actions, env_ids=None):
        """


        Args:
            actions: shape (num_envs, 2)
                - actions[:, 0]:  [-1, 1]
                - actions[:, 1]:  [-1, 1]
            env_ids: IDNone
        """
        if env_ids is None:
            self._env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            if not isinstance(env_ids, torch.Tensor):
                self._env_ids = torch.tensor(
                    env_ids, device=self.device, dtype=torch.int32
                )
            else:
                self._env_ids = env_ids

        assert actions.shape[0] == len(self._env_ids), (
            f"Expected actions shape[0] ({actions.shape[0]}) to match env_ids length ({len(self._env_ids)})"
        )
        assert actions.shape[1] == self.action_dim, (
            f"Expected actions shape[1] to be {self.action_dim} (throttle, steering), but got {actions.shape[1]}"
        )

        self._raw_actions[self._env_ids] = actions

        self._processed_actions[self._env_ids] = self._raw_actions[self._env_ids]

        if self.cfg.clip is not None:
            self._processed_actions[self._env_ids] = torch.clamp(
                self._processed_actions[self._env_ids],
                min=self._clip[self._env_ids, :, 0],
                max=self._clip[self._env_ids, :, 1],
            )

    def apply_actions(self):
        """
         Ackermann


        - throttle_cmd:  [-1, 1]
        - steering_cmd:  [-1, 1]
        """

        throttle_cmd = self._processed_actions[self._env_ids, 0]
        steering_cmd = self._processed_actions[self._env_ids, 1]

        device = self._processed_actions.device
        dtype = self._processed_actions.dtype

        linear_speed = throttle_cmd * self.cfg.max_speed

        if self.cfg.wheel_radius <= 0.0:
            raise ValueError(
                f"wheel_radius must be positive, got: {self.cfg.wheel_radius}"
            )

        wheel_angular_speed = linear_speed / self.cfg.wheel_radius

        wheel_velocities = torch.stack(
            [
                wheel_angular_speed,
                wheel_angular_speed,
                wheel_angular_speed,
                wheel_angular_speed,
            ],
            dim=1,
        )

        target_steering_angle = steering_cmd * self.cfg.max_steering_angle

        if self.cfg.use_ackermann_geometry:
            if self.cfg.wheel_base <= 0.0 or self.cfg.track_width <= 0.0:
                raise ValueError("wheel_base and track_width must be positive")

            wheel_base = torch.tensor(self.cfg.wheel_base, device=device, dtype=dtype)
            track_width = torch.tensor(self.cfg.track_width, device=device, dtype=dtype)
            half_track = track_width / 2.0
            eps = 1e-6

            tan_steering = torch.tan(target_steering_angle)
            is_straight = torch.abs(tan_steering) < eps

            turn_radius = torch.where(
                is_straight,
                torch.sign(tan_steering) * 1e9,
                wheel_base / (tan_steering + eps * torch.sign(tan_steering)),
            )

            inner_radius = torch.clamp(torch.abs(turn_radius) - half_track, min=eps)
            outer_radius = torch.abs(turn_radius) + half_track
            turn_sign = torch.sign(turn_radius)

            angle_inner = torch.atan2(wheel_base, inner_radius) * turn_sign
            angle_outer = torch.atan2(wheel_base, outer_radius) * turn_sign

            is_left_turn = turn_radius > 0
            final_left = torch.where(is_left_turn, angle_inner, angle_outer)
            final_right = torch.where(is_left_turn, angle_outer, angle_inner)

            steering_angles = torch.stack([final_left, final_right], dim=1)
        else:
            steering_angles = torch.stack(
                [target_steering_angle, target_steering_angle], dim=1
            )

        self._processed_actions = torch.cat([wheel_velocities, steering_angles], dim=1)

        self._asset.set_joint_velocity_target(
            target=wheel_velocities,
            joint_ids=self._wheel_joint_ids,
            env_ids=self._env_ids,
        )

        self._asset.set_joint_position_target(
            target=steering_angles,
            joint_ids=self._steering_joint_ids,
            env_ids=self._env_ids,
        )
