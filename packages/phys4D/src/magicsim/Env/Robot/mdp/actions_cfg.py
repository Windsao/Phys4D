from dataclasses import MISSING

import torch

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from magicsim.Env.Robot.mdp.actions import (
    JointPositionToLimitsAction,
    BinaryJointPositionAction,
    DifferentialInverseKinematicsAction,
    HolomonicAction,
    HolomonicUVAction,
    DifferentialAction,
    AckermannSteeringAction,
)

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg


@configclass
class JointPositionToLimitsActionCfg(ActionTermCfg):
    """Configuration for the bounded joint position action term.

    See :class:`JointPositionToLimitsAction` for more details.
    """

    class_type: type[ActionTerm] = JointPositionToLimitsAction

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""

    scale: float | dict[str, float] = 1.0
    """Scale factor for the action (float or dict of regex expressions). Defaults to 1.0."""

    rescale_to_limits: bool = False
    """Whether to rescale the action to the joint limits. Defaults to True.

    If True, the input actions are rescaled to the joint limits, i.e., the action value in
    the range [-1, 1] corresponds to the joint lower and upper limits respectively.

    Note:
        This operation is performed after applying the scale factor.
    """


@configclass
class BinaryJointActionCfg(ActionTermCfg):
    """Configuration for the base binary joint action term.

    See :class:`BinaryJointAction` for more details.
    """

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""
    open_command_expr: dict[str, float] = MISSING
    """The joint command to move to *open* configuration."""
    close_command_expr: dict[str, float] = MISSING
    """The joint command to move to *close* configuration."""


@configclass
class BinaryJointPositionActionCfg(BinaryJointActionCfg):
    """Configuration for the binary joint position action term.

    See :class:`BinaryJointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = BinaryJointPositionAction


@configclass
class DifferentialInverseKinematicsActionCfg(ActionTermCfg):
    """Configuration for inverse differential kinematics action term.

    See :class:`DifferentialInverseKinematicsAction` for more details.
    """

    @configclass
    class OffsetCfg:
        """The offset pose from parent frame to child frame.

        On many robots, end-effector frames are fictitious frames that do not have a corresponding
        rigid body. In such cases, it is easier to define this transform w.r.t. their parent rigid body.
        For instance, for the Franka Emika arm, the end-effector is defined at an offset to the the
        "panda_hand" frame.
        """

        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Translation w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0)."""
        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        """Quaternion rotation ``(w, x, y, z)`` w.r.t. the parent frame. Defaults to (1.0, 0.0, 0.0, 0.0)."""

    class_type: type[ActionTerm] = DifferentialInverseKinematicsAction

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""
    body_name: str = MISSING
    """Name of the body or frame for which IK is performed."""
    body_offset: OffsetCfg | None = None
    """Offset of target frame w.r.t. to the body frame. Defaults to None, in which case no offset is applied."""
    scale: float | tuple[float, ...] = 1.0
    """Scale factor for the action. Defaults to 1.0."""
    controller: DifferentialIKControllerCfg = MISSING
    """The configuration for the differential IK controller."""
    action_space: torch.Tensor = MISSING
    """The action space for the action term. Should be a tuple of (low, high) or a list of such tuples.
    If a list is provided, it should have the same length as the number of joints.
    """


@configclass
class HolomonicActionCfg(ActionTermCfg):
    """Configuration for differential drive control action term.

    This config works with the DifferentialDriveAction class,
    which wraps Isaac Sim’s DifferentialController to convert
    (linear_vel, angular_vel) commands into wheel joint velocities.
    """

    class_type: type[ActionTerm] = HolomonicAction

    """Radius of each wheel (in meters)."""

    """Distance between left and right wheels (in meters)."""

    """Maximum forward/backward speed (m/s)."""

    """Maximum rotational speed (rad/s)."""

    """List of right wheel joint names or regex expressions."""

    clip: dict[str, tuple[float, float]] | None = None
    """Optional action clipping range for velocity commands."""

    scale: float | tuple[float, ...] = 1.0
    """Scale factor for the action. Defaults to 1.0."""
    rescale_to_limits: bool = False
    action_space: torch.Tensor | None = None
    """The action space for the action term. Should be a tuple of (low, high) or a list of such tuples.
    If a list is provided, it should have the same length as the number of joints.
    """


@configclass
class HolomonicUVActionCfg(ActionTermCfg):
    """Configuration for differential drive control action term.

    This config works with the DifferentialDriveAction class,
    which wraps Isaac Sim’s DifferentialController to convert
    (linear_vel, angular_vel) commands into wheel joint velocities.
    """

    class_type: type[ActionTerm] = HolomonicUVAction

    clip: dict[str, tuple[float, float]] | None = None
    """Optional action clipping range for velocity commands."""

    scale: float | tuple[float, ...] = 1.0
    """Scale factor for the action. Defaults to 1.0."""
    rescale_to_limits: bool = False
    action_space: torch.Tensor | None = None
    """The action space for the action term. Should be a tuple of (low, high) or a list of such tuples.
    If a list is provided, it should have the same length as the number of joints.
    """
    wheel_radius = 0.5
    wheel_base = 0.5


@configclass
class DifferentialActionCfg(ActionTermCfg):
    """Configuration for differential drive control action term.

    This config works with the DifferentialDriveAction class,
    which wraps Isaac Sim’s DifferentialController to convert
    (linear_vel, angular_vel) commands into wheel joint velocities.
    """

    class_type: type[ActionTerm] = DifferentialAction

    clip: dict[str, tuple[float, float]] | None = None
    """Optional action clipping range for velocity commands."""

    scale: float | tuple[float, ...] = 1.0
    """Scale factor for the action. Defaults to 1.0."""
    rescale_to_limits: bool = False
    action_space: torch.Tensor | None = None
    """The action space for the action term. Should be a tuple of (low, high) or a list of such tuples.
    If a list is provided, it should have the same length as the number of joints.
    """
    wheel_radius = 0.5
    wheel_base = 0.5
    joint_names: list[str] = MISSING


@configclass
class AckermannSteeringActionCfg(ActionTermCfg):
    """Ackermann"""

    class_type: type[ActionTerm] = AckermannSteeringAction

    wheel_joint_names: list[str] = MISSING
    """"""

    steering_joint_names: list[str] = MISSING
    """"""

    wheel_radius: float = 0.25
    """ (m)"""

    wheel_base: float = 1.65
    """ (m)"""

    track_width: float = 1.25
    """ (m)"""

    max_speed: float = 10.0
    """ (m/s)"""

    max_steering_angle: float = 0.5
    """ (rad) 28.6 """

    action_space: torch.Tensor = MISSING
    """ [[throttle_min, steering_min], [throttle_max, steering_max]]"""

    use_ackermann_geometry: bool = True
    """"""

    clip: dict[str, tuple[float, float]] | None = None
    """ None """
