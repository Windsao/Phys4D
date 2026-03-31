







from __future__ import annotations

import torch
from collections.abc import Sequence
from prettytable import PrettyTable
from typing import TYPE_CHECKING

import omni.log
import omni.physics.tensors.impl.api as physx
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.version import get_version
from pxr import PhysxSchema, UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.actuators import ActuatorBase, ActuatorBaseCfg, ImplicitActuator
from isaaclab.utils.types import ArticulationActions

from ..asset_base import AssetBase
from .articulation_data import ArticulationData

if TYPE_CHECKING:
    from .articulation_cfg import ArticulationCfg


class Articulation(AssetBase):
    """An articulation asset class.

    An articulation is a collection of rigid bodies connected by joints. The joints can be either
    fixed or actuated. The joints can be of different types, such as revolute, prismatic, D-6, etc.
    However, the articulation class has currently been tested with revolute and prismatic joints.
    The class supports both floating-base and fixed-base articulations. The type of articulation
    is determined based on the root joint of the articulation. If the root joint is fixed, then
    the articulation is considered a fixed-base system. Otherwise, it is considered a floating-base
    system. This can be checked using the :attr:`Articulation.is_fixed_base` attribute.

    For an asset to be considered an articulation, the root prim of the asset must have the
    `USD ArticulationRootAPI`_. This API is used to define the sub-tree of the articulation using
    the reduced coordinate formulation. On playing the simulation, the physics engine parses the
    articulation root prim and creates the corresponding articulation in the physics engine. The
    articulation root prim can be specified using the :attr:`AssetBaseCfg.prim_path` attribute.

    The articulation class also provides the functionality to augment the simulation of an articulated
    system with custom actuator models. These models can either be explicit or implicit, as detailed in
    the :mod:`isaaclab.actuators` module. The actuator models are specified using the
    :attr:`ArticulationCfg.actuators` attribute. These are then parsed and used to initialize the
    corresponding actuator models, when the simulation is played.

    During the simulation step, the articulation class first applies the actuator models to compute
    the joint commands based on the user-specified targets. These joint commands are then applied
    into the simulation. The joint commands can be either position, velocity, or effort commands.
    As an example, the following snippet shows how this can be used for position commands:

    .. code-block:: python

        # an example instance of the articulation class
        my_articulation = Articulation(cfg)

        # set joint position targets
        my_articulation.set_joint_position_target(position)
        # propagate the actuator models and apply the computed commands into the simulation
        my_articulation.write_data_to_sim()

        # step the simulation using the simulation context
        sim_context.step()

        # update the articulation state, where dt is the simulation time step
        my_articulation.update(dt)

    .. _`USD ArticulationRootAPI`: https://openusd.org/dev/api/class_usd_physics_articulation_root_a_p_i.html

    """

    cfg: ArticulationCfg
    """Configuration instance for the articulations."""

    actuators: dict[str, ActuatorBase]
    """Dictionary of actuator instances for the articulation.

    The keys are the actuator names and the values are the actuator instances. The actuator instances
    are initialized based on the actuator configurations specified in the :attr:`ArticulationCfg.actuators`
    attribute. They are used to compute the joint commands during the :meth:`write_data_to_sim` function.
    """

    def __init__(self, cfg: ArticulationCfg):
        """Initialize the articulation.

        Args:
            cfg: A configuration instance.
        """
        super().__init__(cfg)

    """
    Properties
    """

    @property
    def data(self) -> ArticulationData:
        return self._data

    @property
    def num_instances(self) -> int:
        return self.root_physx_view.count

    @property
    def is_fixed_base(self) -> bool:
        """Whether the articulation is a fixed-base or floating-base system."""
        return self.root_physx_view.shared_metatype.fixed_base

    @property
    def num_joints(self) -> int:
        """Number of joints in articulation."""
        return self.root_physx_view.shared_metatype.dof_count

    @property
    def num_fixed_tendons(self) -> int:
        """Number of fixed tendons in articulation."""
        return self.root_physx_view.max_fixed_tendons

    @property
    def num_spatial_tendons(self) -> int:
        """Number of spatial tendons in articulation."""
        return self.root_physx_view.max_spatial_tendons

    @property
    def num_bodies(self) -> int:
        """Number of bodies in articulation."""
        return self.root_physx_view.shared_metatype.link_count

    @property
    def joint_names(self) -> list[str]:
        """Ordered names of joints in articulation."""
        return self.root_physx_view.shared_metatype.dof_names

    @property
    def fixed_tendon_names(self) -> list[str]:
        """Ordered names of fixed tendons in articulation."""
        return self._fixed_tendon_names

    @property
    def spatial_tendon_names(self) -> list[str]:
        """Ordered names of spatial tendons in articulation."""
        return self._spatial_tendon_names

    @property
    def body_names(self) -> list[str]:
        """Ordered names of bodies in articulation."""
        return self.root_physx_view.shared_metatype.link_names

    @property
    def root_physx_view(self) -> physx.ArticulationView:
        """Articulation view for the asset (PhysX).

        Note:
            Use this view with caution. It requires handling of tensors in a specific way.
        """
        return self._root_physx_view

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None):

        if env_ids is None:
            env_ids = slice(None)

        for actuator in self.actuators.values():
            actuator.reset(env_ids)

        self._external_force_b[env_ids] = 0.0
        self._external_torque_b[env_ids] = 0.0
        self._external_wrench_positions_b[env_ids] = 0.0

    def write_data_to_sim(self):
        """Write external wrenches and joint commands to the simulation.

        If any explicit actuators are present, then the actuator models are used to compute the
        joint commands. Otherwise, the joint commands are directly set into the simulation.

        Note:
            We write external wrench to the simulation here since this function is called before the simulation step.
            This ensures that the external wrench is applied at every simulation step.
        """

        if self.has_external_wrench:
            if self.uses_external_wrench_positions:
                self.root_physx_view.apply_forces_and_torques_at_position(
                    force_data=self._external_force_b.view(-1, 3),
                    torque_data=self._external_torque_b.view(-1, 3),
                    position_data=self._external_wrench_positions_b.view(-1, 3),
                    indices=self._ALL_INDICES,
                    is_global=self._use_global_wrench_frame,
                )
            else:
                self.root_physx_view.apply_forces_and_torques_at_position(
                    force_data=self._external_force_b.view(-1, 3),
                    torque_data=self._external_torque_b.view(-1, 3),
                    position_data=None,
                    indices=self._ALL_INDICES,
                    is_global=self._use_global_wrench_frame,
                )


        self._apply_actuator_model()

        self.root_physx_view.set_dof_actuation_forces(self._joint_effort_target_sim, self._ALL_INDICES)

        if self._has_implicit_actuators:
            self.root_physx_view.set_dof_position_targets(self._joint_pos_target_sim, self._ALL_INDICES)
            self.root_physx_view.set_dof_velocity_targets(self._joint_vel_target_sim, self._ALL_INDICES)

    def update(self, dt: float):
        self._data.update(dt)

    """
    Operations - Finders.
    """

    def find_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        """Find bodies in the articulation based on the name keys.

        Please check the :meth:`isaaclab.utils.string_utils.resolve_matching_names` function for more
        information on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the body names.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the body indices and names.
        """
        return string_utils.resolve_matching_names(name_keys, self.body_names, preserve_order)

    def find_joints(
        self, name_keys: str | Sequence[str], joint_subset: list[str] | None = None, preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        """Find joints in the articulation based on the name keys.

        Please see the :func:`isaaclab.utils.string.resolve_matching_names` function for more information
        on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the joint names.
            joint_subset: A subset of joints to search for. Defaults to None, which means all joints
                in the articulation are searched.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the joint indices and names.
        """
        if joint_subset is None:
            joint_subset = self.joint_names

        return string_utils.resolve_matching_names(name_keys, joint_subset, preserve_order)

    def find_fixed_tendons(
        self, name_keys: str | Sequence[str], tendon_subsets: list[str] | None = None, preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        """Find fixed tendons in the articulation based on the name keys.

        Please see the :func:`isaaclab.utils.string.resolve_matching_names` function for more information
        on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the joint
                names with fixed tendons.
            tendon_subsets: A subset of joints with fixed tendons to search for. Defaults to None, which means
                all joints in the articulation are searched.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the tendon indices and names.
        """
        if tendon_subsets is None:

            tendon_subsets = self.fixed_tendon_names

        return string_utils.resolve_matching_names(name_keys, tendon_subsets, preserve_order)

    def find_spatial_tendons(
        self, name_keys: str | Sequence[str], tendon_subsets: list[str] | None = None, preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        """Find spatial tendons in the articulation based on the name keys.

        Please see the :func:`isaaclab.utils.string.resolve_matching_names` function for more information
        on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the tendon names.
            tendon_subsets: A subset of tendons to search for. Defaults to None, which means all tendons
                in the articulation are searched.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the tendon indices and names.
        """
        if tendon_subsets is None:
            tendon_subsets = self.spatial_tendon_names

        return string_utils.resolve_matching_names(name_keys, tendon_subsets, preserve_order)

    """
    Operations - State Writers.
    """

    def write_root_state_to_sim(self, root_state: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root state over selected environment indices into the simulation.

        The root state comprises of the cartesian position, quaternion orientation in (w, x, y, z), and linear
        and angular velocity. All the quantities are in the simulation frame.

        Args:
            root_state: Root state in simulation frame. Shape is (len(env_ids), 13).
            env_ids: Environment indices. If None, then all indices are used.
        """
        self.write_root_link_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        self.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)

    def write_root_com_state_to_sim(self, root_state: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root center of mass state over selected environment indices into the simulation.

        The root state comprises of the cartesian position, quaternion orientation in (w, x, y, z), and linear
        and angular velocity. All the quantities are in the simulation frame.

        Args:
            root_state: Root state in simulation frame. Shape is (len(env_ids), 13).
            env_ids: Environment indices. If None, then all indices are used.
        """
        self.write_root_com_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        self.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)

    def write_root_link_state_to_sim(self, root_state: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root link state over selected environment indices into the simulation.

        The root state comprises of the cartesian position, quaternion orientation in (w, x, y, z), and linear
        and angular velocity. All the quantities are in the simulation frame.

        Args:
            root_state: Root state in simulation frame. Shape is (len(env_ids), 13).
            env_ids: Environment indices. If None, then all indices are used.
        """
        self.write_root_link_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        self.write_root_link_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)

    def write_root_pose_to_sim(self, root_pose: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (w, x, y, z).

        Args:
            root_pose: Root poses in simulation frame. Shape is (len(env_ids), 7).
            env_ids: Environment indices. If None, then all indices are used.
        """
        self.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)

    def write_root_link_pose_to_sim(self, root_pose: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root link pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (w, x, y, z).

        Args:
            root_pose: Root poses in simulation frame. Shape is (len(env_ids), 7).
            env_ids: Environment indices. If None, then all indices are used.
        """

        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES



        self._data.root_link_pose_w[env_ids] = root_pose.clone()

        if self._data._root_link_state_w.data is not None:
            self._data.root_link_state_w[env_ids, :7] = self._data.root_link_pose_w[env_ids]
        if self._data._root_state_w.data is not None:
            self._data.root_state_w[env_ids, :7] = self._data.root_link_pose_w[env_ids]


        root_poses_xyzw = self._data.root_link_pose_w.clone()
        root_poses_xyzw[:, 3:] = math_utils.convert_quat(root_poses_xyzw[:, 3:], to="xyzw")


        self._data._body_link_pose_w.timestamp = -1.0
        self._data._body_com_pose_w.timestamp = -1.0
        self._data._body_state_w.timestamp = -1.0
        self._data._body_link_state_w.timestamp = -1.0
        self._data._body_com_state_w.timestamp = -1.0


        self.root_physx_view.set_root_transforms(root_poses_xyzw, indices=physx_env_ids)

    def write_root_com_pose_to_sim(self, root_pose: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root center of mass pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (w, x, y, z).
        The orientation is the orientation of the principle axes of inertia.

        Args:
            root_pose: Root center of mass poses in simulation frame. Shape is (len(env_ids), 7).
            env_ids: Environment indices. If None, then all indices are used.
        """

        if env_ids is None:
            local_env_ids = slice(env_ids)
        else:
            local_env_ids = env_ids


        self._data.root_com_pose_w[local_env_ids] = root_pose.clone()

        if self._data._root_com_state_w.data is not None:
            self._data.root_com_state_w[local_env_ids, :7] = self._data.root_com_pose_w[local_env_ids]


        com_pos_b = self.data.body_com_pos_b[local_env_ids, 0, :]
        com_quat_b = self.data.body_com_quat_b[local_env_ids, 0, :]

        root_link_pos, root_link_quat = math_utils.combine_frame_transforms(
            root_pose[..., :3],
            root_pose[..., 3:7],
            math_utils.quat_apply(math_utils.quat_inv(com_quat_b), -com_pos_b),
            math_utils.quat_inv(com_quat_b),
        )
        root_link_pose = torch.cat((root_link_pos, root_link_quat), dim=-1)


        self.write_root_link_pose_to_sim(root_pose=root_link_pose, env_ids=env_ids)

    def write_root_velocity_to_sim(self, root_velocity: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root center of mass velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        NOTE: This sets the velocity of the root's center of mass rather than the roots frame.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (len(env_ids), 6).
            env_ids: Environment indices. If None, then all indices are used.
        """
        self.write_root_com_velocity_to_sim(root_velocity=root_velocity, env_ids=env_ids)

    def write_root_com_velocity_to_sim(self, root_velocity: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root center of mass velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        NOTE: This sets the velocity of the root's center of mass rather than the roots frame.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (len(env_ids), 6).
            env_ids: Environment indices. If None, then all indices are used.
        """

        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES



        self._data.root_com_vel_w[env_ids] = root_velocity.clone()

        if self._data._root_com_state_w.data is not None:
            self._data.root_com_state_w[env_ids, 7:] = self._data.root_com_vel_w[env_ids]
        if self._data._root_state_w.data is not None:
            self._data.root_state_w[env_ids, 7:] = self._data.root_com_vel_w[env_ids]

        self._data.body_acc_w[env_ids] = 0.0


        self.root_physx_view.set_root_velocities(self._data.root_com_vel_w, indices=physx_env_ids)

    def write_root_link_velocity_to_sim(self, root_velocity: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root link velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        NOTE: This sets the velocity of the root's frame rather than the roots center of mass.

        Args:
            root_velocity: Root frame velocities in simulation world frame. Shape is (len(env_ids), 6).
            env_ids: Environment indices. If None, then all indices are used.
        """

        if env_ids is None:
            local_env_ids = slice(env_ids)
        else:
            local_env_ids = env_ids


        self._data.root_link_vel_w[local_env_ids] = root_velocity.clone()

        if self._data._root_link_state_w.data is not None:
            self._data.root_link_state_w[local_env_ids, 7:] = self._data.root_link_vel_w[local_env_ids]


        quat = self.data.root_link_quat_w[local_env_ids]
        com_pos_b = self.data.body_com_pos_b[local_env_ids, 0, :]

        root_com_velocity = root_velocity.clone()
        root_com_velocity[:, :3] += torch.linalg.cross(
            root_com_velocity[:, 3:], math_utils.quat_apply(quat, com_pos_b), dim=-1
        )


        self.write_root_com_velocity_to_sim(root_velocity=root_com_velocity, env_ids=env_ids)

    def write_joint_state_to_sim(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | slice | None = None,
    ):
        """Write joint positions and velocities to the simulation.

        Args:
            position: Joint positions. Shape is (len(env_ids), len(joint_ids)).
            velocity: Joint velocities. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
        """

        self.write_joint_position_to_sim(position, joint_ids=joint_ids, env_ids=env_ids)
        self.write_joint_velocity_to_sim(velocity, joint_ids=joint_ids, env_ids=env_ids)

    def write_joint_position_to_sim(
        self,
        position: torch.Tensor,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | slice | None = None,
    ):
        """Write joint positions to the simulation.

        Args:
            position: Joint positions. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
        """

        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        if joint_ids is None:
            joint_ids = slice(None)

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]

        self._data.joint_pos[env_ids, joint_ids] = position

        self._data._body_com_vel_w.timestamp = -1.0
        self._data._body_link_vel_w.timestamp = -1.0
        self._data._body_com_pose_b.timestamp = -1.0
        self._data._body_com_pose_w.timestamp = -1.0
        self._data._body_link_pose_w.timestamp = -1.0

        self._data._body_state_w.timestamp = -1.0
        self._data._body_link_state_w.timestamp = -1.0
        self._data._body_com_state_w.timestamp = -1.0

        self.root_physx_view.set_dof_positions(self._data.joint_pos, indices=physx_env_ids)

    def write_joint_velocity_to_sim(
        self,
        velocity: torch.Tensor,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | slice | None = None,
    ):
        """Write joint velocities to the simulation.

        Args:
            velocity: Joint velocities. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
        """

        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        if joint_ids is None:
            joint_ids = slice(None)

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]

        self._data.joint_vel[env_ids, joint_ids] = velocity
        self._data._previous_joint_vel[env_ids, joint_ids] = velocity
        self._data.joint_acc[env_ids, joint_ids] = 0.0

        self.root_physx_view.set_dof_velocities(self._data.joint_vel, indices=physx_env_ids)

    """
    Operations - Simulation Parameters Writers.
    """

    def write_joint_stiffness_to_sim(
        self,
        stiffness: torch.Tensor | float,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Write joint stiffness into the simulation.

        Args:
            stiffness: Joint stiffness. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the stiffness for. Defaults to None (all joints).
            env_ids: The environment indices to set the stiffness for. Defaults to None (all environments).
        """


        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        if joint_ids is None:
            joint_ids = slice(None)

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]

        self._data.joint_stiffness[env_ids, joint_ids] = stiffness

        self.root_physx_view.set_dof_stiffnesses(self._data.joint_stiffness.cpu(), indices=physx_env_ids.cpu())

    def write_joint_damping_to_sim(
        self,
        damping: torch.Tensor | float,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Write joint damping into the simulation.

        Args:
            damping: Joint damping. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the damping for. Defaults to None (all joints).
            env_ids: The environment indices to set the damping for. Defaults to None (all environments).
        """


        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        if joint_ids is None:
            joint_ids = slice(None)

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]

        self._data.joint_damping[env_ids, joint_ids] = damping

        self.root_physx_view.set_dof_dampings(self._data.joint_damping.cpu(), indices=physx_env_ids.cpu())

    def write_joint_position_limit_to_sim(
        self,
        limits: torch.Tensor | float,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
        warn_limit_violation: bool = True,
    ):
        """Write joint position limits into the simulation.

        Args:
            limits: Joint limits. Shape is (len(env_ids), len(joint_ids), 2).
            joint_ids: The joint indices to set the limits for. Defaults to None (all joints).
            env_ids: The environment indices to set the limits for. Defaults to None (all environments).
            warn_limit_violation: Whether to use warning or info level logging when default joint positions
                exceed the new limits. Defaults to True.
        """


        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        if joint_ids is None:
            joint_ids = slice(None)

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]

        self._data.joint_pos_limits[env_ids, joint_ids] = limits

        if torch.any(
            (self._data.default_joint_pos[env_ids, joint_ids] < limits[..., 0])
            | (self._data.default_joint_pos[env_ids, joint_ids] > limits[..., 1])
        ):
            self._data.default_joint_pos[env_ids, joint_ids] = torch.clamp(
                self._data.default_joint_pos[env_ids, joint_ids], limits[..., 0], limits[..., 1]
            )
            violation_message = (
                "Some default joint positions are outside of the range of the new joint limits. Default joint positions"
                " will be clamped to be within the new joint limits."
            )
            if warn_limit_violation:

                omni.log.warn(violation_message)
            else:

                omni.log.info(violation_message)

        self.root_physx_view.set_dof_limits(self._data.joint_pos_limits.cpu(), indices=physx_env_ids.cpu())




        joint_pos_mean = (self._data.joint_pos_limits[..., 0] + self._data.joint_pos_limits[..., 1]) / 2
        joint_pos_range = self._data.joint_pos_limits[..., 1] - self._data.joint_pos_limits[..., 0]
        soft_limit_factor = self.cfg.soft_joint_pos_limit_factor

        self._data.soft_joint_pos_limits[..., 0] = joint_pos_mean - 0.5 * joint_pos_range * soft_limit_factor
        self._data.soft_joint_pos_limits[..., 1] = joint_pos_mean + 0.5 * joint_pos_range * soft_limit_factor

    def write_joint_velocity_limit_to_sim(
        self,
        limits: torch.Tensor | float,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Write joint max velocity to the simulation.

        The velocity limit is used to constrain the joint velocities in the physics engine. The joint will only
        be able to reach this velocity if the joint's effort limit is sufficiently large. If the joint is moving
        faster than this velocity, the physics engine will actually try to brake the joint to reach this velocity.

        Args:
            limits: Joint max velocity. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the max velocity for. Defaults to None (all joints).
            env_ids: The environment indices to set the max velocity for. Defaults to None (all environments).
        """

        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        if joint_ids is None:
            joint_ids = slice(None)

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]

        if isinstance(limits, torch.Tensor):
            limits = limits.to(self.device)

        self._data.joint_vel_limits[env_ids, joint_ids] = limits

        self.root_physx_view.set_dof_max_velocities(self._data.joint_vel_limits.cpu(), indices=physx_env_ids.cpu())

    def write_joint_effort_limit_to_sim(
        self,
        limits: torch.Tensor | float,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Write joint effort limits into the simulation.

        The effort limit is used to constrain the computed joint efforts in the physics engine. If the
        computed effort exceeds this limit, the physics engine will clip the effort to this value.

        Args:
            limits: Joint torque limits. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the joint torque limits for. Defaults to None (all joints).
            env_ids: The environment indices to set the joint torque limits for. Defaults to None (all environments).
        """


        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        if joint_ids is None:
            joint_ids = slice(None)

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]

        if isinstance(limits, torch.Tensor):
            limits = limits.to(self.device)

        self._data.joint_effort_limits[env_ids, joint_ids] = limits

        self.root_physx_view.set_dof_max_forces(self._data.joint_effort_limits.cpu(), indices=physx_env_ids.cpu())

    def write_joint_armature_to_sim(
        self,
        armature: torch.Tensor | float,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Write joint armature into the simulation.

        The armature is directly added to the corresponding joint-space inertia. It helps improve the
        simulation stability by reducing the joint velocities.

        Args:
            armature: Joint armature. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the joint torque limits for. Defaults to None (all joints).
            env_ids: The environment indices to set the joint torque limits for. Defaults to None (all environments).
        """

        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        if joint_ids is None:
            joint_ids = slice(None)

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]

        self._data.joint_armature[env_ids, joint_ids] = armature

        self.root_physx_view.set_dof_armatures(self._data.joint_armature.cpu(), indices=physx_env_ids.cpu())

    def write_joint_friction_coefficient_to_sim(
        self,
        joint_friction_coeff: torch.Tensor | float,
        joint_dynamic_friction_coeff: torch.Tensor | float | None = None,
        joint_viscous_friction_coeff: torch.Tensor | float | None = None,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        r"""Write joint friction coefficients into the simulation.

        For Isaac Sim versions below 5.0, only the static friction coefficient is set.
        This limits the resisting force or torque up to a maximum proportional to the transmitted
        spatial force: :math:`\|F_{resist}\| \leq \mu_s \, \|F_{spatial}\|`.

        For Isaac Sim versions 5.0 and above, the static, dynamic, and viscous friction coefficients
        are set. The model combines Coulomb (static & dynamic) friction with a viscous term:

        - Static friction :math:`\mu_s` defines the maximum effort that prevents motion at rest.
        - Dynamic friction :math:`\mu_d` applies once motion begins and remains constant during motion.
        - Viscous friction :math:`c_v` is a velocity-proportional resistive term.

        Args:
            joint_friction_coeff: Static friction coefficient :math:`\mu_s`.
                Shape is (len(env_ids), len(joint_ids)). Scalars are broadcast to all selections.
            joint_dynamic_friction_coeff: Dynamic (Coulomb) friction coefficient :math:`\mu_d`.
                Same shape as above. If None, the dynamic coefficient is not updated.
            joint_viscous_friction_coeff: Viscous friction coefficient :math:`c_v`.
                Same shape as above. If None, the viscous coefficient is not updated.
            joint_ids: The joint indices to set the friction coefficients for. Defaults to None (all joints).
            env_ids: The environment indices to set the friction coefficients for. Defaults to None (all environments).
        """

        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        if joint_ids is None:
            joint_ids = slice(None)

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]

        self._data.joint_friction_coeff[env_ids, joint_ids] = joint_friction_coeff


        if joint_dynamic_friction_coeff is not None:
            self._data.joint_dynamic_friction_coeff[env_ids, joint_ids] = joint_dynamic_friction_coeff
        if joint_viscous_friction_coeff is not None:
            self._data.joint_viscous_friction_coeff[env_ids, joint_ids] = joint_viscous_friction_coeff


        physx_envs_ids_cpu = physx_env_ids.cpu()


        if int(get_version()[2]) < 5:
            self.root_physx_view.set_dof_friction_coefficients(
                self._data.joint_friction_coeff.cpu(), indices=physx_envs_ids_cpu
            )
        else:
            friction_props = self.root_physx_view.get_dof_friction_properties()
            friction_props[physx_envs_ids_cpu, :, 0] = self._data.joint_friction_coeff[physx_envs_ids_cpu, :].cpu()


            if joint_dynamic_friction_coeff is not None:
                friction_props[physx_envs_ids_cpu, :, 1] = self._data.joint_dynamic_friction_coeff[
                    physx_envs_ids_cpu, :
                ].cpu()


            if joint_viscous_friction_coeff is not None:
                friction_props[physx_envs_ids_cpu, :, 2] = self._data.joint_viscous_friction_coeff[
                    physx_envs_ids_cpu, :
                ].cpu()

            self.root_physx_view.set_dof_friction_properties(friction_props, indices=physx_envs_ids_cpu)

    def write_joint_dynamic_friction_coefficient_to_sim(
        self,
        joint_dynamic_friction_coeff: torch.Tensor | float,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        if int(get_version()[2]) < 5:
            omni.log.warn("Setting joint dynamic friction coefficients are not supported in Isaac Sim < 5.0")
            return

        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        if joint_ids is None:
            joint_ids = slice(None)

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]

        self._data.joint_dynamic_friction_coeff[env_ids, joint_ids] = joint_dynamic_friction_coeff

        friction_props = self.root_physx_view.get_dof_friction_properties()
        friction_props[physx_env_ids.cpu(), :, 1] = self._data.joint_dynamic_friction_coeff[physx_env_ids, :].cpu()
        self.root_physx_view.set_dof_friction_properties(friction_props, indices=physx_env_ids.cpu())

    def write_joint_viscous_friction_coefficient_to_sim(
        self,
        joint_viscous_friction_coeff: torch.Tensor | float,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        if int(get_version()[2]) < 5:
            omni.log.warn("Setting joint viscous friction coefficients are not supported in Isaac Sim < 5.0")
            return

        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        if joint_ids is None:
            joint_ids = slice(None)

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]

        self._data.joint_viscous_friction_coeff[env_ids, joint_ids] = joint_viscous_friction_coeff

        friction_props = self.root_physx_view.get_dof_friction_properties()
        friction_props[physx_env_ids.cpu(), :, 2] = self._data.joint_viscous_friction_coeff[physx_env_ids, :].cpu()
        self.root_physx_view.set_dof_friction_properties(friction_props, indices=physx_env_ids.cpu())

    """
    Operations - Setters.
    """

    def set_external_force_and_torque(
        self,
        forces: torch.Tensor,
        torques: torch.Tensor,
        positions: torch.Tensor | None = None,
        body_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
        is_global: bool = False,
    ):
        """Set external force and torque to apply on the asset's bodies in their local frame.

        For many applications, we want to keep the applied external force on rigid bodies constant over a period of
        time (for instance, during the policy control). This function allows us to store the external force and torque
        into buffers which are then applied to the simulation at every step. Optionally, set the position to apply the
        external wrench at (in the local link frame of the bodies).

        .. caution::
            If the function is called with empty forces and torques, then this function disables the application
            of external wrench to the simulation.

            .. code-block:: python

                # example of disabling external wrench
                asset.set_external_force_and_torque(forces=torch.zeros(0, 3), torques=torch.zeros(0, 3))

        .. caution::
            If the function is called consecutively with and with different values for ``is_global``, then the
            all the external wrenches will be applied in the frame specified by the last call.

            .. code-block:: python

                # example of setting external wrench in the global frame
                asset.set_external_force_and_torque(forces=torch.ones(1, 1, 3), env_ids=[0], is_global=True)
                # example of setting external wrench in the link frame
                asset.set_external_force_and_torque(forces=torch.ones(1, 1, 3), env_ids=[1], is_global=False)
                # Both environments will have the external wrenches applied in the link frame

        .. note::
            This function does not apply the external wrench to the simulation. It only fills the buffers with
            the desired values. To apply the external wrench, call the :meth:`write_data_to_sim` function
            right before the simulation step.

        Args:
            forces: External forces in bodies' local frame. Shape is (len(env_ids), len(body_ids), 3).
            torques: External torques in bodies' local frame. Shape is (len(env_ids), len(body_ids), 3).
            positions: Positions to apply external wrench. Shape is (len(env_ids), len(body_ids), 3). Defaults to None.
            body_ids: Body indices to apply external wrench to. Defaults to None (all bodies).
            env_ids: Environment indices to apply external wrench to. Defaults to None (all instances).
            is_global: Whether to apply the external wrench in the global frame. Defaults to False. If set to False,
                the external wrench is applied in the link frame of the articulations' bodies.
        """
        if forces.any() or torques.any():
            self.has_external_wrench = True
        else:
            self.has_external_wrench = False



        if env_ids is None:
            env_ids = self._ALL_INDICES
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)

        if body_ids is None:
            body_ids = torch.arange(self.num_bodies, dtype=torch.long, device=self.device)
        elif isinstance(body_ids, slice):
            body_ids = torch.arange(self.num_bodies, dtype=torch.long, device=self.device)[body_ids]
        elif not isinstance(body_ids, torch.Tensor):
            body_ids = torch.tensor(body_ids, dtype=torch.long, device=self.device)




        indices = body_ids.repeat(len(env_ids), 1) + env_ids.unsqueeze(1) * self.num_bodies
        indices = indices.view(-1)


        self._external_force_b.flatten(0, 1)[indices] = forces.flatten(0, 1)
        self._external_torque_b.flatten(0, 1)[indices] = torques.flatten(0, 1)

        if is_global != self._use_global_wrench_frame:
            omni.log.warn(
                f"The external wrench frame has been changed from {self._use_global_wrench_frame} to {is_global}. This"
                " may lead to unexpected behavior."
            )
            self._use_global_wrench_frame = is_global


        if positions is not None:


            self.uses_external_wrench_positions = True
            self._external_wrench_positions_b.flatten(0, 1)[indices] = positions.flatten(0, 1)
        else:

            if self.uses_external_wrench_positions:
                self._external_wrench_positions_b.flatten(0, 1)[indices] = 0.0

    def set_joint_position_target(
        self, target: torch.Tensor, joint_ids: Sequence[int] | slice | None = None, env_ids: Sequence[int] | None = None
    ):
        """Set joint position targets into internal buffers.

        This function does not apply the joint targets to the simulation. It only fills the buffers with
        the desired values. To apply the joint targets, call the :meth:`write_data_to_sim` function.

        Args:
            target: Joint position targets. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
        """

        if env_ids is None:
            env_ids = slice(None)
        if joint_ids is None:
            joint_ids = slice(None)

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]

        self._data.joint_pos_target[env_ids, joint_ids] = target

    def set_joint_velocity_target(
        self, target: torch.Tensor, joint_ids: Sequence[int] | slice | None = None, env_ids: Sequence[int] | None = None
    ):
        """Set joint velocity targets into internal buffers.

        This function does not apply the joint targets to the simulation. It only fills the buffers with
        the desired values. To apply the joint targets, call the :meth:`write_data_to_sim` function.

        Args:
            target: Joint velocity targets. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
        """

        if env_ids is None:
            env_ids = slice(None)
        if joint_ids is None:
            joint_ids = slice(None)

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]

        self._data.joint_vel_target[env_ids, joint_ids] = target

    def set_joint_effort_target(
        self, target: torch.Tensor, joint_ids: Sequence[int] | slice | None = None, env_ids: Sequence[int] | None = None
    ):
        """Set joint efforts into internal buffers.

        This function does not apply the joint targets to the simulation. It only fills the buffers with
        the desired values. To apply the joint targets, call the :meth:`write_data_to_sim` function.

        Args:
            target: Joint effort targets. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
        """

        if env_ids is None:
            env_ids = slice(None)
        if joint_ids is None:
            joint_ids = slice(None)

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]

        self._data.joint_effort_target[env_ids, joint_ids] = target

    """
    Operations - Tendons.
    """

    def set_fixed_tendon_stiffness(
        self,
        stiffness: torch.Tensor,
        fixed_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Set fixed tendon stiffness into internal buffers.

        This function does not apply the tendon stiffness to the simulation. It only fills the buffers with
        the desired values. To apply the tendon stiffness, call the :meth:`write_fixed_tendon_properties_to_sim` function.

        Args:
            stiffness: Fixed tendon stiffness. Shape is (len(env_ids), len(fixed_tendon_ids)).
            fixed_tendon_ids: The tendon indices to set the stiffness for. Defaults to None (all fixed tendons).
            env_ids: The environment indices to set the stiffness for. Defaults to None (all environments).
        """

        if env_ids is None:
            env_ids = slice(None)
        if fixed_tendon_ids is None:
            fixed_tendon_ids = slice(None)
        if env_ids != slice(None) and fixed_tendon_ids != slice(None):
            env_ids = env_ids[:, None]

        self._data.fixed_tendon_stiffness[env_ids, fixed_tendon_ids] = stiffness

    def set_fixed_tendon_damping(
        self,
        damping: torch.Tensor,
        fixed_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Set fixed tendon damping into internal buffers.

        This function does not apply the tendon damping to the simulation. It only fills the buffers with
        the desired values. To apply the tendon damping, call the :meth:`write_fixed_tendon_properties_to_sim` function.

        Args:
            damping: Fixed tendon damping. Shape is (len(env_ids), len(fixed_tendon_ids)).
            fixed_tendon_ids: The tendon indices to set the damping for. Defaults to None (all fixed tendons).
            env_ids: The environment indices to set the damping for. Defaults to None (all environments).
        """

        if env_ids is None:
            env_ids = slice(None)
        if fixed_tendon_ids is None:
            fixed_tendon_ids = slice(None)
        if env_ids != slice(None) and fixed_tendon_ids != slice(None):
            env_ids = env_ids[:, None]

        self._data.fixed_tendon_damping[env_ids, fixed_tendon_ids] = damping

    def set_fixed_tendon_limit_stiffness(
        self,
        limit_stiffness: torch.Tensor,
        fixed_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Set fixed tendon limit stiffness efforts into internal buffers.

        This function does not apply the tendon limit stiffness to the simulation. It only fills the buffers with
        the desired values. To apply the tendon limit stiffness, call the :meth:`write_fixed_tendon_properties_to_sim` function.

        Args:
            limit_stiffness: Fixed tendon limit stiffness. Shape is (len(env_ids), len(fixed_tendon_ids)).
            fixed_tendon_ids: The tendon indices to set the limit stiffness for. Defaults to None (all fixed tendons).
            env_ids: The environment indices to set the limit stiffness for. Defaults to None (all environments).
        """

        if env_ids is None:
            env_ids = slice(None)
        if fixed_tendon_ids is None:
            fixed_tendon_ids = slice(None)
        if env_ids != slice(None) and fixed_tendon_ids != slice(None):
            env_ids = env_ids[:, None]

        self._data.fixed_tendon_limit_stiffness[env_ids, fixed_tendon_ids] = limit_stiffness

    def set_fixed_tendon_position_limit(
        self,
        limit: torch.Tensor,
        fixed_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Set fixed tendon limit efforts into internal buffers.

        This function does not apply the tendon limit to the simulation. It only fills the buffers with
        the desired values. To apply the tendon limit, call the :meth:`write_fixed_tendon_properties_to_sim` function.

         Args:
             limit: Fixed tendon limit. Shape is (len(env_ids), len(fixed_tendon_ids)).
             fixed_tendon_ids: The tendon indices to set the limit for. Defaults to None (all fixed tendons).
             env_ids: The environment indices to set the limit for. Defaults to None (all environments).
        """

        if env_ids is None:
            env_ids = slice(None)
        if fixed_tendon_ids is None:
            fixed_tendon_ids = slice(None)
        if env_ids != slice(None) and fixed_tendon_ids != slice(None):
            env_ids = env_ids[:, None]

        self._data.fixed_tendon_pos_limits[env_ids, fixed_tendon_ids] = limit

    def set_fixed_tendon_rest_length(
        self,
        rest_length: torch.Tensor,
        fixed_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Set fixed tendon rest length efforts into internal buffers.

        This function does not apply the tendon rest length to the simulation. It only fills the buffers with
        the desired values. To apply the tendon rest length, call the :meth:`write_fixed_tendon_properties_to_sim` function.

        Args:
            rest_length: Fixed tendon rest length. Shape is (len(env_ids), len(fixed_tendon_ids)).
            fixed_tendon_ids: The tendon indices to set the rest length for. Defaults to None (all fixed tendons).
            env_ids: The environment indices to set the rest length for. Defaults to None (all environments).
        """

        if env_ids is None:
            env_ids = slice(None)
        if fixed_tendon_ids is None:
            fixed_tendon_ids = slice(None)
        if env_ids != slice(None) and fixed_tendon_ids != slice(None):
            env_ids = env_ids[:, None]

        self._data.fixed_tendon_rest_length[env_ids, fixed_tendon_ids] = rest_length

    def set_fixed_tendon_offset(
        self,
        offset: torch.Tensor,
        fixed_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Set fixed tendon offset efforts into internal buffers.

        This function does not apply the tendon offset to the simulation. It only fills the buffers with
        the desired values. To apply the tendon offset, call the :meth:`write_fixed_tendon_properties_to_sim` function.

        Args:
            offset: Fixed tendon offset. Shape is (len(env_ids), len(fixed_tendon_ids)).
            fixed_tendon_ids: The tendon indices to set the offset for. Defaults to None (all fixed tendons).
            env_ids: The environment indices to set the offset for. Defaults to None (all environments).
        """

        if env_ids is None:
            env_ids = slice(None)
        if fixed_tendon_ids is None:
            fixed_tendon_ids = slice(None)
        if env_ids != slice(None) and fixed_tendon_ids != slice(None):
            env_ids = env_ids[:, None]

        self._data.fixed_tendon_offset[env_ids, fixed_tendon_ids] = offset

    def write_fixed_tendon_properties_to_sim(
        self,
        fixed_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Write fixed tendon properties into the simulation.

        Args:
            fixed_tendon_ids: The fixed tendon indices to set the limits for. Defaults to None (all fixed tendons).
            env_ids: The environment indices to set the limits for. Defaults to None (all environments).
        """

        physx_env_ids = env_ids
        if env_ids is None:
            physx_env_ids = self._ALL_INDICES
        if fixed_tendon_ids is None:
            fixed_tendon_ids = slice(None)


        self.root_physx_view.set_fixed_tendon_properties(
            self._data.fixed_tendon_stiffness,
            self._data.fixed_tendon_damping,
            self._data.fixed_tendon_limit_stiffness,
            self._data.fixed_tendon_pos_limits,
            self._data.fixed_tendon_rest_length,
            self._data.fixed_tendon_offset,
            indices=physx_env_ids,
        )

    def set_spatial_tendon_stiffness(
        self,
        stiffness: torch.Tensor,
        spatial_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Set spatial tendon stiffness into internal buffers.

        This function does not apply the tendon stiffness to the simulation. It only fills the buffers with
        the desired values. To apply the tendon stiffness, call the :meth:`write_spatial_tendon_properties_to_sim` function.

        Args:
            stiffness: Spatial tendon stiffness. Shape is (len(env_ids), len(spatial_tendon_ids)).
            spatial_tendon_ids: The tendon indices to set the stiffness for. Defaults to None (all spatial tendons).
            env_ids: The environment indices to set the stiffness for. Defaults to None (all environments).
        """
        if int(get_version()[2]) < 5:
            omni.log.warn(
                "Spatial tendons are not supported in Isaac Sim < 5.0. Please update to Isaac Sim 5.0 or later."
            )
            return

        if env_ids is None:
            env_ids = slice(None)
        if spatial_tendon_ids is None:
            spatial_tendon_ids = slice(None)
        if env_ids != slice(None) and spatial_tendon_ids != slice(None):
            env_ids = env_ids[:, None]

        self._data.spatial_tendon_stiffness[env_ids, spatial_tendon_ids] = stiffness

    def set_spatial_tendon_damping(
        self,
        damping: torch.Tensor,
        spatial_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Set spatial tendon damping into internal buffers.

        This function does not apply the tendon damping to the simulation. It only fills the buffers with
        the desired values. To apply the tendon damping, call the :meth:`write_spatial_tendon_properties_to_sim` function.

        Args:
            damping: Spatial tendon damping. Shape is (len(env_ids), len(spatial_tendon_ids)).
            spatial_tendon_ids: The tendon indices to set the damping for. Defaults to None (all spatial tendons).
            env_ids: The environment indices to set the damping for. Defaults to None (all environments).
        """
        if int(get_version()[2]) < 5:
            omni.log.warn(
                "Spatial tendons are not supported in Isaac Sim < 5.0. Please update to Isaac Sim 5.0 or later."
            )
            return

        if env_ids is None:
            env_ids = slice(None)
        if spatial_tendon_ids is None:
            spatial_tendon_ids = slice(None)
        if env_ids != slice(None) and spatial_tendon_ids != slice(None):
            env_ids = env_ids[:, None]

        self._data.spatial_tendon_damping[env_ids, spatial_tendon_ids] = damping

    def set_spatial_tendon_limit_stiffness(
        self,
        limit_stiffness: torch.Tensor,
        spatial_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Set spatial tendon limit stiffness into internal buffers.

        This function does not apply the tendon limit stiffness to the simulation. It only fills the buffers with
        the desired values. To apply the tendon limit stiffness, call the :meth:`write_spatial_tendon_properties_to_sim` function.

        Args:
            limit_stiffness: Spatial tendon limit stiffness. Shape is (len(env_ids), len(spatial_tendon_ids)).
            spatial_tendon_ids: The tendon indices to set the limit stiffness for. Defaults to None (all spatial tendons).
            env_ids: The environment indices to set the limit stiffness for. Defaults to None (all environments).
        """
        if int(get_version()[2]) < 5:
            omni.log.warn(
                "Spatial tendons are not supported in Isaac Sim < 5.0. Please update to Isaac Sim 5.0 or later."
            )
            return

        if env_ids is None:
            env_ids = slice(None)
        if spatial_tendon_ids is None:
            spatial_tendon_ids = slice(None)
        if env_ids != slice(None) and spatial_tendon_ids != slice(None):
            env_ids = env_ids[:, None]

        self._data.spatial_tendon_limit_stiffness[env_ids, spatial_tendon_ids] = limit_stiffness

    def set_spatial_tendon_offset(
        self,
        offset: torch.Tensor,
        spatial_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Set spatial tendon offset efforts into internal buffers.

        This function does not apply the tendon offset to the simulation. It only fills the buffers with
        the desired values. To apply the tendon offset, call the :meth:`write_spatial_tendon_properties_to_sim` function.

        Args:
            offset: Spatial tendon offset. Shape is (len(env_ids), len(spatial_tendon_ids)).
            spatial_tendon_ids: The tendon indices to set the offset for. Defaults to None (all spatial tendons).
            env_ids: The environment indices to set the offset for. Defaults to None (all environments).
        """
        if int(get_version()[2]) < 5:
            omni.log.warn(
                "Spatial tendons are not supported in Isaac Sim < 5.0. Please update to Isaac Sim 5.0 or later."
            )
            return

        if env_ids is None:
            env_ids = slice(None)
        if spatial_tendon_ids is None:
            spatial_tendon_ids = slice(None)
        if env_ids != slice(None) and spatial_tendon_ids != slice(None):
            env_ids = env_ids[:, None]

        self._data.spatial_tendon_offset[env_ids, spatial_tendon_ids] = offset

    def write_spatial_tendon_properties_to_sim(
        self,
        spatial_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Write spatial tendon properties into the simulation.

        Args:
            spatial_tendon_ids: The spatial tendon indices to set the properties for. Defaults to None (all spatial tendons).
            env_ids: The environment indices to set the properties for. Defaults to None (all environments).
        """

        physx_env_ids = env_ids
        if env_ids is None:
            physx_env_ids = self._ALL_INDICES
        if spatial_tendon_ids is None:
            spatial_tendon_ids = slice(None)


        self.root_physx_view.set_spatial_tendon_properties(
            self._data.spatial_tendon_stiffness,
            self._data.spatial_tendon_damping,
            self._data.spatial_tendon_limit_stiffness,
            self._data.spatial_tendon_offset,
            indices=physx_env_ids,
        )

    """
    Internal helper.
    """

    def _initialize_impl(self):

        self._physics_sim_view = SimulationManager.get_physics_sim_view()

        if self.cfg.articulation_root_prim_path is not None:

            root_prim_path_expr = self.cfg.prim_path + self.cfg.articulation_root_prim_path
        else:



            first_env_matching_prim = sim_utils.find_first_matching_prim(self.cfg.prim_path)
            if first_env_matching_prim is None:
                raise RuntimeError(f"Failed to find prim for expression: '{self.cfg.prim_path}'.")
            first_env_matching_prim_path = first_env_matching_prim.GetPath().pathString


            first_env_root_prims = sim_utils.get_all_matching_child_prims(
                first_env_matching_prim_path,
                predicate=lambda prim: prim.HasAPI(UsdPhysics.ArticulationRootAPI),
                traverse_instance_prims=False,
            )
            if len(first_env_root_prims) == 0:
                raise RuntimeError(
                    f"Failed to find an articulation when resolving '{first_env_matching_prim_path}'."
                    " Please ensure that the prim has 'USD ArticulationRootAPI' applied."
                )
            if len(first_env_root_prims) > 1:
                raise RuntimeError(
                    f"Failed to find a single articulation when resolving '{first_env_matching_prim_path}'."
                    f" Found multiple '{first_env_root_prims}' under '{first_env_matching_prim_path}'."
                    " Please ensure that there is only one articulation in the prim path tree."
                )



            first_env_root_prim_path = first_env_root_prims[0].GetPath().pathString
            root_prim_path_relative_to_prim_path = first_env_root_prim_path[len(first_env_matching_prim_path) :]
            root_prim_path_expr = self.cfg.prim_path + root_prim_path_relative_to_prim_path


        self._root_physx_view = self._physics_sim_view.create_articulation_view(root_prim_path_expr.replace(".*", "*"))


        if self._root_physx_view._backend is None:
            raise RuntimeError(f"Failed to create articulation at: {root_prim_path_expr}. Please check PhysX logs.")

        if int(get_version()[2]) < 5:
            omni.log.warn(
                "Spatial tendons are not supported in Isaac Sim < 5.0: patching spatial-tendon getter"
                " and setter to use dummy value"
            )
            self._root_physx_view.max_spatial_tendons = 0
            self._root_physx_view.get_spatial_tendon_stiffnesses = lambda: torch.empty(0, device=self.device)
            self._root_physx_view.get_spatial_tendon_dampings = lambda: torch.empty(0, device=self.device)
            self._root_physx_view.get_spatial_tendon_limit_stiffnesses = lambda: torch.empty(0, device=self.device)
            self._root_physx_view.get_spatial_tendon_offsets = lambda: torch.empty(0, device=self.device)
            self._root_physx_view.set_spatial_tendon_properties = lambda *args, **kwargs: omni.log.warn(
                "Spatial tendons are not supported in Isaac Sim < 5.0: Calling"
                " set_spatial_tendon_properties has no effect"
            )


        omni.log.info(f"Articulation initialized at: {self.cfg.prim_path} with root '{root_prim_path_expr}'.")
        omni.log.info(f"Is fixed root: {self.is_fixed_base}")
        omni.log.info(f"Number of bodies: {self.num_bodies}")
        omni.log.info(f"Body names: {self.body_names}")
        omni.log.info(f"Number of joints: {self.num_joints}")
        omni.log.info(f"Joint names: {self.joint_names}")
        omni.log.info(f"Number of fixed tendons: {self.num_fixed_tendons}")


        self._data = ArticulationData(self.root_physx_view, self.device)


        self._create_buffers()

        self._process_cfg()
        self._process_actuators_cfg()
        self._process_tendons()

        self._validate_cfg()

        self.update(0.0)

        self._log_articulation_info()

    def _create_buffers(self):

        self._ALL_INDICES = torch.arange(self.num_instances, dtype=torch.long, device=self.device)


        self.has_external_wrench = False
        self.uses_external_wrench_positions = False
        self._external_force_b = torch.zeros((self.num_instances, self.num_bodies, 3), device=self.device)
        self._external_torque_b = torch.zeros_like(self._external_force_b)
        self._external_wrench_positions_b = torch.zeros_like(self._external_force_b)
        self._use_global_wrench_frame = False


        self._data.joint_names = self.joint_names
        self._data.body_names = self.body_names



        self._data.default_joint_pos_limits = self.root_physx_view.get_dof_limits().to(self.device).clone()
        self._data.default_joint_stiffness = self.root_physx_view.get_dof_stiffnesses().to(self.device).clone()
        self._data.default_joint_damping = self.root_physx_view.get_dof_dampings().to(self.device).clone()
        self._data.default_joint_armature = self.root_physx_view.get_dof_armatures().to(self.device).clone()
        if int(get_version()[2]) < 5:
            self._data.default_joint_friction_coeff = (
                self.root_physx_view.get_dof_friction_coefficients().to(self.device).clone()
            )
            self._data.default_joint_dynamic_friction_coeff = torch.zeros_like(self._data.default_joint_friction_coeff)
            self._data.default_joint_viscous_friction_coeff = torch.zeros_like(self._data.default_joint_friction_coeff)
        else:
            friction_props = self.root_physx_view.get_dof_friction_properties()
            self._data.default_joint_friction_coeff = friction_props[:, :, 0].to(self.device).clone()
            self._data.default_joint_dynamic_friction_coeff = friction_props[:, :, 1].to(self.device).clone()
            self._data.default_joint_viscous_friction_coeff = friction_props[:, :, 2].to(self.device).clone()

        self._data.joint_pos_limits = self._data.default_joint_pos_limits.clone()
        self._data.joint_vel_limits = self.root_physx_view.get_dof_max_velocities().to(self.device).clone()
        self._data.joint_effort_limits = self.root_physx_view.get_dof_max_forces().to(self.device).clone()
        self._data.joint_stiffness = self._data.default_joint_stiffness.clone()
        self._data.joint_damping = self._data.default_joint_damping.clone()
        self._data.joint_armature = self._data.default_joint_armature.clone()
        self._data.joint_friction_coeff = self._data.default_joint_friction_coeff.clone()
        self._data.joint_dynamic_friction_coeff = self._data.default_joint_dynamic_friction_coeff.clone()
        self._data.joint_viscous_friction_coeff = self._data.default_joint_viscous_friction_coeff.clone()


        self._data.default_mass = self.root_physx_view.get_masses().clone()
        self._data.default_inertia = self.root_physx_view.get_inertias().clone()


        self._data.joint_pos_target = torch.zeros(self.num_instances, self.num_joints, device=self.device)
        self._data.joint_vel_target = torch.zeros_like(self._data.joint_pos_target)
        self._data.joint_effort_target = torch.zeros_like(self._data.joint_pos_target)

        self._joint_pos_target_sim = torch.zeros_like(self._data.joint_pos_target)
        self._joint_vel_target_sim = torch.zeros_like(self._data.joint_pos_target)
        self._joint_effort_target_sim = torch.zeros_like(self._data.joint_pos_target)


        self._data.computed_torque = torch.zeros_like(self._data.joint_pos_target)
        self._data.applied_torque = torch.zeros_like(self._data.joint_pos_target)


        self._data.soft_joint_vel_limits = torch.zeros(self.num_instances, self.num_joints, device=self.device)
        self._data.gear_ratio = torch.ones(self.num_instances, self.num_joints, device=self.device)


        joint_pos_mean = (self._data.joint_pos_limits[..., 0] + self._data.joint_pos_limits[..., 1]) / 2
        joint_pos_range = self._data.joint_pos_limits[..., 1] - self._data.joint_pos_limits[..., 0]
        soft_limit_factor = self.cfg.soft_joint_pos_limit_factor

        self._data.soft_joint_pos_limits = torch.zeros(self.num_instances, self.num_joints, 2, device=self.device)
        self._data.soft_joint_pos_limits[..., 0] = joint_pos_mean - 0.5 * joint_pos_range * soft_limit_factor
        self._data.soft_joint_pos_limits[..., 1] = joint_pos_mean + 0.5 * joint_pos_range * soft_limit_factor

    def _process_cfg(self):
        """Post processing of configuration parameters."""



        default_root_state = (
            tuple(self.cfg.init_state.pos)
            + tuple(self.cfg.init_state.rot)
            + tuple(self.cfg.init_state.lin_vel)
            + tuple(self.cfg.init_state.ang_vel)
        )
        default_root_state = torch.tensor(default_root_state, dtype=torch.float, device=self.device)
        self._data.default_root_state = default_root_state.repeat(self.num_instances, 1)


        self._data.default_joint_pos = torch.zeros(self.num_instances, self.num_joints, device=self.device)
        self._data.default_joint_vel = torch.zeros_like(self._data.default_joint_pos)

        indices_list, _, values_list = string_utils.resolve_matching_names_values(
            self.cfg.init_state.joint_pos, self.joint_names
        )
        self._data.default_joint_pos[:, indices_list] = torch.tensor(values_list, device=self.device)

        indices_list, _, values_list = string_utils.resolve_matching_names_values(
            self.cfg.init_state.joint_vel, self.joint_names
        )
        self._data.default_joint_vel[:, indices_list] = torch.tensor(values_list, device=self.device)

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""

        super()._invalidate_initialize_callback(event)
        self._root_physx_view = None

    """
    Internal helpers -- Actuators.
    """

    def _process_actuators_cfg(self):
        """Process and apply articulation joint properties."""

        self.actuators = dict()


        self._has_implicit_actuators = False


        for actuator_name, actuator_cfg in self.cfg.actuators.items():

            actuator_cfg: ActuatorBaseCfg

            joint_ids, joint_names = self.find_joints(actuator_cfg.joint_names_expr)

            if len(joint_names) == 0:
                raise ValueError(
                    f"No joints found for actuator group: {actuator_name} with joint name expression:"
                    f" {actuator_cfg.joint_names_expr}."
                )


            if len(joint_names) == self.num_joints:
                joint_ids = slice(None)
            else:
                joint_ids = torch.tensor(joint_ids, device=self.device)


            actuator: ActuatorBase = actuator_cfg.class_type(
                cfg=actuator_cfg,
                joint_names=joint_names,
                joint_ids=joint_ids,
                num_envs=self.num_instances,
                device=self.device,
                stiffness=self._data.default_joint_stiffness[:, joint_ids],
                damping=self._data.default_joint_damping[:, joint_ids],
                armature=self._data.default_joint_armature[:, joint_ids],
                friction=self._data.default_joint_friction_coeff[:, joint_ids],
                dynamic_friction=self._data.default_joint_dynamic_friction_coeff[:, joint_ids],
                viscous_friction=self._data.default_joint_viscous_friction_coeff[:, joint_ids],
                effort_limit=self._data.joint_effort_limits[:, joint_ids],
                velocity_limit=self._data.joint_vel_limits[:, joint_ids],
            )

            model_type = "implicit" if actuator.is_implicit_model else "explicit"
            omni.log.info(
                f"Actuator collection: {actuator_name} with model '{actuator_cfg.class_type.__name__}'"
                f" (type: {model_type}) and joint names: {joint_names} [{joint_ids}]."
            )

            self.actuators[actuator_name] = actuator

            if isinstance(actuator, ImplicitActuator):
                self._has_implicit_actuators = True

                self.write_joint_stiffness_to_sim(actuator.stiffness, joint_ids=actuator.joint_indices)
                self.write_joint_damping_to_sim(actuator.damping, joint_ids=actuator.joint_indices)
            else:


                self.write_joint_stiffness_to_sim(0.0, joint_ids=actuator.joint_indices)
                self.write_joint_damping_to_sim(0.0, joint_ids=actuator.joint_indices)


            self.write_joint_effort_limit_to_sim(actuator.effort_limit_sim, joint_ids=actuator.joint_indices)
            self.write_joint_velocity_limit_to_sim(actuator.velocity_limit_sim, joint_ids=actuator.joint_indices)
            self.write_joint_armature_to_sim(actuator.armature, joint_ids=actuator.joint_indices)
            self.write_joint_friction_coefficient_to_sim(actuator.friction, joint_ids=actuator.joint_indices)
            if int(get_version()[2]) >= 5:
                self.write_joint_dynamic_friction_coefficient_to_sim(
                    actuator.dynamic_friction, joint_ids=actuator.joint_indices
                )
                self.write_joint_viscous_friction_coefficient_to_sim(
                    actuator.viscous_friction, joint_ids=actuator.joint_indices
                )



            self._data.default_joint_stiffness[:, actuator.joint_indices] = actuator.stiffness
            self._data.default_joint_damping[:, actuator.joint_indices] = actuator.damping
            self._data.default_joint_armature[:, actuator.joint_indices] = actuator.armature
            self._data.default_joint_friction_coeff[:, actuator.joint_indices] = actuator.friction
            if int(get_version()[2]) >= 5:
                self._data.default_joint_dynamic_friction_coeff[:, actuator.joint_indices] = actuator.dynamic_friction
                self._data.default_joint_viscous_friction_coeff[:, actuator.joint_indices] = actuator.viscous_friction


        total_act_joints = sum(actuator.num_joints for actuator in self.actuators.values())
        if total_act_joints != (self.num_joints - self.num_fixed_tendons):
            omni.log.warn(
                "Not all actuators are configured! Total number of actuated joints not equal to number of"
                f" joints available: {total_act_joints} != {self.num_joints - self.num_fixed_tendons}."
            )

        if self.cfg.actuator_value_resolution_debug_print:
            t = PrettyTable(["Group", "Property", "Name", "ID", "USD Value", "ActutatorCfg Value", "Applied"])
            for actuator_group, actuator in self.actuators.items():
                group_count = 0
                for property, resolution_details in actuator.joint_property_resolution_table.items():
                    for prop_idx, resolution_detail in enumerate(resolution_details):
                        actuator_group_str = actuator_group if group_count == 0 else ""
                        property_str = property if prop_idx == 0 else ""
                        fmt = [f"{v:.2e}" if isinstance(v, float) else str(v) for v in resolution_detail]
                        t.add_row([actuator_group_str, property_str, *fmt])
                        group_count += 1
            omni.log.warn(f"\nActuatorCfg-USD Value Discrepancy Resolution (matching values are skipped): \n{t}")

    def _process_tendons(self):
        """Process fixed and spatial tendons."""

        self._fixed_tendon_names = list()
        self._spatial_tendon_names = list()

        if self.num_fixed_tendons > 0 or self.num_spatial_tendons > 0:

            joint_paths = self.root_physx_view.dof_paths[0]


            for j in range(self.num_joints):
                usd_joint_path = joint_paths[j]

                joint = UsdPhysics.Joint.Get(self.stage, usd_joint_path)
                if joint.GetPrim().HasAPI(PhysxSchema.PhysxTendonAxisRootAPI):
                    joint_name = usd_joint_path.split("/")[-1]
                    self._fixed_tendon_names.append(joint_name)
                elif joint.GetPrim().HasAPI(PhysxSchema.PhysxTendonAttachmentRootAPI) or joint.GetPrim().HasAPI(
                    PhysxSchema.PhysxTendonAttachmentLeafAPI
                ):
                    joint_name = usd_joint_path.split("/")[-1]
                    self._spatial_tendon_names.append(joint_name)


            self._data.fixed_tendon_names = self._fixed_tendon_names
            self._data.spatial_tendon_names = self._spatial_tendon_names

            self._data.default_fixed_tendon_stiffness = self.root_physx_view.get_fixed_tendon_stiffnesses().clone()
            self._data.default_fixed_tendon_damping = self.root_physx_view.get_fixed_tendon_dampings().clone()
            self._data.default_fixed_tendon_limit_stiffness = (
                self.root_physx_view.get_fixed_tendon_limit_stiffnesses().clone()
            )
            self._data.default_fixed_tendon_pos_limits = self.root_physx_view.get_fixed_tendon_limits().clone()
            self._data.default_fixed_tendon_rest_length = self.root_physx_view.get_fixed_tendon_rest_lengths().clone()
            self._data.default_fixed_tendon_offset = self.root_physx_view.get_fixed_tendon_offsets().clone()
            self._data.default_spatial_tendon_stiffness = self.root_physx_view.get_spatial_tendon_stiffnesses().clone()
            self._data.default_spatial_tendon_damping = self.root_physx_view.get_spatial_tendon_dampings().clone()
            self._data.default_spatial_tendon_limit_stiffness = (
                self.root_physx_view.get_spatial_tendon_limit_stiffnesses().clone()
            )
            self._data.default_spatial_tendon_offset = self.root_physx_view.get_spatial_tendon_offsets().clone()


            self._data.fixed_tendon_stiffness = self._data.default_fixed_tendon_stiffness.clone()
            self._data.fixed_tendon_damping = self._data.default_fixed_tendon_damping.clone()
            self._data.fixed_tendon_limit_stiffness = self._data.default_fixed_tendon_limit_stiffness.clone()
            self._data.fixed_tendon_pos_limits = self._data.default_fixed_tendon_pos_limits.clone()
            self._data.fixed_tendon_rest_length = self._data.default_fixed_tendon_rest_length.clone()
            self._data.fixed_tendon_offset = self._data.default_fixed_tendon_offset.clone()
            self._data.spatial_tendon_stiffness = self._data.default_spatial_tendon_stiffness.clone()
            self._data.spatial_tendon_damping = self._data.default_spatial_tendon_damping.clone()
            self._data.spatial_tendon_limit_stiffness = self._data.default_spatial_tendon_limit_stiffness.clone()
            self._data.spatial_tendon_offset = self._data.default_spatial_tendon_offset.clone()

    def _apply_actuator_model(self):
        """Processes joint commands for the articulation by forwarding them to the actuators.

        The actions are first processed using actuator models. Depending on the robot configuration,
        the actuator models compute the joint level simulation commands and sets them into the PhysX buffers.
        """

        for actuator in self.actuators.values():


            control_action = ArticulationActions(
                joint_positions=self._data.joint_pos_target[:, actuator.joint_indices],
                joint_velocities=self._data.joint_vel_target[:, actuator.joint_indices],
                joint_efforts=self._data.joint_effort_target[:, actuator.joint_indices],
                joint_indices=actuator.joint_indices,
            )

            control_action = actuator.compute(
                control_action,
                joint_pos=self._data.joint_pos[:, actuator.joint_indices],
                joint_vel=self._data.joint_vel[:, actuator.joint_indices],
            )

            if control_action.joint_positions is not None:
                self._joint_pos_target_sim[:, actuator.joint_indices] = control_action.joint_positions
            if control_action.joint_velocities is not None:
                self._joint_vel_target_sim[:, actuator.joint_indices] = control_action.joint_velocities
            if control_action.joint_efforts is not None:
                self._joint_effort_target_sim[:, actuator.joint_indices] = control_action.joint_efforts


            self._data.computed_torque[:, actuator.joint_indices] = actuator.computed_effort
            self._data.applied_torque[:, actuator.joint_indices] = actuator.applied_effort

            self._data.soft_joint_vel_limits[:, actuator.joint_indices] = actuator.velocity_limit

            if hasattr(actuator, "gear_ratio"):
                self._data.gear_ratio[:, actuator.joint_indices] = actuator.gear_ratio

    """
    Internal helpers -- Debugging.
    """

    def _validate_cfg(self):
        """Validate the configuration after processing.

        Note:
            This function should be called only after the configuration has been processed and the buffers have been
            created. Otherwise, some settings that are altered during processing may not be validated.
            For instance, the actuator models may change the joint max velocity limits.
        """

        joint_pos_limits = self.root_physx_view.get_dof_limits()[0].to(self.device)
        out_of_range = self._data.default_joint_pos[0] < joint_pos_limits[:, 0]
        out_of_range |= self._data.default_joint_pos[0] > joint_pos_limits[:, 1]
        violated_indices = torch.nonzero(out_of_range, as_tuple=False).squeeze(-1)

        if len(violated_indices) > 0:

            msg = "The following joints have default positions out of the limits: \n"
            for idx in violated_indices:
                joint_name = self.data.joint_names[idx]
                joint_limit = joint_pos_limits[idx]
                joint_pos = self.data.default_joint_pos[0, idx]

                msg += f"\t- '{joint_name}': {joint_pos:.3f} not in [{joint_limit[0]:.3f}, {joint_limit[1]:.3f}]\n"
            raise ValueError(msg)


        joint_max_vel = self.root_physx_view.get_dof_max_velocities()[0].to(self.device)
        out_of_range = torch.abs(self._data.default_joint_vel[0]) > joint_max_vel
        violated_indices = torch.nonzero(out_of_range, as_tuple=False).squeeze(-1)
        if len(violated_indices) > 0:

            msg = "The following joints have default velocities out of the limits: \n"
            for idx in violated_indices:
                joint_name = self.data.joint_names[idx]
                joint_limit = [-joint_max_vel[idx], joint_max_vel[idx]]
                joint_vel = self.data.default_joint_vel[0, idx]

                msg += f"\t- '{joint_name}': {joint_vel:.3f} not in [{joint_limit[0]:.3f}, {joint_limit[1]:.3f}]\n"
            raise ValueError(msg)

    def _log_articulation_info(self):
        """Log information about the articulation.

        Note: We purposefully read the values from the simulator to ensure that the values are configured as expected.
        """


        stiffnesses = self.root_physx_view.get_dof_stiffnesses()[0].tolist()
        dampings = self.root_physx_view.get_dof_dampings()[0].tolist()

        armatures = self.root_physx_view.get_dof_armatures()[0].tolist()
        if int(get_version()[2]) < 5:
            static_frictions = self.root_physx_view.get_dof_friction_coefficients()[0].tolist()
        else:
            friction_props = self.root_physx_view.get_dof_friction_properties()
            static_frictions = friction_props[:, :, 0][0].tolist()
            dynamic_frictions = friction_props[:, :, 1][0].tolist()
            viscous_frictions = friction_props[:, :, 2][0].tolist()

        position_limits = self.root_physx_view.get_dof_limits()[0].tolist()
        velocity_limits = self.root_physx_view.get_dof_max_velocities()[0].tolist()
        effort_limits = self.root_physx_view.get_dof_max_forces()[0].tolist()

        joint_table = PrettyTable()
        joint_table.title = f"Simulation Joint Information (Prim path: {self.cfg.prim_path})"
        if int(get_version()[2]) < 5:
            joint_table.field_names = [
                "Index",
                "Name",
                "Stiffness",
                "Damping",
                "Armature",
                "Static Friction",
                "Position Limits",
                "Velocity Limits",
                "Effort Limits",
            ]
        else:
            joint_table.field_names = [
                "Index",
                "Name",
                "Stiffness",
                "Damping",
                "Armature",
                "Static Friction",
                "Dynamic Friction",
                "Viscous Friction",
                "Position Limits",
                "Velocity Limits",
                "Effort Limits",
            ]
        joint_table.float_format = ".3"
        joint_table.custom_format["Position Limits"] = lambda f, v: f"[{v[0]:.3f}, {v[1]:.3f}]"

        joint_table.align["Name"] = "l"

        for index, name in enumerate(self.joint_names):
            if int(get_version()[2]) < 5:
                joint_table.add_row([
                    index,
                    name,
                    stiffnesses[index],
                    dampings[index],
                    armatures[index],
                    static_frictions[index],
                    position_limits[index],
                    velocity_limits[index],
                    effort_limits[index],
                ])
            else:
                joint_table.add_row([
                    index,
                    name,
                    stiffnesses[index],
                    dampings[index],
                    armatures[index],
                    static_frictions[index],
                    dynamic_frictions[index],
                    viscous_frictions[index],
                    position_limits[index],
                    velocity_limits[index],
                    effort_limits[index],
                ])

        omni.log.info(f"Simulation parameters for joints in {self.cfg.prim_path}:\n" + joint_table.get_string())


        if self.num_fixed_tendons > 0:

            ft_stiffnesses = self.root_physx_view.get_fixed_tendon_stiffnesses()[0].tolist()
            ft_dampings = self.root_physx_view.get_fixed_tendon_dampings()[0].tolist()

            ft_limit_stiffnesses = self.root_physx_view.get_fixed_tendon_limit_stiffnesses()[0].tolist()
            ft_limits = self.root_physx_view.get_fixed_tendon_limits()[0].tolist()
            ft_rest_lengths = self.root_physx_view.get_fixed_tendon_rest_lengths()[0].tolist()
            ft_offsets = self.root_physx_view.get_fixed_tendon_offsets()[0].tolist()

            tendon_table = PrettyTable()
            tendon_table.title = f"Simulation Fixed Tendon Information (Prim path: {self.cfg.prim_path})"
            tendon_table.field_names = [
                "Index",
                "Stiffness",
                "Damping",
                "Limit Stiffness",
                "Limits",
                "Rest Length",
                "Offset",
            ]
            tendon_table.float_format = ".3"
            joint_table.custom_format["Limits"] = lambda f, v: f"[{v[0]:.3f}, {v[1]:.3f}]"

            for index in range(self.num_fixed_tendons):
                tendon_table.add_row([
                    index,
                    ft_stiffnesses[index],
                    ft_dampings[index],
                    ft_limit_stiffnesses[index],
                    ft_limits[index],
                    ft_rest_lengths[index],
                    ft_offsets[index],
                ])

            omni.log.info(
                f"Simulation parameters for fixed tendons in {self.cfg.prim_path}:\n" + tendon_table.get_string()
            )

        if self.num_spatial_tendons > 0:

            st_stiffnesses = self.root_physx_view.get_spatial_tendon_stiffnesses()[0].tolist()
            st_dampings = self.root_physx_view.get_spatial_tendon_dampings()[0].tolist()

            st_limit_stiffnesses = self.root_physx_view.get_spatial_tendon_limit_stiffnesses()[0].tolist()
            st_offsets = self.root_physx_view.get_spatial_tendon_offsets()[0].tolist()

            tendon_table = PrettyTable()
            tendon_table.title = f"Simulation Spatial Tendon Information (Prim path: {self.cfg.prim_path})"
            tendon_table.field_names = [
                "Index",
                "Stiffness",
                "Damping",
                "Limit Stiffness",
                "Offset",
            ]
            tendon_table.float_format = ".3"

            for index in range(self.num_spatial_tendons):
                tendon_table.add_row([
                    index,
                    st_stiffnesses[index],
                    st_dampings[index],
                    st_limit_stiffnesses[index],
                    st_offsets[index],
                ])

            omni.log.info(
                f"Simulation parameters for spatial tendons in {self.cfg.prim_path}:\n" + tendon_table.get_string()
            )

    """
    Deprecated methods.
    """

    def write_joint_friction_to_sim(
        self,
        joint_friction: torch.Tensor | float,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Write joint friction coefficients into the simulation.

        .. deprecated:: 2.1.0
            Please use :meth:`write_joint_friction_coefficient_to_sim` instead.
        """
        omni.log.warn(
            "The function 'write_joint_friction_to_sim' will be deprecated in a future release. Please"
            " use 'write_joint_friction_coefficient_to_sim' instead."
        )
        self.write_joint_friction_coefficient_to_sim(joint_friction, joint_ids=joint_ids, env_ids=env_ids)

    def write_joint_limits_to_sim(
        self,
        limits: torch.Tensor | float,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
        warn_limit_violation: bool = True,
    ):
        """Write joint limits into the simulation.

        .. deprecated:: 2.1.0
            Please use :meth:`write_joint_position_limit_to_sim` instead.
        """
        omni.log.warn(
            "The function 'write_joint_limits_to_sim' will be deprecated in a future release. Please"
            " use 'write_joint_position_limit_to_sim' instead."
        )
        self.write_joint_position_limit_to_sim(
            limits, joint_ids=joint_ids, env_ids=env_ids, warn_limit_violation=warn_limit_violation
        )

    def set_fixed_tendon_limit(
        self,
        limit: torch.Tensor,
        fixed_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Set fixed tendon position limits into internal buffers.

        .. deprecated:: 2.1.0
            Please use :meth:`set_fixed_tendon_position_limit` instead.
        """
        omni.log.warn(
            "The function 'set_fixed_tendon_limit' will be deprecated in a future release. Please"
            " use 'set_fixed_tendon_position_limit' instead."
        )
        self.set_fixed_tendon_position_limit(limit, fixed_tendon_ids=fixed_tendon_ids, env_ids=env_ids)
