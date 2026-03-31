




"""Pink IK controller implementation for IsaacLab.

This module provides integration between Pink inverse kinematics solver and IsaacLab.
Pink is a differentiable inverse kinematics solver framework that provides task-space control capabilities.

Reference:
    Pink IK Solver: https://github.com/stephane-caron/pink
"""

from __future__ import annotations

import numpy as np
import torch
from typing import TYPE_CHECKING

from pink import solve_ik

from isaaclab.assets import ArticulationCfg
from isaaclab.utils.string import resolve_matching_names_values

from .null_space_posture_task import NullSpacePostureTask
from .pink_kinematics_configuration import PinkKinematicsConfiguration

if TYPE_CHECKING:
    from .pink_ik_cfg import PinkIKControllerCfg


class PinkIKController:
    """Integration of Pink IK controller with Isaac Lab.

    The Pink IK controller solves differential inverse kinematics through weighted tasks. Each task is defined
    by a residual function e(q) that is driven to zero (e.g., e(q) = p_target - p_ee(q) for end-effector positioning).
    The controller computes joint velocities v satisfying J_e(q)v = -αe(q), where J_e(q) is the task Jacobian.
    Multiple tasks are resolved through weighted optimization, formulating a quadratic program that minimizes
    weighted task errors while respecting joint velocity limits.

    It supports user defined tasks, and we have provided a NullSpacePostureTask for maintaining desired joint configurations.

    Reference:
        Pink IK Solver: https://github.com/stephane-caron/pink
    """

    def __init__(
        self, cfg: PinkIKControllerCfg, robot_cfg: ArticulationCfg, device: str, controlled_joint_indices: list[int]
    ):
        """Initialize the Pink IK Controller.

        Args:
            cfg: The configuration for the Pink IK controller containing task definitions, solver parameters,
                and joint configurations.
            robot_cfg: The robot articulation configuration containing initial joint positions and robot
                specifications.
            device: The device to use for computations (e.g., 'cuda:0', 'cpu').
            controlled_joint_indices: A list of joint indices in the USD asset controlled by the Pink IK controller.

        Raises:
            ValueError: When joint_names or all_joint_names are not provided in the configuration.
        """
        if cfg.joint_names is None:
            raise ValueError("joint_names must be provided in the configuration")
        if cfg.all_joint_names is None:
            raise ValueError("all_joint_names must be provided in the configuration")

        self.cfg = cfg
        self.device = device
        self.controlled_joint_indices = controlled_joint_indices


        self._validate_consistency(cfg, controlled_joint_indices)


        self.pink_configuration = PinkKinematicsConfiguration(
            urdf_path=cfg.urdf_path,
            mesh_path=cfg.mesh_path,
            controlled_joint_names=cfg.joint_names,
        )




        pink_joint_names = self.pink_configuration.model.names.tolist()[1:]
        joint_pos_dict = robot_cfg.init_state.joint_pos


        indices, _, values = resolve_matching_names_values(
            joint_pos_dict, pink_joint_names, preserve_order=False, strict=False
        )
        self.init_joint_positions = np.zeros(len(pink_joint_names))
        self.init_joint_positions[indices] = np.array(values)


        for task in cfg.variable_input_tasks:

            if isinstance(task, NullSpacePostureTask):
                task.set_target(self.init_joint_positions)
                continue
            task.set_target_from_configuration(self.pink_configuration)
        for task in cfg.fixed_input_tasks:
            task.set_target_from_configuration(self.pink_configuration)


        self._setup_joint_ordering_mappings()

    def _validate_consistency(self, cfg: PinkIKControllerCfg, controlled_joint_indices: list[int]) -> None:
        """Validate consistency between controlled_joint_indices and controller configuration.

        Args:
            cfg: The Pink IK controller configuration.
            controlled_joint_indices: List of joint indices in Isaac Lab joint space.

        Raises:
            ValueError: If any consistency checks fail.
        """

        if cfg.joint_names is None:
            raise ValueError("cfg.joint_names cannot be None")
        if len(controlled_joint_indices) != len(cfg.joint_names):
            raise ValueError(
                f"Length mismatch: controlled_joint_indices has {len(controlled_joint_indices)} elements "
                f"but cfg.joint_names has {len(cfg.joint_names)} elements"
            )


        actual_joint_names = [cfg.all_joint_names[idx] for idx in controlled_joint_indices]
        if actual_joint_names != cfg.joint_names:
            mismatches = []
            for i, (actual, expected) in enumerate(zip(actual_joint_names, cfg.joint_names)):
                if actual != expected:
                    mismatches.append(
                        f"Index {i}: index {controlled_joint_indices[i]} points to '{actual}' but expected '{expected}'"
                    )
            if mismatches:
                raise ValueError(
                    "Joint name mismatch between controlled_joint_indices and cfg.joint_names:\n"
                    + "\n".join(mismatches)
                )

    def _setup_joint_ordering_mappings(self):
        """Setup joint ordering mappings between Isaac Lab and Pink conventions."""
        pink_joint_names = self.pink_configuration.all_joint_names_pinocchio_order
        isaac_lab_joint_names = self.cfg.all_joint_names

        if pink_joint_names is None:
            raise ValueError("pink_joint_names should not be None")
        if isaac_lab_joint_names is None:
            raise ValueError("isaac_lab_joint_names should not be None")


        self.isaac_lab_to_pink_ordering = np.array(
            [isaac_lab_joint_names.index(pink_joint) for pink_joint in pink_joint_names]
        )
        self.pink_to_isaac_lab_ordering = np.array(
            [pink_joint_names.index(isaac_lab_joint) for isaac_lab_joint in isaac_lab_joint_names]
        )

        pink_controlled_joint_names = self.pink_configuration.controlled_joint_names_pinocchio_order
        isaac_lab_controlled_joint_names = self.cfg.joint_names

        if pink_controlled_joint_names is None:
            raise ValueError("pink_controlled_joint_names should not be None")
        if isaac_lab_controlled_joint_names is None:
            raise ValueError("isaac_lab_controlled_joint_names should not be None")

        self.isaac_lab_to_pink_controlled_ordering = np.array(
            [isaac_lab_controlled_joint_names.index(pink_joint) for pink_joint in pink_controlled_joint_names]
        )
        self.pink_to_isaac_lab_controlled_ordering = np.array(
            [pink_controlled_joint_names.index(isaac_lab_joint) for isaac_lab_joint in isaac_lab_controlled_joint_names]
        )

    def update_null_space_joint_targets(self, curr_joint_pos: np.ndarray):
        """Update the null space joint targets.

        This method updates the target joint positions for null space posture tasks based on the current
        joint configuration. This is useful for maintaining desired joint configurations when the primary
        task allows redundancy.

        Args:
            curr_joint_pos: The current joint positions of shape (num_joints,).
        """
        for task in self.cfg.variable_input_tasks:
            if isinstance(task, NullSpacePostureTask):
                task.set_target(curr_joint_pos)

    def compute(
        self,
        curr_joint_pos: np.ndarray,
        dt: float,
    ) -> torch.Tensor:
        """Compute the target joint positions based on current state and tasks.

        Performs inverse kinematics using the Pink solver to compute target joint positions that satisfy
        the defined tasks. The solver uses quadratic programming to find optimal joint velocities that
        minimize task errors while respecting constraints.

        Args:
            curr_joint_pos: The current joint positions of shape (num_joints,).
            dt: The time step for computing joint position changes in seconds.

        Returns:
            The target joint positions as a tensor of shape (num_joints,) on the specified device.
            If the IK solver fails, returns the current joint positions unchanged to maintain stability.
        """

        curr_controlled_joint_pos = [curr_joint_pos[i] for i in self.controlled_joint_indices]


        joint_positions_pink = curr_joint_pos[self.isaac_lab_to_pink_ordering]


        self.pink_configuration.update(joint_positions_pink)


        try:
            velocity = solve_ik(
                self.pink_configuration,
                self.cfg.variable_input_tasks + self.cfg.fixed_input_tasks,
                dt,
                solver="osqp",
                safety_break=self.cfg.fail_on_joint_limit_violation,
            )
            joint_angle_changes = velocity * dt
        except (AssertionError, Exception) as e:


            if self.cfg.show_ik_warnings:
                print(
                    "Warning: IK quadratic solver could not find a solution! Did not update the target joint"
                    f" positions.\nError: {e}"
                )

            if self.cfg.xr_enabled:
                from isaaclab.ui.xr_widgets import XRVisualization

                XRVisualization.push_event("ik_error", {"error": e})
            return torch.tensor(curr_controlled_joint_pos, device=self.device, dtype=torch.float32)


        joint_vel_isaac_lab = torch.tensor(
            joint_angle_changes[self.pink_to_isaac_lab_controlled_ordering],
            device=self.device,
            dtype=torch.float32,
        )


        target_joint_pos = torch.add(
            joint_vel_isaac_lab, torch.tensor(curr_controlled_joint_pos, device=self.device, dtype=torch.float32)
        )

        return target_joint_pos
