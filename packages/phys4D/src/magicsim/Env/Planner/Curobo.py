from magicsim.Env.Robot.RobotManager import RobotManager


from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.util.usd_helper import UsdHelper
from curobo.util.logger import setup_curobo_logger
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
)

import torch
from termcolor import cprint
from isaacsim.core.utils.stage import get_current_stage
from magicsim.Env.Planner.Planner import Planner
from magicsim.Env.Planner.Utils import (
    quat_mul,
    quat_inv,
    quat_normalize,
    quat_angle_between,
)
from magicsim.Env.Utils.rotations import quat_to_rot_matrix


class Curobo(Planner):
    """
    Curobo Motion Planning.
    """

    def __init__(
        self,
        robot_manager: RobotManager,
        robot_type: str = "Franka",
        robot_name: str = "Franka",
        device: torch.device = torch.device("cpu"),
    ):
        """
        initialize Curobo instance and warmup.
        """
        self.robot_manager = robot_manager
        self.robot_name = robot_name
        self.device = device
        self.robot_yml_file = f"magicsim_{robot_type.lower()}.yml"
        self.robot_cfg = load_yaml(
            join_path(get_robot_configs_path(), self.robot_yml_file)
        )["robot_cfg"]
        self.robot_lock_joints = self.robot_cfg["kinematics"].get("lock_joints", None)

        self.history_target = [None] * self.robot_manager.num_envs
        self.cmd_plan = [None] * self.robot_manager.num_envs
        self.cmd_plan_step = [0] * self.robot_manager.num_envs

        self.current_target = [None] * self.robot_manager.num_envs

        self.last_proposed_target = [None] * self.robot_manager.num_envs

        self.current_plan = [None] * self.robot_manager.num_envs

        self.current_plan_step = [0] * self.robot_manager.num_envs

        self.pos_threshold = 0.01
        self.rot_threshold_deg = 2.0
        self.rot_threshold = torch.deg2rad(
            torch.tensor(self.rot_threshold_deg, device=self.device)
        )

        self.same_target_pos_threshold = 0.001
        self.same_target_rot_threshold_deg = 0.1
        self.same_target_rot_threshold = torch.deg2rad(
            torch.tensor(self.same_target_rot_threshold_deg, device=self.device)
        )

        self.periodic_replan_steps = None
        self.step_count = [0] * self.robot_manager.num_envs

        self.force_replan = [False] * self.robot_manager.num_envs

        setup_curobo_logger("warn")

        self.usd_help = None

        self.tensor_args = TensorDeviceType()

        self.world_cfg_list = []
        for i in range(self.robot_manager.num_envs):
            world_cfg = WorldConfig()
            self.world_cfg_list.append(world_cfg)

        motion_gen_config = MotionGenConfig.load_from_robot_config(
            self.robot_cfg,
            self.world_cfg_list,
            self.tensor_args,
            collision_checker_type=CollisionCheckerType.MESH,
            use_cuda_graph=True,
            interpolation_dt=0.03,
            collision_cache={"obb": 25, "mesh": 25},
            collision_activation_distance=0.025,
            maximum_trajectory_dt=0.25,
        )
        self.motion_gen = MotionGen(motion_gen_config)

        self.plan_config = MotionGenPlanConfig(
            enable_graph=False,
            max_attempts=2,
            enable_finetune_trajopt=True,
            enable_graph_attempt=None,
        )

        cprint("warmup curobo instance", "green", attrs=["bold"])

    def world_to_robot_frame(
        self,
        target_pos: list,
        robot_base_pose: torch.Tensor,
        robot_base_quat: torch.Tensor,
        env_ids: list = None,
    ) -> list:
        """
        Transform target poses from world frame to robot base frame.

        Args:
            target_pos: List of target poses in world frame, format:
                       [[x, y, z, qw, qx, qy, qz], ...] for each env in env_ids
                       Can also be a list containing tensors or a tensor itself
            robot_base_pose: Robot base positions in world frame, shape [num_envs, 3]
            robot_base_quat: Robot base quaternions in world frame, shape [num_envs, 4] with [w, x, y, z]
            env_ids: List of environment IDs corresponding to target_pos. If None, assumes sequential.

        Returns:
            List of target poses in robot base frame, same format as input
        """
        num_targets = len(target_pos)
        device = robot_base_pose.device

        if isinstance(target_pos, torch.Tensor):
            target_tensor = target_pos.to(device=device, dtype=torch.float32)
        elif isinstance(target_pos[0], torch.Tensor):
            target_tensor = torch.stack(
                [
                    t.to(device=device, dtype=torch.float32)
                    if isinstance(t, torch.Tensor)
                    else torch.tensor(t, device=device, dtype=torch.float32)
                    for t in target_pos
                ],
                dim=0,
            )
        else:
            target_tensor = torch.tensor(target_pos, device=device, dtype=torch.float32)

        target_positions = target_tensor[:, :3]
        target_quats = target_tensor[:, 3:]

        if env_ids is not None:
            env_ids_tensor = torch.tensor(env_ids, device=device, dtype=torch.long)
            base_positions = robot_base_pose[env_ids_tensor]
            base_quats = robot_base_quat[env_ids_tensor]
        else:
            base_positions = robot_base_pose[:num_targets]
            base_quats = robot_base_quat[:num_targets]

        pos_relative = target_positions - base_positions

        base_rot_matrices = quat_to_rot_matrix(base_quats)

        base_rot_inv = base_rot_matrices.transpose(-2, -1)

        pos_relative_rotated = torch.bmm(
            base_rot_inv, pos_relative.unsqueeze(-1)
        ).squeeze(-1)

        base_quats_inv = quat_inv(base_quats)
        quats_relative = quat_mul(base_quats_inv, target_quats)

        result = []
        for i in range(num_targets):
            pos = pos_relative_rotated[i].tolist()
            quat = quats_relative[i].tolist()
            result.append(pos + quat)

        return result

    def step(self, target_pos, env_ids, relative_to_world_frame=True):
        """
        Step function with debouncing mechanism. Wraps forward() and manages replanning logic.

        Args:
            target_pos (list): target pos in different envs, format: [[x, y, z, qw, qx, qy, qz], ......, [x, y, z, qw, qx, qy, qz]]
            env_ids (list or torch.Tensor): env id to be solved, example: [0, 1, 3] means env 0, 1, 3 will be solved.
            relative_to_world_frame (bool): whether the target pos is relative to world frame, default is True.

        Returns:
            torch.Tensor: Actions for the specified environments, shape [len(env_ids), action_dim]
            torch.Tensor: Success flags for each environment, shape [len(env_ids)] with dtype bool
        """
        device = self.device

        if isinstance(env_ids, torch.Tensor):
            env_ids_list = env_ids.cpu().tolist()
            if isinstance(env_ids_list, int):
                env_ids_list = [env_ids_list]
        else:
            env_ids_list = list(env_ids)

        if isinstance(target_pos, torch.Tensor):
            target_tensor = target_pos.to(device=device, dtype=torch.float32)
        elif len(target_pos) > 0 and isinstance(target_pos[0], torch.Tensor):
            target_tensor = torch.stack(
                [
                    t.to(device=device, dtype=torch.float32)
                    if isinstance(t, torch.Tensor)
                    else torch.tensor(t, device=device, dtype=torch.float32)
                    for t in target_pos
                ],
                dim=0,
            )
        else:
            target_tensor = torch.tensor(target_pos, device=device, dtype=torch.float32)

        if target_tensor.ndim == 1:
            target_tensor = target_tensor.unsqueeze(0)

        target_positions = target_tensor[:, :3]
        target_quats = target_tensor[:, 3:]
        target_quats = quat_normalize(target_quats)

        need_replan = []
        replan_env_ids = []
        replan_target_pos = []

        for i, env_id in enumerate(env_ids_list):
            pos_in = target_positions[i]
            quat_in = target_quats[i]

            if self.force_replan[env_id]:
                print(f"Force replan for env {env_id}")
                need_replan.append(True)
                replan_env_ids.append(env_id)
                target_item = target_tensor[i].cpu().tolist()
                replan_target_pos.append(target_item)
                self.current_target[env_id] = target_item
                self.force_replan[env_id] = False

                self.last_proposed_target[env_id] = target_item
                continue

            if self.periodic_replan_steps is not None:
                print(f"Periodic replan for env {env_id}")
                self.step_count[env_id] += 1
                if self.step_count[env_id] >= self.periodic_replan_steps:
                    need_replan.append(True)
                    replan_env_ids.append(env_id)
                    target_item = target_tensor[i].cpu().tolist()
                    replan_target_pos.append(target_item)
                    self.current_target[env_id] = target_item
                    self.step_count[env_id] = 0

                    self.last_proposed_target[env_id] = target_item
                    continue

            last_prop = self.last_proposed_target[env_id]
            current_tgt = self.current_target[env_id]

            if last_prop is None:
                need_replan.append(True)
                replan_env_ids.append(env_id)

                target_item = target_tensor[i].cpu().tolist()
                replan_target_pos.append(target_item)

                self.last_proposed_target[env_id] = target_item
                continue

            last_prop_tensor = torch.tensor(
                last_prop, device=device, dtype=torch.float32
            )
            last_prop_pos = last_prop_tensor[:3]
            last_prop_quat = quat_normalize(last_prop_tensor[3:].unsqueeze(0)).squeeze(
                0
            )

            pos_diff = torch.linalg.norm(pos_in - last_prop_pos)
            rot_diff = quat_angle_between(
                last_prop_quat.unsqueeze(0), quat_in.unsqueeze(0)
            ).squeeze(0)

            is_stable = (pos_diff < self.pos_threshold) & (
                rot_diff < self.rot_threshold
            )

            if is_stable:
                if current_tgt is None:
                    print(f"No current target, need to plan for env {env_id}")

                    need_replan.append(True)
                    replan_env_ids.append(env_id)

                    target_item = target_tensor[i].cpu().tolist()
                    replan_target_pos.append(target_item)
                else:
                    current_tgt_tensor = torch.tensor(
                        current_tgt, device=device, dtype=torch.float32
                    )
                    current_tgt_pos = current_tgt_tensor[:3]
                    current_tgt_quat = quat_normalize(
                        current_tgt_tensor[3:].unsqueeze(0)
                    ).squeeze(0)

                    pos_diff_current = torch.linalg.norm(pos_in - current_tgt_pos)
                    rot_diff_current = quat_angle_between(
                        current_tgt_quat.unsqueeze(0), quat_in.unsqueeze(0)
                    ).squeeze(0)

                    is_same_as_current = (
                        pos_diff_current < self.same_target_pos_threshold
                    ) & (rot_diff_current < self.same_target_rot_threshold)

                    if is_same_as_current:
                        if self.current_plan[env_id] is not None:
                            need_replan.append(False)
                        else:
                            print(
                                f"Same target as previous failed plan for env {env_id}, skipping replan"
                            )
                            need_replan.append(False)
                    else:
                        print(
                            f"Different from current target, need to replan for env {env_id}"
                        )
                        need_replan.append(True)
                        replan_env_ids.append(env_id)

                        target_item = target_tensor[i].cpu().tolist()
                        replan_target_pos.append(target_item)

            else:
                print(f"Target is not stable yet, don't replan for env {env_id}")
                need_replan.append(False)

            target_item = target_tensor[i].cpu().tolist()
            self.last_proposed_target[env_id] = target_item

        actions_dict = {}
        success_dict = {}
        if len(replan_env_ids) > 0:
            replan_success = self.forward(
                replan_target_pos, replan_env_ids, relative_to_world_frame
            )

            robot_states = self.robot_manager.get_robot_state(noise_flag=False)[0]
            robot_joint_poses = robot_states[self.robot_name]["joint_pos"]
            robot_dof_name = self.robot_manager.robots[self.robot_name].joint_names

            for idx, env_id in enumerate(replan_env_ids):
                success_flag = replan_success[idx].item()

                success_dict[env_id] = success_flag

                if success_flag:
                    if idx < len(replan_target_pos):
                        self.current_target[env_id] = replan_target_pos[idx]

                    if self.cmd_plan[env_id] is not None:
                        plan_tensor = self.cmd_plan[env_id].position.to(device)
                        self.current_plan[env_id] = plan_tensor
                        self.current_plan_step[env_id] = 1

                        if plan_tensor.shape[0] > 0:
                            actions_dict[env_id] = plan_tensor[0]
                        else:
                            joint_pos = robot_joint_poses[env_id]
                            action = []
                            for x in robot_dof_name:
                                if x in self.robot_lock_joints:
                                    continue
                                joint_idx = self.robot_manager.robots[
                                    self.robot_name
                                ].find_joints(x)[0]
                                action.append(joint_pos[joint_idx])
                            actions_dict[env_id] = torch.tensor(
                                action, device=device, dtype=torch.float32
                            )
                    else:
                        raise RuntimeError(
                            f"Plan failed for env {env_id} but should have been successful"
                        )
                else:
                    if idx < len(replan_target_pos):
                        self.current_target[env_id] = replan_target_pos[idx]
                        print(
                            f"Plan failed for env {env_id}, but current_target recorded to avoid repeated planning"
                        )

                    if self.current_plan[env_id] is not None:
                        plan = self.current_plan[env_id]
                        step_idx = self.current_plan_step[env_id]
                        if step_idx < plan.shape[0]:
                            actions_dict[env_id] = plan[step_idx]
                            self.current_plan_step[env_id] += 1
                        else:
                            actions_dict[env_id] = plan[-1]
                    else:
                        joint_pos = robot_joint_poses[env_id]
                        action = []
                        for x in robot_dof_name:
                            if x in self.robot_lock_joints:
                                continue
                            joint_idx = self.robot_manager.robots[
                                self.robot_name
                            ].find_joints(x)[0]
                            action.append(joint_pos[joint_idx])
                        actions_dict[env_id] = torch.tensor(
                            action, device=device, dtype=torch.float32
                        )

        actions_list = []
        success_list = []
        for i, env_id in enumerate(env_ids_list):
            if not need_replan[i]:
                if self.current_plan[env_id] is not None:
                    plan = self.current_plan[env_id]
                    step_idx = self.current_plan_step[env_id]

                    if step_idx < plan.shape[0]:
                        action = plan[step_idx]
                        self.current_plan_step[env_id] += 1
                    else:
                        action = plan[-1]
                    actions_list.append(action)

                    success_list.append(True)
                else:
                    robot_states = self.robot_manager.get_robot_state(noise_flag=False)[
                        0
                    ]
                    robot_joint_poses = robot_states[self.robot_name]["joint_pos"]
                    robot_dof_name = self.robot_manager.robots[
                        self.robot_name
                    ].joint_names

                    joint_pos = robot_joint_poses[env_id]
                    action = []
                    for x in robot_dof_name:
                        if x in self.robot_lock_joints:
                            continue
                        joint_idx = self.robot_manager.robots[
                            self.robot_name
                        ].find_joints(x)[0]
                        action.append(joint_pos[joint_idx])
                    actions_list.append(
                        torch.tensor(action, device=device, dtype=torch.float32)
                    )

                    success_list.append(False)
            else:
                actions_list.append(actions_dict[env_id])

                success_list.append(success_dict[env_id])

        actions = torch.stack(actions_list, dim=0)
        success_flags = torch.tensor(success_list, device=device, dtype=torch.bool)
        return actions, success_flags

    def forward(self, target_pos, env_ids, relative_to_world_frame=True):
        """
        Args:
            target_pos (list): target pos in different envs, format: [[x, y, z, qw, qx, qy, qz], ......, [x, y, z, qw, qx, qy, qz]]
            env_ids (list): env id to be solved, example: [0, 1, 3] means env 0, 1, 3 will be solved.
            relative_to_world_frame (bool): whether the target pos is relative to world frame, default is True.
        Returns:
            torch.Tensor: Success flags for each environment, shape [len(env_ids)] with dtype bool
        """

        robot_base_pose = self.robot_manager.get_robot_state(noise_flag=False)[0][
            self.robot_name
        ]["base_pos"]
        robot_base_quat = self.robot_manager.get_robot_state(noise_flag=False)[0][
            self.robot_name
        ]["base_quat"]
        if relative_to_world_frame:
            target_pos = self.world_to_robot_frame(
                target_pos, robot_base_pose, robot_base_quat, env_ids
            )

        robot_states = self.robot_manager.get_robot_state(noise_flag=False)[0]
        robot_joint_poses = robot_states[self.robot_name]["joint_pos"]
        robot_joint_vels = robot_states[self.robot_name]["joint_vel"]
        robot_eef_poses = robot_states[self.robot_name]["eef_relative_pos"]
        robot_eef_quats = robot_states[self.robot_name]["eef_relative_quat"]
        robot_eef = torch.cat([robot_eef_poses, robot_eef_quats], dim=1)
        robot_dof_name = self.robot_manager.robots[self.robot_name].joint_names

        final_target_pos = [None] * self.robot_manager.num_envs
        for i in range(len(env_ids)):
            final_target_pos[env_ids[i]] = target_pos[i]
        for i in range(len(final_target_pos)):
            if final_target_pos[i] is None:
                final_target_pos[i] = robot_eef[i].tolist()

        final_target_positions = [pos[:3] for pos in final_target_pos]
        final_target_quaternions = [pos[3:] for pos in final_target_pos]
        ik_goal = Pose(
            position=self.tensor_args.to_device(final_target_positions),
            quaternion=self.tensor_args.to_device(final_target_quaternions),
        )

        for i in range(self.robot_manager.num_envs):
            robot_joint_pose = robot_joint_poses[i]
            robot_joint_vel = robot_joint_vels[i]
            cur_js = JointState(
                position=self.tensor_args.to_device(robot_joint_pose).view(1, -1),
                velocity=self.tensor_args.to_device(robot_joint_vel).view(1, -1) * 0.0,
                acceleration=self.tensor_args.to_device(robot_joint_pose).view(1, -1)
                * 0.0,
                jerk=self.tensor_args.to_device(robot_joint_pose).view(1, -1) * 0.0,
                joint_names=robot_dof_name,
            )
            if i == 0:
                self.full_js = cur_js
            else:
                self.full_js = self.full_js.stack(cur_js)
        self.full_js = self.full_js.get_ordered_joint_state(
            self.motion_gen.kinematics.joint_names
        )

        result = self.motion_gen.plan_batch_env(
            self.full_js, ik_goal, self.plan_config.clone()
        )

        success_flags = {}

        if torch.count_nonzero(result.success) > 0:
            trajs = result.get_paths()
            for s in range(len(result.success)):
                if s in env_ids:
                    if result.success[s]:
                        self.cmd_plan[s] = self.motion_gen.get_full_js(trajs[s])

                        idx_list = []
                        common_js_names = []
                        for x in robot_dof_name:
                            if x in self.robot_lock_joints:
                                continue
                            if x in self.cmd_plan[s].joint_names:
                                idx_list.append(
                                    self.robot_manager.robots[
                                        self.robot_name
                                    ].find_joints(x)[0]
                                )
                                common_js_names.append(x)

                        self.cmd_plan[s] = self.cmd_plan[s].get_ordered_joint_state(
                            common_js_names
                        )
                        success_flags[s] = True
                    else:
                        print(f"Plan failed for env {s}")
                        success_flags[s] = False
        else:
            for s in env_ids:
                success_flags[s] = False

        success_list = []

        for env_id in env_ids:
            if env_id in success_flags and success_flags[env_id]:
                success_list.append(True)
            else:
                success_list.append(False)

        success_tensor = torch.tensor(
            success_list, device=self.device, dtype=torch.bool
        )

        return success_tensor

    def reset_idx(self, env_ids):
        for id in env_ids:
            self.history_target[id] = None
            self.cmd_plan[id] = None
            self.cmd_plan_step[id] = 0

            self.current_target[id] = None
            self.last_proposed_target[id] = None
            self.current_plan[id] = None
            self.current_plan_step[id] = 0

            self.step_count[id] = 0
            self.force_replan[id] = False

    def force_replan_all(self, env_ids=None):
        """
        Force replanning for specified environments (or all if env_ids is None).
        Useful for dynamic obstacle avoidance.

        Args:
            env_ids: List of environment IDs to force replan. If None, replans all envs.
        """
        if env_ids is None:
            env_ids = list(range(self.robot_manager.num_envs))

        for env_id in env_ids:
            self.force_replan[env_id] = True

    def set_periodic_replan_steps(self, steps: int):
        """
        Set periodic replanning interval. Replan every N steps.

        Args:
            steps: Number of steps between replanning. Set to None to disable periodic replanning.
        """
        self.periodic_replan_steps = steps

        self.step_count = [0] * self.robot_manager.num_envs

    def update_obstacles(
        self,
        obstacle_avoidance_path_list: list = None,
        obstacle_ignore_path_list: list = None,
        env_ids: list = None,
    ):
        """
        Args:
            obstacle_avoidance_path_list: list of obstacle avoidance paths
            obstacle_ignore_path_list: list of obstacle ignore paths
            env_ids: list of env ids to update obstacles
        """
        if env_ids is None:
            env_ids = range(self.robot_manager.num_envs)
        if self.usd_help is None:
            self.usd_help = UsdHelper()
            self.usd_help.load_stage(get_current_stage())
        for i in env_ids:
            if obstacle_avoidance_path_list is None:
                world_cfg = WorldConfig()
            else:
                obstacle_avoidance_paths = []
                for obstacle_avoidance_path in obstacle_avoidance_path_list:
                    obstacle_avoidance_paths.append(
                        f"/World/envs/env_{i}/{obstacle_avoidance_path}"
                    )

                obstacle_ignore_paths = [f"/World/envs/env_{i}/robot_0"]
                if obstacle_ignore_path_list is not None:
                    for obstacle_ignore_path in obstacle_ignore_path_list:
                        obstacle_ignore_paths.append(
                            f"/World/envs/env_{i}/{obstacle_ignore_path}"
                        )

                world_cfg = self.usd_help.get_obstacles_from_stage(
                    only_paths=obstacle_avoidance_paths,
                    reference_prim_path=f"/World/envs/env_{i}/robot_0",
                    ignore_substring=obstacle_ignore_paths,
                ).get_collision_check_world()

                self.motion_gen.world_coll_checker.load_collision_model(
                    world_cfg,
                    env_idx=i,
                    fix_cache_reference=self.motion_gen.use_cuda_graph,
                )
            self.world_cfg_list[i] = world_cfg
