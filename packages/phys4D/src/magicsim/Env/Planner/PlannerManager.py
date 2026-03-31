from typing import Dict, List, Sequence, Tuple
from omegaconf import DictConfig
from magicsim.Env.Robot.RobotManager import RobotManager
from magicsim.Env.Planner.Planner import Planner
from magicsim.Env.Utils.file import Logger
import torch
import gymnasium as gym
import numpy as np
from magicsim.Env.Planner.Dwb import Dwb

from magicsim.Env.Sensor.OccupancyManager import OccupancyManager


class PlannerManager:
    def __init__(
        self,
        num_envs: int,
        robot_config: DictConfig,
        device: torch.device,
        logger: Logger,
    ):
        """
        init corresponding parameters.
        """
        self.robot_manager = None
        self.occupancy_manager = None
        self.planners: Dict[str, Dict[str, Planner]] = {}
        self.num_envs = num_envs
        self.robot_config = robot_config
        self.device = device
        self.logger = logger
        self.single_action_space = gym.spaces.Dict()
        self.planner_configs: Dict[str, Dict[str, DictConfig]] = {}
        self.total_action_dim = 0
        self.planner_dict: Dict[str, Dict[str, Planner]] = {}
        self.planner_type_dict: Dict[str, Dict[str, str]] = {}
        self.planner_slice_dict: Dict[str, Dict[str, Tuple[int, int]]] = {}
        self.planner_flatten_list: List[Planner] = []
        self.planner_flatten_type_list: List[str] = []
        self.planner_flatten_slice_list: List[Tuple[int, int]] = []

    def initialize(
        self,
        robot_manager: RobotManager,
        occupancy_manager: OccupancyManager,
    ):
        """
        Initialize the planner after the world is initialized.

        Args:
            robot_manager: RobotManager instance
        """
        self.robot_manager = robot_manager
        self.occupancy_manager = occupancy_manager

    def step(
        self,
        action: torch.Tensor | dict[str, torch.Tensor] = None,
        env_ids: Sequence[int] = None,
    ):
        """
        Process all planners (robot) in a unified way

        Args:
            action: Robot action (optional)
            env_ids: List of environment IDs

        Returns:
            If only action is provided: returns planned robot action (backward compatible)
            If action is provided: returns planned robot action
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            if not isinstance(env_ids, torch.Tensor):
                env_ids = torch.tensor(env_ids, device=self.device)
            else:
                env_ids = env_ids.to(self.device)

        action = self._flatten_actions(action, env_ids)
        processed_action = torch.tensor([], device=self.device)
        plan_success_flags = torch.tensor([True] * len(env_ids), device=self.device)

        assert action.shape[0] == len(env_ids), (
            f"Action shape[0] {action.shape[0]} should be equal to env_ids length {len(env_ids)}"
        )
        assert action.shape[1] == self.total_action_dim, (
            f"Action shape[1] {action.shape[1]} should be equal to total_action_dim {self.total_action_dim}"
        )

        for planner, (s, e) in zip(
            self.planner_flatten_list, self.planner_flatten_slice_list
        ):
            if planner is not None:
                cur_processed_action, success_flag = planner.step(
                    action[:, s:e], env_ids
                )
                processed_action = torch.cat(
                    [processed_action, cur_processed_action], dim=1
                )
                if success_flag is not None:
                    plan_success_flags = plan_success_flags & success_flag
            else:
                if processed_action is None:
                    processed_action = action[:, s:e]
                else:
                    processed_action = torch.cat(
                        [processed_action, action[:, s:e]], dim=1
                    )
        return processed_action, plan_success_flags

    def reset_idx(self, env_ids):
        for robot_name, planners in self.planners.items():
            for planner_name, planner in planners.items():
                if planner is not None:
                    planner.reset_idx(env_ids)

    def _flatten_actions(
        self,
        actions: torch.Tensor | dict[str, torch.Tensor],
        env_ids: torch.Tensor | Sequence[int],
    ):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device)
        batch_size = len(env_ids)
        if isinstance(actions, torch.Tensor):
            if len(actions.shape) == 1:
                actions = actions.unsqueeze(0)
            if actions.shape[0] == self.num_envs and batch_size != self.num_envs:
                actions = actions[env_ids]
            return actions.to(self.device)
        chunks = []
        for rname, rspace in self.single_action_space.spaces.items():
            for k in rspace.spaces.keys():
                t = torch.as_tensor(
                    actions[rname][k], dtype=torch.float32, device=self.device
                )

                if t.shape[0] == self.num_envs and batch_size != self.num_envs:
                    t = t[env_ids]

                if t.ndim == 1:
                    if t.shape[0] == batch_size:
                        t = t.unsqueeze(1)
                    elif t.shape[0] == self.num_envs:
                        t = t[env_ids].unsqueeze(1)
                    else:
                        t = t.unsqueeze(0)
                if t.ndim > 2:
                    t = t.reshape(t.shape[0], -1)
                chunks.append(t)

        return torch.cat(chunks, dim=1)

    def sample_actions(self, batched: bool = True, env_ids: Sequence[int] = None):
        if env_ids is None:
            return (
                self.action_space.sample()
                if batched
                else self.single_action_space.sample()
            )
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.sim.device)
        sampled = {}
        for _ in range(len(env_ids)):
            one = self.single_action_space.sample()
            for rname, ract in one.items():
                if rname not in sampled:
                    sampled[rname] = {}
                for k, v in ract.items():
                    sampled[rname].setdefault(k, []).append(v)
        for rname in sampled:
            for k in sampled[rname]:
                sampled[rname][k] = torch.tensor(
                    np.array(sampled[rname][k]), dtype=torch.float32, device=self.device
                )
        return sampled

    def reset(self):
        self.setup_planner()
        env_idx = list(range(self.num_envs))
        for robot_name, planners in self.planners.items():
            for planner_name, planner in planners.items():
                if planner is not None:
                    planner.reset_idx(env_idx)

    def setup_planner(self):
        offset = 0
        for robot_name, robot_config in self.robot_config.items():
            planner_config = robot_config.get("planner", None)
            self.planner_configs[robot_name] = planner_config
            self.planner_dict[robot_name] = {}
            self.planner_type_dict[robot_name] = {}
            self.planner_slice_dict[robot_name] = {}
            self.single_action_space[robot_name] = gym.spaces.Dict()
            if robot_config.type.lower() == "manipulator":
                cur_planner = {}
                arm_action_space = self.robot_manager.single_action_space[robot_name][
                    "arm_action"
                ]
                eef_action_space = self.robot_manager.single_action_space[
                    robot_name
                ].get("eef_action", None)
                if isinstance(arm_action_space, gym.spaces.Discrete):
                    arm_action_dim = arm_action_space.n
                else:
                    arm_action_dim = arm_action_space.shape[0]
                if eef_action_space is None:
                    eef_action_dim = 0
                    eef_action_space = None
                else:
                    if isinstance(eef_action_space, gym.spaces.Discrete):
                        eef_action_dim = eef_action_space.n
                    else:
                        eef_action_dim = eef_action_space.shape[0]
                if planner_config is None:
                    cur_planner = {"arm": None, "eef": None}
                    self.planner_dict[robot_name] = cur_planner
                    self.planner_type_dict[robot_name] = {"arm": None, "eef": None}
                    self.planner_slice_dict[robot_name] = {
                        "arm": (offset, offset + arm_action_dim),
                        "eef": (
                            offset + arm_action_dim,
                            offset + arm_action_dim + eef_action_dim,
                        ),
                    }
                    self.planner_flatten_list.extend([None, None])
                    self.planner_flatten_type_list.extend([None, None])
                    self.planner_flatten_slice_list.extend(
                        [
                            (offset, offset + arm_action_dim),
                            (
                                offset + arm_action_dim,
                                offset + arm_action_dim + eef_action_dim,
                            ),
                        ]
                    )
                    offset += arm_action_dim + eef_action_dim
                    self.single_action_space[robot_name]["arm_action"] = (
                        arm_action_space
                    )
                    if eef_action_space is not None:
                        self.single_action_space[robot_name]["eef_action"] = (
                            eef_action_space
                        )
                else:
                    cur_robot_cfg = self.robot_manager.robot_cfgs[robot_name]
                    cur_planner_cfg = cur_robot_cfg.planner
                    arm_planner_config = planner_config.get("arm", None)
                    if arm_planner_config is not None:
                        arm_planner_type = arm_planner_config.get("type", None)
                        if (
                            arm_planner_type is None
                            or arm_planner_type.lower() == "default"
                        ):
                            cur_planner["arm"] = None
                            self.planner_dict[robot_name]["arm"] = cur_planner["arm"]
                            self.planner_type_dict[robot_name]["arm"] = None
                            self.planner_slice_dict[robot_name]["arm"] = (
                                offset,
                                offset + arm_action_dim,
                            )
                            self.planner_flatten_list.append(cur_planner["arm"])
                            self.planner_flatten_type_list.append(None)
                            self.planner_flatten_slice_list.append(
                                (offset, offset + arm_action_dim)
                            )
                            offset += arm_action_dim
                            self.single_action_space[robot_name]["arm_action"] = (
                                arm_action_space
                            )
                        elif arm_planner_type.lower() == "curobo":
                            from magicsim.Env.Planner.Curobo import Curobo

                            cur_planner["arm"] = Curobo(
                                self.robot_manager,
                                robot_type=robot_config.name,
                                robot_name=robot_name,
                                device=self.device,
                            )
                            arm_action_space = cur_planner_cfg.arm_action_space[
                                arm_planner_type.lower()
                            ]
                            arm_action_dim = cur_planner_cfg.arm_action_dim[
                                arm_planner_type.lower()
                            ]
                            self.planner_dict[robot_name]["arm"] = cur_planner["arm"]
                            self.planner_type_dict[robot_name]["arm"] = arm_planner_type
                            self.planner_slice_dict[robot_name]["arm"] = (
                                offset,
                                offset + arm_action_dim,
                            )
                            self.planner_flatten_list.append(cur_planner["arm"])
                            self.planner_flatten_type_list.append(arm_planner_type)
                            self.planner_flatten_slice_list.append(
                                (offset, offset + arm_action_dim)
                            )
                            offset += arm_action_dim
                            self.single_action_space[robot_name]["arm_action"] = (
                                gym.spaces.Box(
                                    low=arm_action_space[0].cpu().numpy(),
                                    high=arm_action_space[1].cpu().numpy(),
                                    shape=(arm_action_dim,),
                                )
                            )
                        else:
                            raise NotImplementedError(
                                f"Planner type {arm_planner_type} not supported."
                            )
                    else:
                        raise ValueError(
                            f"Arm planner config is not set for robot {robot_name}."
                        )

                    eef_planner_config = planner_config.get("eef", None)
                    if eef_planner_config is not None:
                        eef_planner_type = eef_planner_config.get("type", None)
                        if (
                            eef_planner_type is None
                            or eef_planner_type.lower() == "default"
                        ):
                            cur_planner["eef"] = None
                            self.planner_dict[robot_name]["eef"] = cur_planner["eef"]
                            self.planner_type_dict[robot_name]["eef"] = None
                            self.planner_slice_dict[robot_name]["eef"] = (
                                offset,
                                offset + eef_action_dim,
                            )
                            self.planner_flatten_list.append(cur_planner["eef"])
                            self.planner_flatten_type_list.append(None)
                            self.planner_flatten_slice_list.append(
                                (offset, offset + eef_action_dim)
                            )
                            offset += eef_action_dim
                            self.single_action_space[robot_name]["eef_action"] = (
                                eef_action_space
                            )
                        else:
                            raise NotImplementedError(
                                f"Planner type {eef_planner_type} not supported."
                            )
                    else:
                        if eef_action_space is not None:
                            raise ValueError(
                                f"Eef planner config is not set for robot {robot_name}."
                            )
                    self.planners[robot_name] = cur_planner
            elif robot_config.type.lower() == "mobile":
                cur_planner = {}
                base_planner_config = planner_config.get("base", None)
                base_action_space = self.robot_manager.single_action_space[robot_name][
                    "base_action"
                ]

                if isinstance(base_action_space, gym.spaces.Discrete):
                    base_action_dim = base_action_space.n
                else:
                    base_action_dim = base_action_space.shape[0]
                if planner_config is None:
                    cur_planner = {"base": None}
                    self.planner_dict[robot_name] = cur_planner
                    self.planner_type_dict[robot_name] = {"base": None}
                    self.planner_slice_dict[robot_name] = {
                        "base": (offset, offset + base_action_dim)
                    }
                    self.planner_flatten_list.append(cur_planner["base"])
                    self.planner_flatten_type_list.append(None)
                    self.planner_flatten_slice_list.append(
                        (offset, offset + base_action_dim)
                    )
                    offset += base_action_dim
                    self.single_action_space[robot_name]["base_action"] = (
                        base_action_space
                    )
                else:
                    cur_robot_cfg = self.robot_manager.robot_cfgs[robot_name]
                    cur_planner_cfg = cur_robot_cfg.planner
                    base_planner_config = planner_config.get("base", None)
                    if base_planner_config is not None:
                        base_planner_type = base_planner_config.get("type", None)
                        if (
                            base_planner_type is None
                            or base_planner_type.lower() == "default"
                        ):
                            cur_planner["base"] = None
                            self.planner_dict[robot_name]["base"] = cur_planner["base"]
                            self.planner_type_dict[robot_name]["base"] = None
                            self.planner_slice_dict[robot_name]["base"] = (
                                offset,
                                offset + base_action_dim,
                            )
                            self.planner_flatten_list.append(cur_planner["base"])
                            self.planner_flatten_type_list.append(None)
                            self.planner_flatten_slice_list.append(
                                (offset, offset + base_action_dim)
                            )
                            offset += base_action_dim
                            self.single_action_space[robot_name]["base_action"] = (
                                base_action_space
                            )
                        elif base_planner_type.lower() == "dwb":
                            cur_planner["base"] = Dwb(
                                robot_manager=self.robot_manager,
                                robot_type=robot_config.name,
                                robot_name=robot_name,
                                device=self.device,
                                occupancy_manager=self.occupancy_manager,
                            )
                            base_action_space = cur_planner_cfg.base_action_space[
                                base_planner_type.lower()
                            ]
                            base_action_dim = cur_planner_cfg.base_action_dim[
                                base_planner_type.lower()
                            ]
                            self.planner_dict[robot_name]["base"] = cur_planner["base"]
                            self.planner_type_dict[robot_name]["base"] = (
                                base_planner_type
                            )
                            self.planner_slice_dict[robot_name]["base"] = (
                                offset,
                                offset + base_action_dim,
                            )
                            self.planner_flatten_list.append(cur_planner["base"])
                            self.planner_flatten_type_list.append(base_planner_type)
                            self.planner_flatten_slice_list.append(
                                (offset, offset + base_action_dim)
                            )
                            offset += base_action_dim
                            self.single_action_space[robot_name]["base_action"] = (
                                gym.spaces.Box(
                                    low=base_action_space[0].cpu().numpy(),
                                    high=base_action_space[1].cpu().numpy(),
                                    shape=(base_action_dim,),
                                )
                            )
                        else:
                            raise NotImplementedError(
                                f"Planner type {base_planner_type} not supported."
                            )
                self.planners[robot_name] = cur_planner
            else:
                raise NotImplementedError(
                    f"Robot type {robot_config.type} not supported."
                )
        self.total_action_dim = offset - 1
        self.action_space = gym.vector.utils.batch_space(
            self.single_action_space, self.num_envs
        )

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
            env_ids = range(self.num_envs)
        for robot_name, planners in self.planners.items():
            for planner_name, planner in planners.items():
                if planner is not None:
                    planner.update_obstacles(
                        obstacle_avoidance_path_list, obstacle_ignore_path_list, env_ids
                    )
