"""
This is a robot base environment class for sync rl training(low-level rl training) and 3D/4D data sync data with robot.
All the action here is synchronous and atomic. If you require high-level action please use AsyncRobotEnv.
The action here is atomic, meaning that the action will be executed in a single step.

"""

from typing import Any, Dict, Sequence

import torch
from magicsim.Env.Environment.SyncCollectEnv import SyncCollectEnv
from magicsim.Env.Robot.RobotManager import RobotManager
from magicsim.Env.Planner.PlannerManager import PlannerManager


class SyncRobotEnv(SyncCollectEnv):
    """
    This is a robot base environment class for sync rl training(low-level rl training) and 3D/4D data sync data with robot.
    """

    def __init__(self, config, cli_args, logger):
        super().__init__(config, cli_args, logger)
        self.robot_config = config.robot
        self.robot_manager = RobotManager(
            num_envs=self.num_envs,
            config=self.robot_config,
            device=self.device,
            logger=logger,
            seeds_per_env=self.env_seed_list,
        )
        self.planner_manager = PlannerManager(
            num_envs=self.num_envs,
            robot_config=self.robot_config,
            device=self.device,
            logger=logger,
        )

    def _setup_scene(self, sim):
        """
        Initialize the environment.
        This function will be called before simulation context create and be called by isaaclab _setup_scene function
        """
        self.robot_manager.initialize(sim)
        if self.nav_manager is not None:
            occupancy_manager = self.nav_manager.occupancy_manager
        else:
            occupancy_manager = None
        self.planner_manager.initialize(
            self.robot_manager,
            occupancy_manager=occupancy_manager,
        )
        super()._setup_scene(sim)

    def step(
        self,
        action: torch.Tensor | list[Dict] = None,
        env_ids: Sequence[int] | None = None,
    ):
        """
        Args:
            action (torch.Tensor): The action to be executed by the robot.
                The shape of the action should be (num_envs, action_dim).
        """
        if action is None:
            self.sim.sim_step()
            return {}, None

        assert len(action) == len(env_ids), (
            "Action length should be equal to env_ids length"
        )

        action_info = {}

        action_info["command"] = action

        processed_action, planner_success_flags = self.planner_manager.step(
            action=action, env_ids=env_ids
        )

        action_info["robot_action"] = processed_action
        step_info = self.robot_manager.step(
            action=action_info["robot_action"], env_ids=env_ids
        )
        action_info.update(step_info)

        return action_info, planner_success_flags

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        """
        Reset the environment.
        This should only be called once at the beginning of the environment.
        In this function, we will call scene_manager.reset(soft=False) to load all the objects managed in scene manager
        It will also reset the reset count.
        """
        super().reset(seed=seed, options=options)
        print("Scene Reset Finished")
        self.robot_manager.reset()
        self.planner_manager.reset()

    def reset_idx(
        self,
        env_ids: Sequence[int] = None,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        """
        Reset specific environments.
        """
        if env_ids is None:
            env_id_list = list(range(self.num_envs))
        elif isinstance(env_ids, torch.Tensor):
            env_id_list = env_ids.detach().cpu().tolist()
        else:
            env_id_list = [int(i) for i in env_ids]

        super().reset_idx(env_ids=env_id_list, seed=seed, options=options)
        self.robot_manager.reset_idx(env_ids=env_id_list)
        self.planner_manager.reset_idx(env_ids=env_id_list)
        self.sim.sim_step()

    def _update_seed_managers(self):
        super()._update_seed_managers()
        seeds = self.env_seeds
        if hasattr(self, "robot_manager") and self.robot_manager is not None:
            self.robot_manager.update_env_seeds(seeds)

    def _apply_action(self, sim):
        self.robot_manager._apply_action(sim)

    def _pre_physics_step(self, sim, actions, env_ids):
        self.robot_manager.pre_physics_step(sim, actions, env_ids)
