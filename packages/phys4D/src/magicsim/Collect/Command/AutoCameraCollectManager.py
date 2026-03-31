from typing import Any, Dict, List, Sequence
import torch
from magicsim.Collect.Command import STR2TASK
from magicsim.Collect.Command.Task import Task
from magicsim.Env.Utils.file import Logger
from omegaconf import DictConfig
from magicsim.StardardEnv.Camera.TaskCameraBaseEnv import TaskCameraBaseEnv
import numpy as np


class AutoCameraCollectManager:
    def __init__(
        self,
        env: TaskCameraBaseEnv,
        num_envs: int,
        task_string: Dict[str, float],
        task_config: DictConfig,
        auto_collect_config: DictConfig,
        device=torch.device("cpu"),
        logger: Logger = None,
    ):
        self.env = env
        self.num_envs = num_envs
        self.auto_collect_config = auto_collect_config
        self.task_config = task_config
        self.task_string = task_string
        self.device = device
        self.logger = logger
        self.task_list: List[Task] = [None] * num_envs
        self.task_type_list: List[str] = [None] * num_envs
        self.info_list: List[Dict[str, Any]] = [None] * self.num_envs

    def get_next_task(self):
        task_type = np.random.choice(
            list(self.task_string.keys()), p=list(self.task_string.values())
        )
        return task_type

    def step(self, env_ids: Sequence[int]):
        camera_actions = []
        for i, env_id in enumerate(env_ids):
            if self.task_type_list[env_id] is None:
                task_type = self.get_next_task()
                self.task_type_list[env_id] = task_type
                self.task_list[env_id] = STR2TASK[task_type](
                    self.task_config[task_type], self.env, env_id, self.logger
                )
                action = self.task_list[env_id].step()

                camera_actions.append(action)
            else:
                camera_actions.append(None)
        return camera_actions

    def update(self, info: Dict[str, Any]):
        """
        Update auto-collect state for all environments and produce info compatible
        with RecordManager.update(..)['auto_collect_info'] expectations.

        Rules:
        - If only camera task exists, camera success is directly considered as success.
        - If both camera task and physics task (env.get_done) exist, both must succeed
          for overall success (logical AND).
        - If only physics task exists, physics task result is used.
        """

        camera_task_info: List[Dict[str, Any] | None] = [None] * self.num_envs
        for env_id in range(self.num_envs):
            if self.task_type_list[env_id] is None:
                continue

            task_info = self.task_list[env_id].update(info)
            camera_task_info[env_id] = task_info

            if task_info.get("finished", False) or task_info.get("truncated", 0) > 0:
                self.task_type_list[env_id] = None
                self.task_list[env_id] = None

        has_physics_done = hasattr(self.env, "get_done")
        physics_done_mask = None
        if has_physics_done:
            physics_done_mask = self.env.get_done()

        for env_id in range(self.num_envs):
            camera_info = camera_task_info[env_id]
            camera_finished = (
                camera_info.get("finished", False) if camera_info else False
            )
            camera_truncated = camera_info.get("truncated", 0) if camera_info else 0

            physics_finished = (
                bool(physics_done_mask[env_id].item()) if has_physics_done else False
            )

            if has_physics_done:
                finished = physics_finished
            elif camera_info is not None:
                finished = camera_finished
            else:
                finished = False

            truncated = camera_truncated

            if finished:
                state = "success:completed"
            elif truncated > 0:
                state = (
                    camera_info.get("state", "failed:truncated")
                    if camera_info
                    else "failed:truncated"
                )
            else:
                base_state = None
                if camera_info is not None:
                    base_state = camera_info.get("state", "running")
                elif has_physics_done:
                    base_state = "running"
                else:
                    base_state = "idle"
                state = f"success:{base_state}"

            self.info_list[env_id] = {
                "type": camera_info.get("type", "Composite")
                if camera_info
                else "Composite",
                "camera_finished": camera_finished if camera_info is not None else None,
                "physics_finished": physics_finished if has_physics_done else None,
                "finished": finished,
                "truncated": truncated,
                "state": state,
            }

        return self.info_list

    def get_manager_info(self):
        return {"task_type_list": self.task_type_list, "task_list": self.task_list}

    def reset(self):
        return [None] * self.num_envs
