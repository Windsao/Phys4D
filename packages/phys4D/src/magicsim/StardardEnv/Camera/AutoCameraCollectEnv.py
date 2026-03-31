from typing import Any, Dict, List, Sequence

from magicsim.Collect.Command.AutoCameraCollectManager import AutoCameraCollectManager
from magicsim.Collect.Record.RecordManager import RecordManager
from magicsim.StardardEnv.Camera.AsyncCameraEnv import AsyncCameraEnv
from omegaconf import DictConfig
from magicsim.Env.Utils.file import Logger


class AutoCameraCollectEnv(AsyncCameraEnv):
    def __init__(
        self,
        task_string: Dict[str, float],
        config: DictConfig,
        cli_args: DictConfig,
        logger: Logger,
    ):
        super().__init__(config, cli_args, logger)
        self.task_config = config.task
        self.task_string = task_string
        self.auto_collect_config = config.auto_camera_collect
        self.auto_collect_manager = AutoCameraCollectManager(
            self.env,
            self.num_envs,
            self.task_string,
            self.task_config,
            self.auto_collect_config,
            self.device,
            self.logger,
        )
        self.record_config = config.record
        self.record_manager = RecordManager(
            self.env,
            self.num_envs,
            self.record_config,
            self.device,
            self.logger,
        )

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        info = super().reset(seed, options)
        info["auto_collect_info"] = self.auto_collect_manager.reset()
        self.record_manager.reset(info)
        return info

    def step(
        self,
        info: List[Dict[str, Any]],
        env_ids: Sequence[int] | None = None,
    ):
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        camera_actions = self.auto_collect_manager.step(env_ids)

        info, valid_env_ids = super().step(camera_actions, env_ids)

        auto_collect_info = self.auto_collect_manager.update(info)
        info["auto_collect_info"] = auto_collect_info
        self.record_manager.step(info, valid_env_ids)
        info["record_info"] = self.record_manager.update(info)
        return info

    def start_collect(self):
        info = self.reset()
        while True:
            info = self.step(info)
