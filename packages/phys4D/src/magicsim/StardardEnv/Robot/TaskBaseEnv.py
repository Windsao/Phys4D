import gymnasium as gym
from magicsim.Env.Environment.SyncRobotEnv import SyncRobotEnv
from typing import Any, Dict, Sequence
import torch
from magicsim.Env.Environment.Utils.Basic import seed_everywhere


class TaskBaseEnv(gym.Env):
    """
    Base Environment for Robot Tasks.
    """

    def __init__(self, config, cli_args, logger):
        self.config = config
        self.Scene_Config = config.Scene

        if hasattr(config, "Nav") and "nav" not in self.Scene_Config:
            from omegaconf import OmegaConf

            struct_flag = OmegaConf.is_struct(self.Scene_Config)
            if struct_flag:
                OmegaConf.set_struct(self.Scene_Config, False)
            self.Scene_Config.nav = config.Nav
            if struct_flag:
                OmegaConf.set_struct(self.Scene_Config, True)
        self.scene: SyncRobotEnv = gym.make(
            "SyncRobotEnv-V0",
            config=self.Scene_Config,
            cli_args=cli_args,
            logger=logger,
        )
        self.device = self.scene.device
        self.num_envs = self.scene.num_envs
        self.config = config
        self.cli_args = cli_args
        self.logger = logger

        self._reward_mode = None

        self.episode_length_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.reset_terminated = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.reset_truncated = torch.zeros_like(self.reset_terminated)
        self.reset_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.last_action = None
        self._configure_gym_env_spaces()

    @property
    def reward_mode(self):
        return self._reward_mode

    def _configure_gym_env_spaces(self):
        """
        Configure the observation and action spaces for the Gym environment.
        This method should be overridden by subclasses to define specific spaces.
        """

        self.action_space = self.scene.robot_manager.action_space
        self.observation_space = self.get_obs_space()

    def sample_actions(
        self, batched: bool = True, env_ids: Sequence[int] | None = None
    ) -> torch.Tensor | list[Dict]:
        """
        Sample random actions for the robot.
        Args:
            batched (bool): If True, sample actions for all environments. If False, sample for a single environment.
        """
        return self.scene.robot_manager.sample_actions(batched=batched, env_ids=env_ids)

    def get_obs(
        self,
    ) -> Dict[str, Any]:
        obs_dict = {}
        obs_dict["policy_obs"] = self.get_policy_obs()
        obs_dict["privilege_obs"] = self.get_privilege_obs()

        obs_dict["policy_obs"]["last_action"] = self.last_action
        return obs_dict

    def get_policy_obs(
        self,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def get_privilege_obs(
        self,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def process_action(self, action: torch.Tensor | list[Dict]):
        raise NotImplementedError

    def step(
        self,
        action: torch.Tensor | list[Dict],
        env_ids: Sequence[int] | None = None,
        failed_env_ids: Sequence[int] | None = None,
    ):
        """
        Args:
            action (torch.Tensor): The action to be executed by the robot.
                The shape of the action should be (num_envs, action_dim).
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            if not isinstance(env_ids, torch.Tensor):
                env_ids = torch.tensor(env_ids, device=self.device)
            else:
                env_ids = env_ids.to(self.device)
        if action is None:
            self.scene.sim.sim_step()
            self.last_action = None
            step_success_flags = None
        else:
            self.processed_action = self.process_action(action)
            action_info, step_success_flags = self.scene.step(
                self.processed_action, env_ids=env_ids
            )

            padded_action_info = self._pad_action_info_to_num_envs(
                action_info, env_ids, self.num_envs
            )
            padded_success_flags = self._pad_success_flags_to_num_envs(
                step_success_flags, env_ids, self.num_envs
            )
            self.last_action = padded_action_info
            step_success_flags = padded_success_flags

        reward = self.get_reward(action, env_ids)
        terminated, truncated = self.get_termination()
        self.reset_terminated[terminated] = 1
        self.reset_truncated[truncated] = 1
        if failed_env_ids is not None:
            self.reset_truncated[failed_env_ids] = 1
        if step_success_flags is not None:
            failed_mask = ~torch.isnan(step_success_flags) & (step_success_flags == 0.0)
            self.reset_truncated[failed_mask] = 1
            truncated[failed_mask] = 1
        self.episode_length_buf[env_ids] += 1
        self.reset_buf = self.reset_terminated | self.reset_truncated

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
            self.scene.sim.sim_step()

        assert len(reward) == self.num_envs, "Reward length should be equal to num_envs"
        assert len(terminated) == self.num_envs, (
            "Terminated length should be equal to num_envs"
        )
        assert len(truncated) == self.num_envs, (
            "Truncated length should be equal to num_envs"
        )

        info = self.get_info()
        obs = self.get_obs()

        self._check_dict_values_length(obs, self.num_envs, "obs")
        self._check_dict_values_length(info, self.num_envs, "info")

        return obs, reward, terminated, truncated, info

    def get_info(
        self,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def get_reward(
        self,
        action: torch.Tensor | list[Dict],
        env_ids: Sequence[int] | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def get_termination(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def get_obs_space(self) -> gym.spaces.Space:
        raise NotImplementedError

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        """
        Reset the environment.
        This should only be called once at the beginning of the environment.
        In this function, we will call scene.reset(soft=False) to load all the objects managed in scene manager
        It will also reset the reset count.
        """
        if seed is not None:
            seed_everywhere(seed)
        self.scene.reset(options=options)
        self.scene.sim.sim_step()
        self.episode_length_buf[:] = 0
        self.reset_terminated[:] = 0
        self.reset_truncated[:] = 0
        self.reset_buf[:] = 0
        self.last_action = None
        obs = self.get_obs()
        info = self.get_info()
        return obs, info

    def reset_idx(
        self,
        env_ids: Sequence[int] = None,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        """
        Reset specific environments.
        """
        if seed is not None:
            seed_everywhere(seed)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.scene.sim.device)
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(
                env_ids, device=self.scene.sim.device, dtype=torch.int32
            )

        print(f"Resetting env {env_ids} with seed {seed} and options {options}")
        self.scene.reset_idx(env_ids=env_ids, seed=seed, options=options)
        self.episode_length_buf[env_ids] = 0
        self.reset_terminated[env_ids] = 0
        self.reset_truncated[env_ids] = 0
        self.reset_buf[env_ids] = 0
        obs = self.get_obs()
        info = self.get_info()
        return obs, info

    def _pad_action_info_to_num_envs(
        self, action_info: Dict[str, Any], env_ids: torch.Tensor, num_envs: int
    ) -> Dict[str, Any]:
        """Pad action_info from len(env_ids) to num_envs length.

        Args:
            action_info: Dictionary containing action information for env_ids
            env_ids: Tensor of environment IDs that have actual data
            num_envs: Total number of environments

        Returns:
            Padded action_info with shape (num_envs, ...) for all tensors,
            with torch.nan used for padding.
        """

        if isinstance(env_ids, torch.Tensor):
            env_ids_list = env_ids.detach().cpu().tolist()
            if isinstance(env_ids_list, int):
                env_ids_list = [env_ids_list]
        else:
            env_ids_list = list(env_ids)

        padded_info = {}
        for key, value in action_info.items():
            if isinstance(value, torch.Tensor):
                if value.ndim == 0:
                    padded_tensor = torch.full(
                        (num_envs,), torch.nan, device=value.device, dtype=value.dtype
                    )
                    padded_tensor[env_ids_list] = value.expand(len(env_ids_list))
                else:
                    shape = list(value.shape)
                    shape[0] = num_envs
                    padded_tensor = torch.full(
                        shape, torch.nan, device=value.device, dtype=value.dtype
                    )
                    padded_tensor[env_ids_list] = value
                padded_info[key] = padded_tensor
            elif isinstance(value, dict):
                padded_info[key] = self._pad_action_info_to_num_envs(
                    value, env_ids, num_envs
                )
            else:
                padded_info[key] = value

        return padded_info

    def _pad_success_flags_to_num_envs(
        self, success_flags: torch.Tensor | None, env_ids: torch.Tensor, num_envs: int
    ) -> torch.Tensor | None:
        """Pad success_flags from len(env_ids) to num_envs length.

        Args:
            success_flags: Tensor of success flags for env_ids, or None
            env_ids: Tensor of environment IDs that have actual data
            num_envs: Total number of environments

        Returns:
            Padded success_flags tensor with shape (num_envs,), with torch.nan used for padding.
            Returns None if input is None.
        """
        if success_flags is None:
            return None

        if isinstance(env_ids, torch.Tensor):
            env_ids_list = env_ids.detach().cpu().tolist()
            if isinstance(env_ids_list, int):
                env_ids_list = [env_ids_list]
        else:
            env_ids_list = list(env_ids)

        if success_flags.dtype == torch.bool:
            padded_flags = torch.full(
                (num_envs,), torch.nan, device=success_flags.device, dtype=torch.float32
            )

            success_flags_float = success_flags.float()
            padded_flags[env_ids_list] = success_flags_float
        else:
            padded_flags = torch.full(
                (num_envs,),
                torch.nan,
                device=success_flags.device,
                dtype=success_flags.dtype,
            )
            padded_flags[env_ids_list] = success_flags

        return padded_flags

    def _check_dict_values_length(
        self, data: Dict[str, Any], expected_length: int, path: str = ""
    ) -> None:
        """Recursively check that all innermost values in a dictionary have the expected length.

        Args:
            data: Dictionary to check (can be nested)
            expected_length: Expected length (num_envs)
            path: Current path in the dictionary (for error messages)
        """
        if not isinstance(data, dict):
            return

        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key

            if isinstance(value, torch.Tensor):
                if value.ndim > 0:
                    actual_length = value.shape[0]
                    assert actual_length == expected_length, (
                        f"Value at path '{current_path}' has length {actual_length}, "
                        f"expected {expected_length}. Shape: {value.shape}"
                    )

            elif isinstance(value, dict):
                self._check_dict_values_length(value, expected_length, current_path)
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    if isinstance(item, torch.Tensor):
                        if item.ndim > 0:
                            actual_item_length = item.shape[0]
                            assert actual_item_length == expected_length, (
                                f"Value at path '{current_path}[{i}]' has length {actual_item_length}, "
                                f"expected {expected_length}. Shape: {item.shape}"
                            )
                    elif isinstance(item, dict):
                        self._check_dict_values_length(
                            item, expected_length, f"{current_path}[{i}]"
                        )
