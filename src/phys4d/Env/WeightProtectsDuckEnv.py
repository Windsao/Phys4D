from typing import Any, Dict, Sequence

import torch
import gymnasium as gym
from magicsim.StardardEnv.Camera.TaskCameraBaseEnv import TaskCameraBaseEnv


class WeightProtectsDuckEnv(TaskCameraBaseEnv):
    """
    Weight Protects Duck get_done  step>=50

    -  TaskCameraBaseEnv
    - get_done sim.step  50  True
    """

    def __init__(self, config, cli_args, logger):
        super().__init__(config, cli_args, logger)
        self._primary_camera_name: str | None = None

        self._step_counts: torch.Tensor | None = None

    def get_obs_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({})

    def get_policy_obs(self, env_ids: Sequence[int] | None = None) -> Dict[str, Any]:
        if env_ids is None:
            env_ids = torch.arange(self.scene.num_envs, device=self.device)
        camera_info = self.scene.capture_manager.step(env_ids=env_ids)
        return {
            "camera_info": camera_info,
        }

    def get_privilege_obs(self, env_ids: Sequence[int] | None = None) -> Dict[str, Any]:
        return {}

    def process_camera_action(
        self,
        camera_action: Any | None,
        env_ids: Sequence[int] | None = None,
    ) -> Dict[str, Any] | None:
        return camera_action

    def _get_primary_camera_name(self) -> str:
        if self._primary_camera_name is None:
            camera_names = list(self.scene.camera_manager.camera_config.keys())
            if not camera_names:
                raise RuntimeError("No camera configured in camera_manager.")
            self._primary_camera_name = camera_names[0]
        return self._primary_camera_name

    def get_info(self, env_ids: Sequence[int] | None = None) -> Dict[str, Any]:
        """env"""

        def _to_python(obj: Any) -> Any:
            if isinstance(obj, torch.Tensor):
                if obj.ndim == 0:
                    return obj.item()
                return obj.detach().cpu().tolist()
            if isinstance(obj, dict):
                return {k: _to_python(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_to_python(v) for v in obj]
            return obj

        if env_ids is None:
            env_ids = torch.arange(
                self.scene.num_envs, device=self.device, dtype=torch.long
            )
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        all_env_states = []
        for env_id in env_ids:
            env_id_int = int(env_id)
            state = self.scene.scene_manager.get_state(
                is_relative=True,
                env_ids=[env_id_int],
            )
            all_env_states.append(_to_python(state))

        return {"state": all_env_states}

    def get_reward(
        self,
        camera_action: Dict[str, torch.Tensor] | None,
        env_ids: Sequence[int] | None = None,
    ) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def get_termination(
        self,
        env_ids: Sequence[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if env_ids is None:
            env_ids = torch.arange(
                self.scene.num_envs, device=self.device, dtype=torch.long
            )
        termination = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = torch.zeros_like(termination)
        return termination, truncated

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        """Reset all environments and clear step counts."""
        obs, info = super().reset(seed=seed, options=options)

        if self._step_counts is None or self._step_counts.numel() != self.num_envs:
            self._step_counts = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
        else:
            self._step_counts.zero_()
        return obs, info

    def reset_idx(
        self,
        env_ids: Sequence[int] = None,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        """Reset specific environments and clear their step counts."""
        obs, info = super().reset_idx(env_ids=env_ids, seed=seed, options=options)
        if self._step_counts is None or self._step_counts.numel() != self.num_envs:
            self._step_counts = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
        else:
            if env_ids is not None:
                if not isinstance(env_ids, torch.Tensor):
                    env_ids = torch.tensor(
                        env_ids, device=self.device, dtype=torch.long
                    )
                self._step_counts[env_ids] = 0
            else:
                self._step_counts.zero_()

        return obs, info

    def get_done(
        self,
        env_ids: Sequence[int] | None = None,
        position_threshold: float = 0.001,
        debug: bool = False,
    ) -> torch.Tensor:
        """
        get_done sim.step  50

        -  sim.step  env  +1
        -  env  >= 50  True env
        """
        if env_ids is None:
            env_ids = torch.arange(
                self.scene.num_envs, device=self.device, dtype=torch.long
            )
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        if self._step_counts is None or self._step_counts.numel() != self.num_envs:
            self._step_counts = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )

        self._step_counts[env_ids] += 1

        is_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        done_mask = self._step_counts[env_ids] >= 50
        is_done[env_ids] = done_mask

        if done_mask.any():
            done_env_ids = env_ids[done_mask]
            self._step_counts[done_env_ids] = 0

        if debug:
            for env_id in env_ids:
                env_id_int = int(env_id)
                if env_id_int == 0:
                    print(
                        f"  env_0: step_count={self._step_counts[env_id_int].item()}, is_done={is_done[env_id_int].item()}"
                    )

        return is_done
