from typing import Any, Dict, Sequence

import torch
import gymnasium as gym
from magicsim.StardardEnv.Camera.TaskCameraBaseEnv import TaskCameraBaseEnv


class BallTrainEnv(TaskCameraBaseEnv):
    """
    phys4d: Ball Train scene.
    get_done rolling_ball_1, static_ball_1, static_ball_2
    """

    def __init__(self, config, cli_args, logger):
        super().__init__(config, cli_args, logger)
        self._primary_camera_name: str | None = None

        self._prev_ball_positions: Dict[int, Dict[str, torch.Tensor]] = {}

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
        """Return per-environment physics state for recording.

        The returned ``state`` is a list with one entry per env, where each entry
        is a nested dict converted to pure Python types (no torch.Tensors), so it
        is compatible with TaskCameraBaseEnv._check_dict_values_length.
        """

        def _to_python(obj: Any) -> Any:
            """Recursively convert tensors to Python / list types."""
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
        """Reset all environments and clear previous ball positions."""
        obs, info = super().reset(seed=seed, options=options)

        self._prev_ball_positions.clear()
        return obs, info

    def reset_idx(
        self,
        env_ids: Sequence[int] = None,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        """Reset specific environments and clear their previous ball positions."""
        obs, info = super().reset_idx(env_ids=env_ids, seed=seed, options=options)

        if env_ids is not None:
            if not isinstance(env_ids, torch.Tensor):
                env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
            for env_id in env_ids:
                env_id_int = int(env_id)
                if env_id_int in self._prev_ball_positions:
                    del self._prev_ball_positions[env_id_int]
        else:
            self._prev_ball_positions.clear()
        return obs, info

    def _get_ball_position(
        self, ball, env_id_int: int, ball_name: str, debug: bool = False
    ) -> torch.Tensor | None:
        """Get ball position from its state."""
        try:
            ball_state = ball.get_state(is_relative=False)

            if "root_pose" not in ball_state:
                if debug and env_id_int == 0:
                    print(f"  env_0: {ball_name} root_pose not in state")
                return None

            root_pose = ball_state["root_pose"]

            if isinstance(root_pose, torch.Tensor):
                root_pose = root_pose.to(device=self.device, dtype=torch.float32)
            else:
                root_pose = torch.tensor(
                    root_pose, device=self.device, dtype=torch.float32
                )

            if root_pose.numel() >= 3:
                current_pos = root_pose[:3]
            else:
                current_pos = torch.zeros(3, device=self.device, dtype=torch.float32)
                current_pos[: root_pose.numel()] = root_pose

            if torch.isnan(current_pos).any():
                if debug and env_id_int == 0:
                    print(f"  env_0: {ball_name} root_pose contains nan")
                return None

            return current_pos
        except Exception as e:
            if debug and env_id_int == 0:
                print(f"  env_0: Error getting {ball_name} position: {e}")
            return None

    def get_done(
        self,
        env_ids: Sequence[int] | None = None,
        position_threshold: float = 0.001,
        debug: bool = False,
    ) -> torch.Tensor:
        """
        Check if all three balls (rolling_ball_1, static_ball_1, static_ball_2) have come to rest
        by comparing position changes between steps.

        Args:
            env_ids: Environment IDs to check. If None, checks all environments.
            position_threshold: Maximum position change (m) between steps to consider as "rested". Default: 0.001
            debug: If True, print debug information for env_0

        Returns:
            Tensor of shape (num_envs,) with True for environments where all three balls are rested, False otherwise.
        """
        if env_ids is None:
            env_ids = torch.arange(
                self.scene.num_envs, device=self.device, dtype=torch.long
            )
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        is_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        for env_id in env_ids:
            env_id_int = int(env_id)

            ball_positions = {}

            if (
                "rolling_ball" in self.scene.scene_manager._rigid_objects[env_id_int]
                and len(
                    self.scene.scene_manager._rigid_objects[env_id_int]["rolling_ball"]
                )
                > 0
            ):
                rolling_ball = self.scene.scene_manager._rigid_objects[env_id_int][
                    "rolling_ball"
                ][0]
                ball_pos = self._get_ball_position(
                    rolling_ball, env_id_int, "rolling_ball_1", debug
                )
                if ball_pos is not None:
                    ball_positions["rolling_ball_1"] = ball_pos

            if (
                "static_ball" in self.scene.scene_manager._rigid_objects[env_id_int]
                and len(
                    self.scene.scene_manager._rigid_objects[env_id_int]["static_ball"]
                )
                >= 2
            ):
                static_ball_1 = self.scene.scene_manager._rigid_objects[env_id_int][
                    "static_ball"
                ][0]
                static_ball_2 = self.scene.scene_manager._rigid_objects[env_id_int][
                    "static_ball"
                ][1]

                ball_pos_1 = self._get_ball_position(
                    static_ball_1, env_id_int, "static_ball_1", debug
                )
                if ball_pos_1 is not None:
                    ball_positions["static_ball_1"] = ball_pos_1

                ball_pos_2 = self._get_ball_position(
                    static_ball_2, env_id_int, "static_ball_2", debug
                )
                if ball_pos_2 is not None:
                    ball_positions["static_ball_2"] = ball_pos_2

            if len(ball_positions) != 3:
                is_done[env_id_int] = False
                if debug and env_id_int == 0:
                    print(f"  env_0: Found {len(ball_positions)} balls, expected 3")
                continue

            if env_id_int not in self._prev_ball_positions:
                self._prev_ball_positions[env_id_int] = {
                    name: pos.clone() for name, pos in ball_positions.items()
                }
                is_done[env_id_int] = False
                if debug and env_id_int == 0:
                    print(
                        f"  env_0: First step, saving positions for {len(ball_positions)} balls"
                    )
                continue

            prev_positions = self._prev_ball_positions[env_id_int]

            all_rested = True
            for ball_name in ball_positions.keys():
                if ball_name not in prev_positions:
                    all_rested = False
                    if debug and env_id_int == 0:
                        print(f"  env_0: {ball_name} not in previous positions")
                    break

                current_pos = ball_positions[ball_name]
                prev_pos = prev_positions[ball_name]

                pos_change = torch.norm(current_pos - prev_pos)

                if debug and env_id_int == 0:
                    print(
                        f"  env_0: {ball_name} pos_change={pos_change.item():.6f}, threshold={position_threshold}"
                    )

                if pos_change >= position_threshold:
                    all_rested = False
                    if debug and env_id_int == 0:
                        print(
                            f"  env_0: {ball_name} is still moving (pos_change={pos_change.item():.6f})"
                        )

            is_done[env_id_int] = all_rested

            self._prev_ball_positions[env_id_int] = {
                name: pos.clone() for name, pos in ball_positions.items()
            }

            if debug and env_id_int == 0:
                print(f"  env_0: is_done={is_done[env_id_int].item()}")

        return is_done
