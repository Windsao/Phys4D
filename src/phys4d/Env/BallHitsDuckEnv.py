from typing import Any, Dict, Sequence

import torch
import gymnasium as gym
from magicsim.StardardEnv.Camera.TaskCameraBaseEnv import TaskCameraBaseEnv


class BallHitsDuckEnv(TaskCameraBaseEnv):
    """
    phys4d: Ball Hits Duck scene.
    get_info  BallAndBlockEnv
    get_done  rolling_ball_1  duck_block_1
    """

    def __init__(self, config, cli_args, logger):
        super().__init__(config, cli_args, logger)
        self._primary_camera_name: str | None = None

        self._prev_obj_positions: Dict[int, Dict[str, torch.Tensor]] = {}

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
        """BallAndBlockEnv ."""

        def _to_python(obj: Any) -> Any:
            """tensor  Python / list ."""
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

        all_env_states: list[dict[str, Any]] = []
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

    def get_done(
        self,
        env_ids: Sequence[int] | None = None,
        position_threshold: float = 0.001,
        debug: bool = False,
    ) -> torch.Tensor:
        """
        rolling_ball_1  duck_block_1
        BallAndBlockEnv.get_done
        """
        if env_ids is None:
            env_ids = torch.arange(
                self.scene.num_envs, device=self.device, dtype=torch.long
            )
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        is_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        target_objects = ["rolling_ball", "duck_block"]

        for env_id in env_ids:
            env_id_int = int(env_id)

            all_static = True

            for obj_name in target_objects:
                try:
                    obj_list = self.scene.scene_manager._rigid_objects[env_id_int].get(
                        obj_name, []
                    )
                    if not obj_list:
                        all_static = False
                        if debug and env_id_int == 0:
                            print(f"  env_0: rigid object '{obj_name}' not found")
                        continue

                    obj = obj_list[0]
                    state = obj.get_state(is_relative=False)

                    if "root_pose" not in state:
                        all_static = False
                        if debug and env_id_int == 0:
                            print(f"  env_0: root_pose not in state for '{obj_name}'")
                        continue

                    root_pose = state["root_pose"]

                    if isinstance(root_pose, torch.Tensor):
                        root_pose = root_pose.to(
                            device=self.device, dtype=torch.float32
                        )
                    else:
                        root_pose = torch.tensor(
                            root_pose, device=self.device, dtype=torch.float32
                        )

                    if root_pose.numel() >= 3:
                        current_pos = root_pose[:3]
                    else:
                        current_pos = torch.zeros(
                            3, device=self.device, dtype=torch.float32
                        )
                        current_pos[: root_pose.numel()] = root_pose

                    if torch.isnan(current_pos).any():
                        all_static = False
                        if debug and env_id_int == 0:
                            print(f"  env_0: root_pose NaN for '{obj_name}'")
                        continue

                    if env_id_int not in self._prev_obj_positions:
                        self._prev_obj_positions[env_id_int] = {}

                    if obj_name in self._prev_obj_positions[env_id_int]:
                        prev_pos = self._prev_obj_positions[env_id_int][obj_name]
                        pos_change = torch.norm(current_pos - prev_pos)

                        if debug and env_id_int == 0:
                            print(
                                f"  env_0 [{obj_name}]: pos_change={pos_change.item():.6f}, "
                                f"threshold={position_threshold}"
                            )

                        if pos_change >= position_threshold:
                            all_static = False
                    else:
                        all_static = False

                    self._prev_obj_positions[env_id_int][obj_name] = current_pos.clone()

                except Exception as e:
                    all_static = False
                    if debug and env_id_int == 0:
                        print(f"  env_0: error in get_done for '{obj_name}': {e}")

            is_done[env_id_int] = all_static

        return is_done
