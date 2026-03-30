from typing import Any, Dict, Sequence

import math
import torch
import gymnasium as gym
from magicsim.StardardEnv.Camera.TaskCameraBaseEnv import TaskCameraBaseEnv


class MirrorTeapotRotateEnv(TaskCameraBaseEnv):
    """
    Mirror Teapot Rotate  cylinder + teapot_on_cylinderget_done  step>=100

    -  TaskCameraBaseEnv
    -  _init_cylinder_controllers TeapotSpinController  rotating_cylinder + teapot_on_cylinder
    - get_done sim.step  100  True
    """

    def __init__(self, config, cli_args, logger):
        super().__init__(config, cli_args, logger)
        self._primary_camera_name: str | None = None

        self._step_counts: torch.Tensor | None = None

        self._cylinder_controllers: Dict[int, "_TeapotSpinController"] = {}

        self._controller_config = config

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

    def _init_cylinder_controllers(self):
        """env rotating_cylinder + teapot_on_cylinder"""
        angular_speed = 4.0
        self._cylinder_controllers.clear()

        for env_id in range(self.num_envs):
            objects_cylinder = self.scene.scene_manager.get_objects(
                env_ids=[env_id], object_name="rotating_cylinder", object_type="rigid"
            )
            cylinder = None
            for obj_list in objects_cylinder.values():
                if obj_list:
                    cylinder = obj_list[0]
                    break

            objects_teapot = self.scene.scene_manager.get_objects(
                env_ids=[env_id], object_name="teapot_on_cylinder", object_type="rigid"
            )
            teapot = None
            for obj_list in objects_teapot.values():
                if obj_list:
                    teapot = obj_list[0]
                    break

            if cylinder is None:
                continue

            translation, _ = cylinder.get_local_pose()
            if isinstance(translation, torch.Tensor):
                pivot = translation.detach().clone().to(dtype=torch.float32)
            else:
                pivot = torch.tensor(translation, dtype=torch.float32)
            initial_angle_deg = 0.0

            self._cylinder_controllers[env_id] = _TeapotSpinController(
                cylinder=cylinder,
                teapot=teapot,
                angular_speed=angular_speed,
                pivot_translation=pivot,
                initial_angle_deg=initial_angle_deg,
            )

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        """env step"""
        obs, info = super().reset(seed=seed, options=options)

        self._step_counts = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        if not self._cylinder_controllers:
            self._init_cylinder_controllers()

        for controller in self._cylinder_controllers.values():
            controller.angle = 0.0
            controller.apply_current_pose()

        return obs, info

    def reset_idx(
        self,
        env_ids: Sequence[int] = None,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        """env step"""
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

        if not self._cylinder_controllers:
            self._init_cylinder_controllers()

        if env_ids is not None:
            if not isinstance(env_ids, torch.Tensor):
                env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
            for env_id in env_ids:
                env_id_int = int(env_id)
                if env_id_int in self._cylinder_controllers:
                    ctrl = self._cylinder_controllers[env_id_int]
                    ctrl.angle = 0.0
                    ctrl.apply_current_pose()
        else:
            for ctrl in self._cylinder_controllers.values():
                ctrl.angle = 0.0
                ctrl.apply_current_pose()

        return obs, info

    def step(
        self,
        camera_action: Dict[str, torch.Tensor] | None = None,
        env_ids: Sequence[int] | None = None,
        failed_env_ids: Sequence[int] | None = None,
    ):
        """step  rotating_cylinder"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device)

        physics_dt = float(getattr(self.scene.sim, "physics_dt", 1.0 / 60.0))

        for env_id in env_ids:
            env_id_int = int(env_id)
            controller = self._cylinder_controllers.get(env_id_int, None)
            if controller is not None:
                controller.step(physics_dt)

        return super().step(
            camera_action=camera_action, env_ids=env_ids, failed_env_ids=failed_env_ids
        )

    def get_done(
        self,
        env_ids: Sequence[int] | None = None,
        position_threshold: float = 0.001,
        debug: bool = False,
    ) -> torch.Tensor:
        """
        get_done sim.step  100

        -  sim.step  env  +1
        -  env  >= 100  True env
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
        done_mask = self._step_counts[env_ids] >= 100
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


class _TeapotSpinController:
    """rotating_cylinder  teapot_on_cylinder"""

    def __init__(
        self,
        cylinder,
        teapot,
        angular_speed: float = 4.0,
        pivot_translation: torch.Tensor | None = None,
        initial_angle_deg: float = 0.0,
    ):
        self.cylinder = cylinder
        self.teapot = teapot
        self.angular_speed = angular_speed
        self.angle = 0.0

        translation, orientation = cylinder.get_local_pose()
        if pivot_translation is not None:
            if isinstance(pivot_translation, torch.Tensor):
                self.base_translation = (
                    pivot_translation.detach().clone().to(dtype=torch.float32)
                )
            else:
                self.base_translation = torch.tensor(
                    pivot_translation, dtype=torch.float32
                )
        else:
            if isinstance(translation, torch.Tensor):
                self.base_translation = (
                    translation.detach().clone().to(dtype=torch.float32)
                )
            else:
                self.base_translation = torch.tensor(translation, dtype=torch.float32)

        if isinstance(orientation, torch.Tensor):
            self.base_orientation = orientation.detach().clone().to(dtype=torch.float32)
        else:
            self.base_orientation = torch.tensor(orientation, dtype=torch.float32)

        self.initial_angle = float(initial_angle_deg) * math.pi / 180.0

        if teapot is not None:
            teapot_translation, teapot_orientation = teapot.get_local_pose()
            if isinstance(teapot_translation, torch.Tensor):
                offset = teapot_translation.detach().clone().to(dtype=torch.float32)
            else:
                offset = torch.tensor(teapot_translation, dtype=torch.float32)
            self.teapot_offset = offset - self.base_translation

            if isinstance(teapot_orientation, torch.Tensor):
                base_ori = teapot_orientation.detach().clone().to(dtype=torch.float32)
            else:
                base_ori = torch.tensor(teapot_orientation, dtype=torch.float32)
            self.teapot_base_orientation = base_ori
        else:
            self.teapot_offset = None
            self.teapot_base_orientation = None

        self.apply_current_pose()

    def step(self, dt: float):
        if self.cylinder is None:
            return
        self.angle += self.angular_speed * float(dt)
        self.apply_current_pose()

    def _yaw_to_quat(self, angle: float) -> torch.Tensor:
        half = float(angle) * 0.5
        return torch.tensor(
            [math.cos(half), 0.0, 0.0, math.sin(half)], dtype=torch.float32
        )

    def _quat_mul(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return torch.tensor(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=torch.float32,
        )

    def apply_current_pose(self):
        rotation_quat = self._yaw_to_quat(self.initial_angle + self.angle)
        new_orientation = self._quat_mul(rotation_quat, self.base_orientation)
        self.cylinder.set_local_pose(
            translation=self.base_translation, orientation=new_orientation
        )
        self.cylinder.set_linear_velocity(torch.zeros(3))
        self.cylinder.set_angular_velocity(torch.zeros(3))

        if (
            self.teapot is None
            or self.teapot_offset is None
            or self.teapot_base_orientation is None
        ):
            return

        angle_rad = self.initial_angle + self.angle
        cos_a = math.cos(float(angle_rad))
        sin_a = math.sin(float(angle_rad))
        offset = self.teapot_offset
        rotated_offset = torch.tensor(
            [
                offset[0] * cos_a - offset[1] * sin_a,
                offset[0] * sin_a + offset[1] * cos_a,
                offset[2],
            ],
            dtype=torch.float32,
        )

        new_translation = self.base_translation + rotated_offset
        new_teapot_orientation = self._quat_mul(
            rotation_quat, self.teapot_base_orientation
        )
        self.teapot.set_local_pose(
            translation=new_translation, orientation=new_teapot_orientation
        )
        self.teapot.set_linear_velocity(torch.zeros(3))
        self.teapot.set_angular_velocity(torch.zeros(3))
