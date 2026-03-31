from typing import Any, Dict

import torch

from magicsim.Collect.CameraAtomicSkill.CameraAtomicSkill import CameraAtomicSkill
from magicsim.Env.Utils.rotations import euler_angles_to_quat


class GoTo(CameraAtomicSkill):
    """Camera atomic skill that moves a camera to a target pose relative to a scene object
    or along a pre-defined absolute pose list (orbit_poses)."""

    def __init__(self, config, env, env_id, logger):
        super().__init__(config, env, env_id, logger)
        self.obj_type: str | None = None
        self.obj_name: str | None = None
        self.obj_id: int | None = None

        self.position_offset = torch.tensor(
            getattr(config, "position_offset", [0.0, 0.0, 0.2]), dtype=torch.float32
        )
        self.position_threshold = float(getattr(config, "position_threshold", 0.02))

        self.use_absolute = bool(getattr(config, "use_absolute", False))

        self.orbit_poses = getattr(config, "orbit_poses", None)
        self.orbit_idx: int = 0

        if hasattr(config, "absolute_orientation"):
            orientation_euler = torch.tensor(
                getattr(config, "absolute_orientation", [0.0, -45.0, 0.0]),
                dtype=torch.float32,
            )
        else:
            orientation_euler = torch.tensor(
                getattr(config, "orientation", [0.0, -45.0, 0.0]),
                dtype=torch.float32,
            )
        self.orientation_quat = euler_angles_to_quat(
            orientation_euler, degrees=True
        ).reshape(4)

        self.absolute_pos = torch.tensor(
            getattr(config, "absolute_pos", [0.0, 0.0, 0.0]), dtype=torch.float32
        )
        self.current_target_pose: Dict[str, torch.Tensor] | None = None

    def reset(self, camera_name: str, obj_type: str, obj_name: str, obj_id: int):
        self.camera_name = camera_name
        self.obj_type = obj_type
        self.obj_name = obj_name
        self.obj_id = obj_id
        self.current_state = "ready"
        self.current_command = ["GoTo", camera_name, obj_type, obj_name, obj_id]
        self.current_target_pose = self._compute_target_pose()
        return {"state": self.current_state, "target_pose": self.current_target_pose}

    def refresh(self, camera_name: str, obj_type: str, obj_name: str, obj_id: int):
        """Update target object / camera configuration while the skill is running.

        Semantics are aligned with robot Reach.refresh: only updates command and target,
        without resetting the state.
        """
        self.camera_name = camera_name
        self.obj_type = obj_type
        self.obj_name = obj_name
        self.obj_id = obj_id
        self.current_command = ["GoTo", camera_name, obj_type, obj_name, obj_id]
        self.current_target_pose = self._compute_target_pose()
        return {"state": self.current_state, "target_pose": self.current_target_pose}

    def update_config(
        self,
        position_offset: torch.Tensor | None = None,
        orientation: torch.Tensor | None = None,
        absolute_pos: torch.Tensor | None = None,
        use_absolute: bool | None = None,
    ):
        """Update position offset and/or orientation dynamically.

        Args:
            position_offset: New position offset relative to object (if not absolute mode)
            orientation: New orientation as Euler angles in degrees
            absolute_pos: New absolute position (if use_absolute=True)
            use_absolute: Whether to use absolute positioning
        """
        if position_offset is not None:
            if not isinstance(position_offset, torch.Tensor):
                position_offset = torch.tensor(position_offset, dtype=torch.float32)
            self.position_offset = position_offset

        if orientation is not None:
            if not isinstance(orientation, torch.Tensor):
                orientation = torch.tensor(orientation, dtype=torch.float32)
            self.orientation_quat = euler_angles_to_quat(
                orientation, degrees=True
            ).reshape(4)

        if absolute_pos is not None:
            if not isinstance(absolute_pos, torch.Tensor):
                absolute_pos = torch.tensor(absolute_pos, dtype=torch.float32)
            self.absolute_pos = absolute_pos

        if use_absolute is not None:
            self.use_absolute = use_absolute

        self.current_target_pose = self._compute_target_pose()

    def _compute_target_pose(self):
        if self.orbit_poses is not None and len(self.orbit_poses) > 0:
            pose_cfg = self.orbit_poses[self.orbit_idx]
            self.orbit_idx = (self.orbit_idx + 1) % len(self.orbit_poses)

            pos_cfg = pose_cfg["pos"]
            ori_cfg = pose_cfg["ori"]
            pos = torch.tensor(pos_cfg, dtype=torch.float32, device=self.env.device)
            euler = torch.tensor(ori_cfg, dtype=torch.float32, device=self.env.device)
            quat = euler_angles_to_quat(euler, degrees=True).reshape(4)

            return {"pos": pos, "quat": quat}

        if self.use_absolute:
            target_pos = self.absolute_pos.to(self.env.device)
        else:
            target_obj = self.env.scene.scene_manager.get_category(self.obj_type)[
                self.env_id
            ][self.obj_name][self.obj_id]
            obj_pos, _ = target_obj.get_local_pose()
            offset = self.position_offset.to(obj_pos.device)

            target_pos = obj_pos + offset
        return {
            "pos": target_pos,
            "quat": self.orientation_quat.to(target_pos.device),
        }

    def step(self):
        self.current_target_pose = self._compute_target_pose()
        self.current_state = "running"
        target_tensor = torch.cat(
            [
                self.current_target_pose["pos"],
                self.current_target_pose["quat"],
            ],
            dim=0,
        )
        self.current_action = {
            "NavTo": {
                "camera_name": self.camera_name,
                "target_pose": target_tensor,
            }
        }
        return self.current_action

    def update(self, info: Dict[str, Any]):
        if self.orbit_poses is not None and len(self.orbit_poses) > 0:
            self.current_state = "running"
            return {
                "type": "GoTo",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 0,
            }

        camera_state = self.env.scene.camera_manager.get_camera_state(
            camera_name=self.camera_name, env_ids=[int(self.env_id)]
        )
        current_pos = camera_state["pos"][0]
        target_pos = self.current_target_pose["pos"]
        distance = torch.norm(current_pos - target_pos)

        if distance < self.position_threshold:
            self.current_state = "finished"

            return {
                "type": "GoTo",
                "command": self.current_command,
                "action": self.current_action,
                "finished": True,
                "state": self.current_state,
                "truncated": 0,
            }

        self.current_state = "running"
        return {
            "type": "GoTo",
            "command": self.current_command,
            "action": self.current_action,
            "finished": False,
            "state": self.current_state,
            "truncated": 0,
        }
