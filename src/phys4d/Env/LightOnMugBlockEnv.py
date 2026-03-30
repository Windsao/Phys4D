import torch
from phys4d.Env.LightOnBlockEnv import LightOnBlockEnv


class LightOnMugBlockEnv(LightOnBlockEnv):
    """
    phys4d: Light On Mug Block  MagicSim

    -  phys4d  TestLightOnMugBlockEnv.MultIObjectSpinController
       rotating_cylinder  mug_on_cylinder  cube_on_cylinder  Z
    - get_done  LightOnBlockEnv sim.step  100  True
    """

    def __init__(self, config, cli_args, logger):
        super().__init__(config, cli_args, logger)

    def _init_cylinder_controllers(self):
        """env rotating_cylinder + mug_on_cylinder + cube_on_cylinder"""

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

            objects_mug = self.scene.scene_manager.get_objects(
                env_ids=[env_id], object_name="mug_on_cylinder", object_type="rigid"
            )
            mug = None
            for obj_list in objects_mug.values():
                if obj_list:
                    mug = obj_list[0]
                    break

            objects_cube = self.scene.scene_manager.get_objects(
                env_ids=[env_id], object_name="cube_on_cylinder", object_type="rigid"
            )
            cube = None
            for obj_list in objects_cube.values():
                if obj_list:
                    cube = obj_list[0]
                    break

            if cylinder is None:
                continue

            translation, _ = cylinder.get_local_pose()

            if isinstance(translation, torch.Tensor):
                pivot = translation.detach().clone().to(dtype=torch.float32)
            else:
                pivot = torch.tensor(translation, dtype=torch.float32)
            initial_angle_deg = 0.0

            followers = [
                {"object": mug},
                {"object": cube},
            ]
            self._cylinder_controllers[env_id] = _MultiObjectSpinController(
                cylinder=cylinder,
                followers=followers,
                angular_speed=angular_speed,
                pivot_translation=pivot,
                initial_angle_deg=initial_angle_deg,
            )


class _MultiObjectSpinController:
    """rotating_cylinder “”mugcube"""

    def __init__(
        self,
        cylinder,
        followers: list[dict],
        angular_speed: float = 4.0,
        pivot_translation: torch.Tensor | None = None,
        initial_angle_deg: float = 0.0,
    ):
        self.cylinder = cylinder
        self.followers = followers
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

        self.initial_angle = float(initial_angle_deg) * torch.pi / 180.0

        for follower in self.followers:
            obj = follower.get("object", None)
            if obj is None:
                follower["offset"] = None
                follower["base_orientation"] = None
                continue
            obj_translation, obj_orientation = obj.get_local_pose()
            if isinstance(obj_translation, torch.Tensor):
                offset = obj_translation.detach().clone().to(dtype=torch.float32)
            else:
                offset = torch.tensor(obj_translation, dtype=torch.float32)
            follower["offset"] = offset - self.base_translation

            if isinstance(obj_orientation, torch.Tensor):
                base_ori = obj_orientation.detach().clone().to(dtype=torch.float32)
            else:
                base_ori = torch.tensor(obj_orientation, dtype=torch.float32)
            follower["base_orientation"] = base_ori

        self.apply_current_pose()

    def step(self, dt: float):
        if self.cylinder is None:
            return
        self.angle += self.angular_speed * float(dt)
        self.apply_current_pose()

    def _yaw_to_quat(self, angle: float) -> torch.Tensor:
        half = float(angle) * 0.5

        import math

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

        for follower in self.followers:
            obj = follower.get("object", None)
            offset = follower.get("offset", None)
            base_ori = follower.get("base_orientation", None)
            if obj is None or offset is None or base_ori is None:
                continue

            angle_rad = self.initial_angle + self.angle
            cos_a = torch.cos(torch.tensor(angle_rad))
            sin_a = torch.sin(torch.tensor(angle_rad))
            rotated_offset = torch.tensor(
                [
                    offset[0] * cos_a - offset[1] * sin_a,
                    offset[0] * sin_a + offset[1] * cos_a,
                    offset[2],
                ],
                dtype=torch.float32,
            )

            new_translation = self.base_translation + rotated_offset
            new_obj_orientation = self._quat_mul(rotation_quat, base_ori)
            obj.set_local_pose(
                translation=new_translation, orientation=new_obj_orientation
            )
            obj.set_linear_velocity(torch.zeros(3))
            obj.set_angular_velocity(torch.zeros(3))
