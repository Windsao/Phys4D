




from isaaclab.utils import configclass

from .rough_env_cfg import DigitRoughEnvCfg


@configclass
class DigitFlatEnvCfg(DigitRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()


        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        self.scene.height_scanner = None
        self.observations.policy.height_scan = None

        self.curriculum.terrain_levels = None


class DigitFlatEnvCfg_PLAY(DigitFlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()


        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        self.observations.policy.enable_corruption = False

        self.events.base_external_force_torque = None
        self.events.push_robot = None
