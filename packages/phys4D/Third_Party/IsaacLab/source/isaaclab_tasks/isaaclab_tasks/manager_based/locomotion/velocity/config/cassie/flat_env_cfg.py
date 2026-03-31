




from isaaclab.utils import configclass

from .rough_env_cfg import CassieRoughEnvCfg


@configclass
class CassieFlatEnvCfg(CassieRoughEnvCfg):
    def __post_init__(self):

        super().__post_init__()

        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 5.0
        self.rewards.joint_deviation_hip.params["asset_cfg"].joint_names = ["hip_rotation_.*"]

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        self.scene.height_scanner = None
        self.observations.policy.height_scan = None

        self.curriculum.terrain_levels = None


class CassieFlatEnvCfg_PLAY(CassieFlatEnvCfg):
    def __post_init__(self) -> None:

        super().__post_init__()


        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        self.observations.policy.enable_corruption = False
