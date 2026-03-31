




from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg




from isaaclab_assets.robots.anymal import ANYMAL_D_CFG


@configclass
class AnymalDRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):

        super().__post_init__()

        self.scene.robot = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class AnymalDRoughEnvCfg_PLAY(AnymalDRoughEnvCfg):
    def __post_init__(self):

        super().__post_init__()


        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        self.scene.terrain.max_init_terrain_level = None

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False


        self.observations.policy.enable_corruption = False

        self.events.base_external_force_torque = None
        self.events.push_robot = None
