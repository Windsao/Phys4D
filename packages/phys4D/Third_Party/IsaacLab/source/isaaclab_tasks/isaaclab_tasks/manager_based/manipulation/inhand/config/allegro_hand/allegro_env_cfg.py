




from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.inhand.inhand_env_cfg as inhand_env_cfg




from isaaclab_assets import ALLEGRO_HAND_CFG


@configclass
class AllegroCubeEnvCfg(inhand_env_cfg.InHandObjectEnvCfg):
    def __post_init__(self):

        super().__post_init__()


        self.scene.robot = ALLEGRO_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.clone_in_fabric = True


@configclass
class AllegroCubeEnvCfg_PLAY(AllegroCubeEnvCfg):
    def __post_init__(self):

        super().__post_init__()

        self.scene.num_envs = 50

        self.observations.policy.enable_corruption = False

        self.terminations.time_out = None







@configclass
class AllegroCubeNoVelObsEnvCfg(AllegroCubeEnvCfg):
    def __post_init__(self):

        super().__post_init__()


        self.observations.policy = inhand_env_cfg.ObservationsCfg.NoVelocityKinematicObsGroupCfg()


@configclass
class AllegroCubeNoVelObsEnvCfg_PLAY(AllegroCubeNoVelObsEnvCfg):
    def __post_init__(self):

        super().__post_init__()

        self.scene.num_envs = 50

        self.observations.policy.enable_corruption = False

        self.terminations.time_out = None
