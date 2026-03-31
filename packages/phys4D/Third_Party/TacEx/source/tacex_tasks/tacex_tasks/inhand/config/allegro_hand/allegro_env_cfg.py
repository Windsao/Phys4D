




from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass

from tacex_assets import GELSIGHT_MINI_TAXIM_FOTS_CFG





from tacex_assets.robots.allegro_gsmini import ALLEGRO_HAND_GSMINI_CFG


from ...inhand_env_cfg import InHandObjectEnvCfg, ObservationsCfg


@configclass
class AllegroCubeEnvCfg(InHandObjectEnvCfg):
    def __post_init__(self):

        super().__post_init__()


        self.scene.num_envs = 50


        self.scene.robot = ALLEGRO_HAND_GSMINI_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


        self.scene.gsmini_ring = GELSIGHT_MINI_TAXIM_FOTS_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot/Case_ring_3",
            debug_vis=True,

            marker_motion_sim_cfg=GELSIGHT_MINI_TAXIM_FOTS_CFG.marker_motion_sim_cfg.replace(
                frame_transformer_cfg=FrameTransformerCfg(
                    prim_path="/World/envs/env_.*/Robot/Gelpad_ring_3",
                    target_frames=[FrameTransformerCfg.FrameCfg(prim_path="/World/envs/env_.*/object")],
                    debug_vis=False,
                ),
            ),
        )
        self.scene.gsmini_middle = GELSIGHT_MINI_TAXIM_FOTS_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot/Case_middle_3",
            debug_vis=True,

            marker_motion_sim_cfg=GELSIGHT_MINI_TAXIM_FOTS_CFG.marker_motion_sim_cfg.replace(
                frame_transformer_cfg=FrameTransformerCfg(
                    prim_path="/World/envs/env_.*/Robot/Gelpad_middle_3",
                    target_frames=[FrameTransformerCfg.FrameCfg(prim_path="/World/envs/env_.*/object")],
                    debug_vis=False,
                ),
            ),
        )
        self.scene.gsmini_index = GELSIGHT_MINI_TAXIM_FOTS_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot/Case_index_3",
            debug_vis=True,

            marker_motion_sim_cfg=GELSIGHT_MINI_TAXIM_FOTS_CFG.marker_motion_sim_cfg.replace(
                frame_transformer_cfg=FrameTransformerCfg(
                    prim_path="/World/envs/env_.*/Robot/Gelpad_index_3",
                    target_frames=[FrameTransformerCfg.FrameCfg(prim_path="/World/envs/env_.*/object")],
                    debug_vis=False,
                ),
            ),
        )
        self.scene.gsmini_thumb = GELSIGHT_MINI_TAXIM_FOTS_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot/Case_thumb_3",
            debug_vis=True,

            marker_motion_sim_cfg=GELSIGHT_MINI_TAXIM_FOTS_CFG.marker_motion_sim_cfg.replace(
                frame_transformer_cfg=FrameTransformerCfg(
                    prim_path="/World/envs/env_.*/Robot/Gelpad_thumb_3",
                    target_frames=[FrameTransformerCfg.FrameCfg(prim_path="/World/envs/env_.*/object")],
                    debug_vis=False,
                ),
            ),
        )


        self.observations.policy = ObservationsCfg.TactileObsGroupCfg()


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


        self.observations.policy = ObservationsCfg.NoVelocityKinematicObsGroupCfg()


@configclass
class AllegroCubeNoVelObsEnvCfg_PLAY(AllegroCubeNoVelObsEnvCfg):
    def __post_init__(self):

        super().__post_init__()

        self.scene.num_envs = 50

        self.observations.policy.enable_corruption = False

        self.terminations.time_out = None
