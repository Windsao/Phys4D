





from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.place.config.agibot.place_upright_mug_rmp_rel_env_cfg import (
    RmpFlowAgibotPlaceUprightMugEnvCfg,
)


@configclass
class RmpFlowAgibotPlaceUprightMugMimicEnvCfg(RmpFlowAgibotPlaceUprightMugEnvCfg, MimicEnvCfg):
    """
    Isaac Lab Mimic environment config class for Agibot Place Upright Mug env.
    """

    def __post_init__(self):

        super().__post_init__()

        self.datagen_config.name = "demo_src_place_upright_mug_task_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1


        subtask_configs = []
        subtask_configs.append(
            SubTaskConfig(

                object_ref="mug",

                subtask_term_signal="grasp",

                subtask_term_offset_range=(15, 30),

                selection_strategy="nearest_neighbor_object",



                action_noise=0.01,

                num_interpolation_steps=5,

                num_fixed_steps=0,

                apply_noise_during_interpolation=False,
            )
        )
        subtask_configs.append(
            SubTaskConfig(

                object_ref="mug",

                subtask_term_signal=None,

                subtask_term_offset_range=(0, 0),

                selection_strategy="nearest_neighbor_object",



                action_noise=0.01,

                num_interpolation_steps=15,

                num_fixed_steps=0,

                apply_noise_during_interpolation=False,
            )
        )
        self.subtask_configs["agibot"] = subtask_configs
