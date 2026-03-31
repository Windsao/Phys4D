




from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.pick_place.nutpour_gr1t2_pink_ik_env_cfg import NutPourGR1T2PinkIKEnvCfg


@configclass
class NutPourGR1T2MimicEnvCfg(NutPourGR1T2PinkIKEnvCfg, MimicEnvCfg):

    def __post_init__(self):

        super().__post_init__()


        self.datagen_config.name = "gr1t2_nut_pouring_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = False
        self.datagen_config.generation_num_trials = 1000
        self.datagen_config.generation_select_src_per_subtask = False
        self.datagen_config.generation_select_src_per_arm = False
        self.datagen_config.generation_relative = False
        self.datagen_config.generation_joint_pos = False
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.num_demo_to_render = 10
        self.datagen_config.num_fail_demo_to_render = 25
        self.datagen_config.seed = 10


        subtask_configs = []
        subtask_configs.append(
            SubTaskConfig(

                object_ref="sorting_bowl",


                subtask_term_signal="idle_right",
                first_subtask_start_offset_range=(0, 0),

                subtask_term_offset_range=(0, 0),

                selection_strategy="nearest_neighbor_object",

                selection_strategy_kwargs={"nn_k": 3},

                action_noise=0.003,

                num_interpolation_steps=5,

                num_fixed_steps=0,

                apply_noise_during_interpolation=False,
            )
        )
        subtask_configs.append(
            SubTaskConfig(

                object_ref="sorting_bowl",


                subtask_term_signal="grasp_right",
                first_subtask_start_offset_range=(0, 0),

                subtask_term_offset_range=(0, 0),

                selection_strategy="nearest_neighbor_object",

                selection_strategy_kwargs={"nn_k": 3},

                action_noise=0.003,

                num_interpolation_steps=3,

                num_fixed_steps=0,

                apply_noise_during_interpolation=False,
            )
        )
        subtask_configs.append(
            SubTaskConfig(

                object_ref="sorting_scale",

                subtask_term_signal=None,

                subtask_term_offset_range=(0, 0),

                selection_strategy="nearest_neighbor_object",

                selection_strategy_kwargs={"nn_k": 3},

                action_noise=0.003,

                num_interpolation_steps=3,

                num_fixed_steps=0,

                apply_noise_during_interpolation=False,
            )
        )
        self.subtask_configs["right"] = subtask_configs

        subtask_configs = []
        subtask_configs.append(
            SubTaskConfig(

                object_ref="sorting_beaker",


                subtask_term_signal="grasp_left",
                first_subtask_start_offset_range=(0, 0),

                subtask_term_offset_range=(0, 0),

                selection_strategy="nearest_neighbor_object",

                selection_strategy_kwargs={"nn_k": 3},

                action_noise=0.003,

                num_interpolation_steps=5,

                num_fixed_steps=0,

                apply_noise_during_interpolation=False,
            )
        )
        subtask_configs.append(
            SubTaskConfig(

                object_ref="sorting_bowl",

                subtask_term_signal=None,

                subtask_term_offset_range=(0, 0),

                selection_strategy="nearest_neighbor_object",

                selection_strategy_kwargs={"nn_k": 3},

                action_noise=0.003,

                num_interpolation_steps=5,

                num_fixed_steps=0,

                apply_noise_during_interpolation=False,
            )
        )
        self.subtask_configs["left"] = subtask_configs
