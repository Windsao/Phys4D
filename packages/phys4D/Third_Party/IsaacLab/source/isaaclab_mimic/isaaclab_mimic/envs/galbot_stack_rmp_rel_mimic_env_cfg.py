





from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.stack.config.galbot.stack_rmp_rel_env_cfg import (
    RmpFlowGalbotLeftArmCubeStackEnvCfg,
    RmpFlowGalbotRightArmCubeStackEnvCfg,
)


@configclass
class RmpFlowGalbotLeftArmGripperCubeStackRelMimicEnvCfg(RmpFlowGalbotLeftArmCubeStackEnvCfg, MimicEnvCfg):
    """
    Isaac Lab Mimic environment config class for Galbot Gripper Cube Stack IK Rel env.
    """

    def __post_init__(self):

        super().__post_init__()


        self.datagen_config.name = "demo_src_stack_isaac_lab_task_D0"
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

                object_ref="cube_2",


                subtask_term_signal="grasp_1",


                subtask_term_offset_range=(
                    18,
                    25,
                ),

                selection_strategy="nearest_neighbor_object",

                selection_strategy_kwargs={"nn_k": 3},

                action_noise=0.01,

                num_interpolation_steps=15,

                num_fixed_steps=0,

                apply_noise_during_interpolation=False,
            )
        )
        subtask_configs.append(
            SubTaskConfig(

                object_ref="cube_1",

                subtask_term_signal="stack_1",

                subtask_term_offset_range=(
                    18,
                    25,
                ),

                selection_strategy="nearest_neighbor_object",

                selection_strategy_kwargs={"nn_k": 3},

                action_noise=0.01,

                num_interpolation_steps=15,

                num_fixed_steps=0,

                apply_noise_during_interpolation=False,
            )
        )
        subtask_configs.append(
            SubTaskConfig(

                object_ref="cube_3",

                subtask_term_signal="grasp_2",

                subtask_term_offset_range=(
                    25,
                    30,
                ),

                selection_strategy="nearest_neighbor_object",

                selection_strategy_kwargs={"nn_k": 3},

                action_noise=0.01,

                num_interpolation_steps=15,

                num_fixed_steps=0,

                apply_noise_during_interpolation=False,
            )
        )
        subtask_configs.append(
            SubTaskConfig(

                object_ref="cube_2",

                subtask_term_signal=None,

                subtask_term_offset_range=(0, 0),

                selection_strategy="nearest_neighbor_object",

                selection_strategy_kwargs={"nn_k": 3},

                action_noise=0.01,

                num_interpolation_steps=15,

                num_fixed_steps=0,

                apply_noise_during_interpolation=False,
            )
        )
        self.subtask_configs["galbot"] = subtask_configs


@configclass
class RmpFlowGalbotRightArmSuctionCubeStackRelMimicEnvCfg(RmpFlowGalbotRightArmCubeStackEnvCfg, MimicEnvCfg):
    """
    Isaac Lab Mimic environment config class for Galbot Suction Gripper Cube Stack RmpFlow Rel env.
    """

    def __post_init__(self):

        super().__post_init__()


        self.datagen_config.name = "demo_src_stack_isaac_lab_task_D0"
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

                object_ref="cube_2",


                subtask_term_signal="grasp_1",


                subtask_term_offset_range=(5, 10),

                selection_strategy="nearest_neighbor_object",

                selection_strategy_kwargs={"nn_k": 3},

                action_noise=0.01,

                num_interpolation_steps=15,

                num_fixed_steps=0,

                apply_noise_during_interpolation=False,
            )
        )
        subtask_configs.append(
            SubTaskConfig(

                object_ref="cube_1",

                subtask_term_signal="stack_1",

                subtask_term_offset_range=(2, 10),

                selection_strategy="nearest_neighbor_object",

                selection_strategy_kwargs={"nn_k": 3},

                action_noise=0.01,

                num_interpolation_steps=15,

                num_fixed_steps=0,

                apply_noise_during_interpolation=False,
            )
        )
        subtask_configs.append(
            SubTaskConfig(

                object_ref="cube_3",

                subtask_term_signal="grasp_2",

                subtask_term_offset_range=(5, 10),

                selection_strategy="nearest_neighbor_object",

                selection_strategy_kwargs={"nn_k": 3},

                action_noise=0.01,

                num_interpolation_steps=15,

                num_fixed_steps=0,

                apply_noise_during_interpolation=False,
            )
        )
        subtask_configs.append(
            SubTaskConfig(

                object_ref="cube_2",

                subtask_term_signal=None,

                subtask_term_offset_range=(0, 0),

                selection_strategy="nearest_neighbor_object",

                selection_strategy_kwargs={"nn_k": 3},

                action_noise=0.01,

                num_interpolation_steps=15,

                num_fixed_steps=0,

                apply_noise_during_interpolation=False,
            )
        )
        self.subtask_configs["galbot"] = subtask_configs
