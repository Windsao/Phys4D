




"""
This file contains the settings for the tests.
"""

import os

ISAACLAB_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
"""Path to the root directory of the Isaac Lab repository."""

DEFAULT_TIMEOUT = 300
"""The default timeout for each test in seconds."""

PER_TEST_TIMEOUTS = {
    "test_articulation.py": 500,
    "test_stage_in_memory.py": 500,
    "test_environments.py": 2500,
    "test_environments_with_stage_in_memory.py": (
        2500
    ),
    "test_environment_determinism.py": 1000,
    "test_factory_environments.py": 1000,
    "test_multi_agent_environments.py": 800,
    "test_generate_dataset.py": 500,
    "test_pink_ik.py": 1000,
    "test_environments_training.py": (
        6000
    ),
    "test_simulation_render_config.py": 500,
    "test_operational_space.py": 500,
    "test_non_headless_launch.py": 1000,
    "test_rl_games_wrapper.py": 500,
}
"""A dictionary of tests and their timeouts in seconds.

Note: Any tests not listed here will use the default timeout.
"""

TESTS_TO_SKIP = [

    "test_argparser_launch.py",
    "test_build_simulation_context_nonheadless.py",
    "test_env_var_launch.py",
    "test_kwarg_launch.py",
    "test_differential_ik.py",

    "test_record_video.py",
    "test_tiled_camera_env.py",
]
"""A list of tests to skip by run_tests.py"""

TEST_RL_ENVS = [

    "Isaac-Ant-v0",
    "Isaac-Cartpole-v0",

    "Isaac-Lift-Cube-Franka-v0",
    "Isaac-Open-Drawer-Franka-v0",

    "Isaac-Repose-Cube-Allegro-v0",

    "Isaac-Velocity-Flat-Unitree-Go2-v0",
    "Isaac-Velocity-Rough-Anymal-D-v0",
    "Isaac-Velocity-Rough-G1-v0",
]
"""A list of RL environments to test training on by run_train_envs.py"""
