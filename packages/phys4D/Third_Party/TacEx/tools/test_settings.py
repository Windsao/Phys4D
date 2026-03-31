




"""
This file contains the settings for the tests.
"""

import os

ISAACLAB_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
"""Path to the root directory of the Isaac Lab repository."""

DEFAULT_TIMEOUT = 120
"""The default timeout for each test in seconds."""

PER_TEST_TIMEOUTS = {
    "test_articulation.py": 200,
    "test_deformable_object.py": 200,
    "test_rigid_object_collection.py": 200,
    "test_environments.py": 1850,
    "test_environment_determinism.py": 200,
    "test_factory_environments.py": 300,
    "test_env_rendering_logic.py": 300,
    "test_camera.py": 500,
    "test_tiled_camera.py": 300,
    "test_generate_dataset.py": 300,
    "test_rsl_rl_wrapper.py": 200,
    "test_sb3_wrapper.py": 200,
    "test_skrl_wrapper.py": 200,
    "test_operational_space.py": 300,
    "test_terrain_importer.py": 200,
    "test_environments_training.py": 5000,
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

DIRS_TO_SKIP = [

    "/tacex_uipc/libuipc",
]
"""A list of directories with tests to skip. Paths are concatenated to the source dir path."""

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
