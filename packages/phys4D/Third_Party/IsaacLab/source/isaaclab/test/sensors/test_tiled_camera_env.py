




"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description=(
        "Test Isaac-Cartpole-RGB-Camera-Direct-v0 environment with different resolutions and number of environments."
    )
)
parser.add_argument("--save_images", action="store_true", default=False, help="Save out renders to file.")
parser.add_argument("unittest_args", nargs="*")


args_cli = parser.parse_args()

sys.argv[1:] = args_cli.unittest_args


simulation_app = AppLauncher(headless=True, enable_cameras=True).app

"""Rest everything follows."""

import gymnasium as gym
import sys

import omni.usd
import pytest

from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.sensors import save_images_to_file

import isaaclab_tasks
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


@pytest.mark.skip(reason="Currently takes too long to run")
def test_tiled_resolutions_tiny():
    """Define settings for resolution and number of environments"""
    num_envs = 1024
    tile_widths = range(32, 48)
    tile_heights = range(32, 48)
    _launch_tests(tile_widths, tile_heights, num_envs)


@pytest.mark.skip(reason="Currently takes too long to run")
def test_tiled_resolutions_small():
    """Define settings for resolution and number of environments"""
    num_envs = 300
    tile_widths = range(128, 156)
    tile_heights = range(128, 156)
    _launch_tests(tile_widths, tile_heights, num_envs)


@pytest.mark.skip(reason="Currently takes too long to run")
def test_tiled_resolutions_medium():
    """Define settings for resolution and number of environments"""
    num_envs = 64
    tile_widths = range(320, 400, 20)
    tile_heights = range(320, 400, 20)
    _launch_tests(tile_widths, tile_heights, num_envs)


@pytest.mark.skip(reason="Currently takes too long to run")
def test_tiled_resolutions_large():
    """Define settings for resolution and number of environments"""
    num_envs = 4
    tile_widths = range(480, 640, 40)
    tile_heights = range(480, 640, 40)
    _launch_tests(tile_widths, tile_heights, num_envs)


@pytest.mark.skip(reason="Currently takes too long to run")
def test_tiled_resolutions_edge_cases():
    """Define settings for resolution and number of environments"""
    num_envs = 1000
    tile_widths = [12, 67, 93, 147]
    tile_heights = [12, 67, 93, 147]
    _launch_tests(tile_widths, tile_heights, num_envs)


@pytest.mark.skip(reason="Currently takes too long to run")
def test_tiled_num_envs_edge_cases():
    """Define settings for resolution and number of environments"""
    num_envs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 53, 359, 733, 927]
    tile_widths = [67, 93, 147]
    tile_heights = [67, 93, 147]
    for n_envs in num_envs:
        _launch_tests(tile_widths, tile_heights, n_envs)





def _launch_tests(tile_widths: range, tile_heights: range, num_envs: int):
    """Run through different resolutions for tiled rendering"""
    device = "cuda:0"
    task_name = "Isaac-Cartpole-RGB-Camera-Direct-v0"

    for width in tile_widths:
        for height in tile_heights:

            omni.usd.get_context().new_stage()

            env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg = parse_env_cfg(task_name, device=device, num_envs=num_envs)
            env_cfg.tiled_camera.width = width
            env_cfg.tiled_camera.height = height
            print(f">>> Running test for resolution: {width} x {height}")

            _run_environment(env_cfg)

            print(f">>> Closing environment: {task_name}")
            print("-" * 80)


def _run_environment(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg):
    """Run environment and capture a rendered image."""

    env: ManagerBasedRLEnv | DirectRLEnv = gym.make("Isaac-Cartpole-RGB-Camera-Direct-v0", cfg=env_cfg)


    env.sim.set_setting("/physics/cooking/ujitsoCollisionCooking", False)


    obs, _ = env.reset()

    if args_cli.save_images:
        save_images_to_file(
            obs["policy"] + 0.93,
            f"output_{env.num_envs}_{env_cfg.tiled_camera.width}x{env_cfg.tiled_camera.height}.png",
        )


    env.close()
