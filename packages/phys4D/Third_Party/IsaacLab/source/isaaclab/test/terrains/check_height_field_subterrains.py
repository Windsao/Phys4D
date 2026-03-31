




"""Launch Isaac Sim Simulator first."""

import argparse

parser = argparse.ArgumentParser(description="Generate terrains using trimesh")
parser.add_argument(
    "--headless", action="store_true", default=False, help="Don't create a window to display each output."
)
args_cli = parser.parse_args()

from isaaclab.app import AppLauncher



simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import os
import trimesh

import isaaclab.terrains.height_field as hf_gen
from isaaclab.terrains.utils import color_meshes_by_height


def test_random_uniform_terrain(difficulty: float, output_dir: str, headless: bool):

    cfg = hf_gen.HfRandomUniformTerrainCfg(
        size=(8.0, 8.0),
        horizontal_scale=0.1,
        vertical_scale=0.005,
        border_width=0.0,
        noise_range=(-0.05, 0.05),
        noise_step=0.005,
        downsampled_scale=0.2,
    )

    meshes, origin = cfg.function(difficulty=difficulty, cfg=cfg)

    colored_mesh = color_meshes_by_height(meshes)

    origin_transform = trimesh.transformations.translation_matrix(origin)
    origin_marker = trimesh.creation.axis(origin_size=0.1, transform=origin_transform)

    scene = trimesh.Scene([colored_mesh, origin_marker])

    data = scene.save_image(resolution=(640, 480))

    with open(os.path.join(output_dir, "random_uniform_terrain.jpg"), "wb") as f:
        f.write(data)

    if not headless:
        trimesh.viewer.SceneViewer(scene=scene, caption="Random Uniform Terrain")


def test_pyramid_sloped_terrain(difficulty: float, inverted: bool, output_dir: str, headless: bool):

    cfg = hf_gen.HfPyramidSlopedTerrainCfg(
        size=(8.0, 8.0),
        horizontal_scale=0.1,
        vertical_scale=0.005,
        border_width=0.0,
        slope_range=(0.0, 0.4),
        platform_width=1.5,
        inverted=inverted,
    )

    meshes, origin = cfg.function(difficulty=difficulty, cfg=cfg)

    colored_mesh = color_meshes_by_height(meshes)

    origin_transform = trimesh.transformations.translation_matrix(origin)
    origin_marker = trimesh.creation.axis(origin_size=0.1, transform=origin_transform)

    scene = trimesh.Scene([colored_mesh, origin_marker])

    data = scene.save_image(resolution=(640, 480))

    if inverted:
        caption = "Inverted Pyramid Sloped Terrain"
        filename = "inverted_pyramid_sloped_terrain.jpg"
    else:
        caption = "Pyramid Sloped Terrain"
        filename = "pyramid_sloped_terrain.jpg"

    with open(os.path.join(output_dir, filename), "wb") as f:
        f.write(data)

    if not headless:
        trimesh.viewer.SceneViewer(scene=scene, caption=caption)


def test_pyramid_stairs_terrain(difficulty: float, inverted: bool, output_dir: str, headless: bool):

    cfg = hf_gen.HfPyramidStairsTerrainCfg(
        size=(8.0, 8.0),
        horizontal_scale=0.1,
        vertical_scale=0.005,
        border_width=0.0,
        platform_width=1.5,
        step_width=0.301,
        step_height_range=(0.05, 0.23),
        inverted=inverted,
    )

    meshes, origin = cfg.function(difficulty=difficulty, cfg=cfg)

    colored_mesh = color_meshes_by_height(meshes)

    origin_transform = trimesh.transformations.translation_matrix(origin)
    origin_marker = trimesh.creation.axis(origin_size=0.1, transform=origin_transform)

    scene = trimesh.Scene([colored_mesh, origin_marker])

    data = scene.save_image(resolution=(640, 480))

    if inverted:
        caption = "Inverted Pyramid Stairs Terrain"
        filename = "inverted_pyramid_stairs_terrain.jpg"
    else:
        caption = "Pyramid Stairs Terrain"
        filename = "pyramid_stairs_terrain.jpg"

    with open(os.path.join(output_dir, filename), "wb") as f:
        f.write(data)

    if not headless:
        trimesh.viewer.SceneViewer(scene=scene, caption=caption)


def test_discrete_obstacles_terrain(difficulty: float, obstacle_height_mode: str, output_dir: str, headless: bool):

    cfg = hf_gen.HfDiscreteObstaclesTerrainCfg(
        size=(8.0, 8.0),
        horizontal_scale=0.1,
        vertical_scale=0.005,
        border_width=0.0,
        num_obstacles=50,
        obstacle_height_mode=obstacle_height_mode,
        obstacle_width_range=(0.25, 0.75),
        obstacle_height_range=(1.0, 2.0),
        platform_width=1.5,
    )

    meshes, origin = cfg.function(difficulty=difficulty, cfg=cfg)

    colored_mesh = color_meshes_by_height(meshes)

    origin_transform = trimesh.transformations.translation_matrix(origin)
    origin_marker = trimesh.creation.axis(origin_size=0.1, transform=origin_transform)

    scene = trimesh.Scene([colored_mesh, origin_marker])

    data = scene.save_image(resolution=(640, 480))

    if obstacle_height_mode == "choice":
        caption = "Discrete Obstacles Terrain (Sampled Height)"
        filename = "discrete_obstacles_terrain_choice.jpg"
    elif obstacle_height_mode == "fixed":
        caption = "Discrete Obstacles Terrain (Fixed Height)"
        filename = "discrete_obstacles_terrain_fixed.jpg"
    else:
        raise ValueError(f"Unknown obstacle height mode: {obstacle_height_mode}")

    with open(os.path.join(output_dir, filename), "wb") as f:
        f.write(data)

    if not headless:
        trimesh.viewer.SceneViewer(scene=scene, caption=caption)


def test_wave_terrain(difficulty: float, output_dir: str, headless: bool):

    cfg = hf_gen.HfWaveTerrainCfg(
        size=(8.0, 8.0),
        horizontal_scale=0.1,
        vertical_scale=0.005,
        border_width=0.0,
        num_waves=5,
        amplitude_range=(0.5, 1.0),
    )

    meshes, origin = cfg.function(difficulty=difficulty, cfg=cfg)

    colored_mesh = color_meshes_by_height(meshes)

    origin_transform = trimesh.transformations.translation_matrix(origin)
    origin_marker = trimesh.creation.axis(origin_size=0.1, transform=origin_transform)

    scene = trimesh.Scene([colored_mesh, origin_marker])

    data = scene.save_image(resolution=(640, 480))

    with open(os.path.join(output_dir, "wave_terrain.jpg"), "wb") as f:
        f.write(data)

    if not headless:
        trimesh.viewer.SceneViewer(scene=scene, caption="Wave Terrain")


def test_stepping_stones_terrain(difficulty: float, output_dir: str, headless: bool):

    cfg = hf_gen.HfSteppingStonesTerrainCfg(
        size=(8.0, 8.0),
        horizontal_scale=0.1,
        vertical_scale=0.005,
        platform_width=1.5,
        border_width=0.0,
        stone_width_range=(0.25, 1.575),
        stone_height_max=0.2,
        stone_distance_range=(0.05, 0.1),
        holes_depth=-2.0,
    )

    meshes, origin = cfg.function(difficulty=difficulty, cfg=cfg)

    colored_mesh = color_meshes_by_height(meshes)

    origin_transform = trimesh.transformations.translation_matrix(origin)
    origin_marker = trimesh.creation.axis(origin_size=0.1, transform=origin_transform)

    scene = trimesh.Scene([colored_mesh, origin_marker])

    data = scene.save_image(resolution=(640, 480))

    with open(os.path.join(output_dir, "stepping_stones_terrain.jpg"), "wb") as f:
        f.write(data)

    if not headless:
        trimesh.viewer.SceneViewer(scene=scene, caption="Stepping Stones Terrain")


def main():

    test_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(test_dir, "output", "terrains", "height_field")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    headless = args_cli.headless

    test_random_uniform_terrain(difficulty=0.25, output_dir=output_dir, headless=headless)
    test_pyramid_sloped_terrain(difficulty=0.25, inverted=False, output_dir=output_dir, headless=headless)
    test_pyramid_sloped_terrain(difficulty=0.25, inverted=True, output_dir=output_dir, headless=headless)
    test_pyramid_stairs_terrain(difficulty=0.25, inverted=False, output_dir=output_dir, headless=headless)
    test_pyramid_stairs_terrain(difficulty=0.25, inverted=True, output_dir=output_dir, headless=headless)
    test_discrete_obstacles_terrain(
        difficulty=0.25, obstacle_height_mode="choice", output_dir=output_dir, headless=headless
    )
    test_discrete_obstacles_terrain(
        difficulty=0.25, obstacle_height_mode="fixed", output_dir=output_dir, headless=headless
    )
    test_wave_terrain(difficulty=0.25, output_dir=output_dir, headless=headless)
    test_stepping_stones_terrain(difficulty=1.0, output_dir=output_dir, headless=headless)


if __name__ == "__main__":

    main()

    simulation_app.close()
