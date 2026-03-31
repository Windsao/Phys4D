




"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher


simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import numpy as np
import os
import shutil
import torch

import isaacsim.core.utils.torch as torch_utils
import pytest

from isaaclab.terrains import FlatPatchSamplingCfg, TerrainGenerator, TerrainGeneratorCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG


@pytest.fixture
def output_dir():
    """Create directory to dump results."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(test_dir, "output", "generator")
    yield output_dir

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)


def test_generation(output_dir):
    """Generates assorted terrains and tests that the resulting mesh has the expected size."""

    cfg = ROUGH_TERRAINS_CFG
    terrain_generator = TerrainGenerator(cfg=cfg)


    print(terrain_generator)


    bounds = terrain_generator.terrain_mesh.bounds
    actualSize = abs(bounds[1] - bounds[0])

    expectedSizeX = cfg.size[0] * cfg.num_rows + 2 * cfg.border_width
    expectedSizeY = cfg.size[1] * cfg.num_cols + 2 * cfg.border_width


    assert actualSize[0] == pytest.approx(expectedSizeX)
    assert actualSize[1] == pytest.approx(expectedSizeY)


@pytest.mark.parametrize("use_global_seed", [True, False])
@pytest.mark.parametrize("seed", [20, 40, 80])
def test_generation_reproducibility(use_global_seed, seed):
    """Generates assorted terrains and tests that the resulting mesh is reproducible.

    We check both scenarios where the seed is set globally only and when it is set both globally and locally.
    Setting only locally is not tested as it is not supported.
    """

    torch_utils.set_seed(seed)


    cfg = ROUGH_TERRAINS_CFG
    cfg.use_cache = False
    cfg.seed = seed if use_global_seed else None
    terrain_generator = TerrainGenerator(cfg=cfg)


    terrain_mesh_1 = terrain_generator.terrain_mesh.copy()


    torch_utils.set_seed(seed)


    terrain_generator = TerrainGenerator(cfg=cfg)


    terrain_mesh_2 = terrain_generator.terrain_mesh.copy()


    np.testing.assert_allclose(
        terrain_mesh_1.vertices, terrain_mesh_2.vertices, atol=1e-5, err_msg="Vertices are not equal"
    )
    np.testing.assert_allclose(terrain_mesh_1.faces, terrain_mesh_2.faces, atol=1e-5, err_msg="Faces are not equal")


@pytest.mark.parametrize("curriculum", [True, False])
def test_generation_cache(output_dir, curriculum):
    """Generate the terrain and check that caching works.

    When caching is enabled, the terrain should be generated only once and the same terrain should be returned
    when the terrain generator is created again.
    """

    cfg: TerrainGeneratorCfg = ROUGH_TERRAINS_CFG
    cfg.use_cache = True
    cfg.seed = 0
    cfg.cache_dir = output_dir
    cfg.curriculum = curriculum
    terrain_generator = TerrainGenerator(cfg=cfg)

    terrain_mesh_1 = terrain_generator.terrain_mesh.copy()



    hash_ids_1 = set(os.listdir(cfg.cache_dir))
    assert os.listdir(cfg.cache_dir)



    torch_utils.set_seed(12456)


    terrain_generator = TerrainGenerator(cfg=cfg)

    terrain_mesh_2 = terrain_generator.terrain_mesh.copy()


    hash_ids_2 = set(os.listdir(cfg.cache_dir))
    assert len(hash_ids_1) == len(hash_ids_2)
    assert hash_ids_1 == hash_ids_2



    assert terrain_mesh_1 is not terrain_mesh_2


    np.testing.assert_allclose(
        terrain_mesh_1.vertices, terrain_mesh_2.vertices, atol=1e-5, err_msg="Vertices are not equal"
    )
    np.testing.assert_allclose(terrain_mesh_1.faces, terrain_mesh_2.faces, atol=1e-5, err_msg="Faces are not equal")


def test_terrain_flat_patches():
    """Test the flat patches generation."""

    cfg = ROUGH_TERRAINS_CFG

    for _, sub_terrain_cfg in cfg.sub_terrains.items():
        sub_terrain_cfg.flat_patch_sampling = {
            "root_spawn": FlatPatchSamplingCfg(num_patches=8, patch_radius=0.5, max_height_diff=0.05),
            "target_spawn": FlatPatchSamplingCfg(num_patches=5, patch_radius=0.35, max_height_diff=0.05),
        }

    terrain_generator = TerrainGenerator(cfg=cfg)


    assert terrain_generator.flat_patches

    assert terrain_generator.flat_patches["root_spawn"].shape == (cfg.num_rows, cfg.num_cols, 8, 3)
    assert terrain_generator.flat_patches["target_spawn"].shape == (cfg.num_rows, cfg.num_cols, 5, 3)

    for _, flat_patches in terrain_generator.flat_patches.items():
        assert not torch.allclose(flat_patches, torch.zeros_like(flat_patches))
