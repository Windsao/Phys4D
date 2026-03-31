




"""
This script demonstrates procedural terrains with flat patches.

Example usage:

.. code-block:: bash

    # Generate terrain with height color scheme
    ./isaaclab.sh -p scripts/demos/procedural_terrain.py --color_scheme height

    # Generate terrain with random color scheme
    ./isaaclab.sh -p scripts/demos/procedural_terrain.py --color_scheme random

    # Generate terrain with no color scheme
    ./isaaclab.sh -p scripts/demos/procedural_terrain.py --color_scheme none

    # Generate terrain with curriculum
    ./isaaclab.sh -p scripts/demos/procedural_terrain.py --use_curriculum

    # Generate terrain with curriculum along with flat patches
    ./isaaclab.sh -p scripts/demos/procedural_terrain.py --use_curriculum --show_flat_patches

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="This script demonstrates procedural terrain generation.")
parser.add_argument(
    "--color_scheme",
    type=str,
    default="none",
    choices=["height", "random", "none"],
    help="Color scheme to use for the terrain generation.",
)
parser.add_argument(
    "--use_curriculum",
    action="store_true",
    default=False,
    help="Whether to use the curriculum for the terrain generation.",
)
parser.add_argument(
    "--show_flat_patches",
    action="store_true",
    default=False,
    help="Whether to show the flat patches computed during the terrain generation.",
)

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import random
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBase
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.terrains import FlatPatchSamplingCfg, TerrainImporter, TerrainImporterCfg




from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG


def design_scene() -> tuple[dict, torch.Tensor]:
    """Designs the scene."""

    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)


    terrain_gen_cfg = ROUGH_TERRAINS_CFG.replace(curriculum=args_cli.use_curriculum, color_scheme=args_cli.color_scheme)





    if args_cli.show_flat_patches:
        for sub_terrain_name, sub_terrain_cfg in terrain_gen_cfg.sub_terrains.items():
            sub_terrain_cfg.flat_patch_sampling = {
                sub_terrain_name: FlatPatchSamplingCfg(num_patches=10, patch_radius=0.5, max_height_diff=0.05)
            }


    terrain_importer_cfg = TerrainImporterCfg(
        num_envs=2048,
        env_spacing=3.0,
        prim_path="/World/ground",
        max_init_terrain_level=None,
        terrain_type="generator",
        terrain_generator=terrain_gen_cfg,
        debug_vis=True,
    )

    if args_cli.color_scheme in ["height", "random"]:
        terrain_importer_cfg.visual_material = None

    terrain_importer = TerrainImporter(terrain_importer_cfg)


    if args_cli.show_flat_patches:

        vis_cfg = VisualizationMarkersCfg(prim_path="/Visuals/TerrainFlatPatches", markers={})
        for name in terrain_importer.flat_patches:
            vis_cfg.markers[name] = sim_utils.CylinderCfg(
                radius=0.5,
                height=0.1,
                visual_material=sim_utils.GlassMdlCfg(glass_color=(random.random(), random.random(), random.random())),
            )
        flat_patches_visualizer = VisualizationMarkers(vis_cfg)


        all_patch_locations = []
        all_patch_indices = []
        for i, patch_locations in enumerate(terrain_importer.flat_patches.values()):
            num_patch_locations = patch_locations.view(-1, 3).shape[0]

            all_patch_locations.append(patch_locations.view(-1, 3))
            all_patch_indices += [i] * num_patch_locations

        flat_patches_visualizer.visualize(torch.cat(all_patch_locations), marker_indices=all_patch_indices)


    scene_entities = {"terrain": terrain_importer}
    return scene_entities, terrain_importer.env_origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, AssetBase], origins: torch.Tensor):
    """Runs the simulation loop."""

    while simulation_app.is_running():

        sim.step()


def main():
    """Main function."""

    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view(eye=[5.0, 5.0, 5.0], target=[0.0, 0.0, 0.0])

    scene_entities, scene_origins = design_scene()

    sim.reset()

    print("[INFO]: Setup complete...")

    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":

    main()

    simulation_app.close()
