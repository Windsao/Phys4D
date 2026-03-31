"""Configuration for custom terrains."""

from isaaclab.utils import configclass
from dataclasses import MISSING
from . import terrains as terrain_gen


@configclass
class MagicTerrainGeneratorCfg(terrain_gen.TerrainGeneratorCfg):
    size: int = MISSING
    num_rows: int = MISSING
    num_cols: int = MISSING
    sub_terrains: dict[str, terrain_gen.SubTerrainBaseCfg] = None
    horizontal_scale = 0.1
    vertical_scale = 0.005
    slope_threshold = 0.75
    use_cache = False
    sub_terrains = {
        "hf_pyramid_stair": terrain_gen.HfPyramidStairsTerrainCfg(
            proportion=0.0,
            step_height_range=(0.08, 0.23),
            step_width=0.35,
            platform_width=3.0,
            border_width=0.1,
        ),
        "magic_parkour": terrain_gen.ParkourTerrainCfg(
            proportion=0.5,
            x_range=[0.5, 1.0],
            y_range=[0.3, 0.4],
            stone_len_range=[0.8, 1.0],
            stone_width_range=[0.6, 0.8],
            incline_height=0.1,
            pit_depth=[0.5, 1.0],
            num_goals=12,
        ),
        "magic_hurdle": terrain_gen.HurdleTerrainCfg(
            proportion=0.0,
            hurdle_range=[0.1, 0.3],
            hurdle_height_range=[0.08, 0.18],
            flat_size=0.8,
        ),
        "magic_bridge": terrain_gen.BridgeTerrainCfg(
            proportion=0.5,
            bridge_width_range=[0.5, 0.4],
            bridge_height=0.7,
            platform_width=1.5,
        ),
    }
