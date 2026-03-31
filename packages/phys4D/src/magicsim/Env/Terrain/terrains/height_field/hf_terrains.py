"""Functions to generate height fields for different terrains."""

from __future__ import annotations

import numpy as np
import scipy.interpolate as interpolate
from typing import TYPE_CHECKING

from .utils import height_field_to_mesh

if TYPE_CHECKING:
    from . import hf_terrains_cfg


@height_field_to_mesh
def random_uniform_terrain(
    difficulty: float, cfg: hf_terrains_cfg.HfRandomUniformTerrainCfg
) -> np.ndarray:
    """Generate a terrain with height sampled uniformly from a specified range.

    .. image:: ../../_static/terrains/height_field/random_uniform_terrain.jpg
       :width: 40%
       :align: center

    Note:
        The :obj:`difficulty` parameter is ignored for this terrain.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.

    Raises:
        ValueError: When the downsampled scale is smaller than the horizontal scale.
    """

    if cfg.downsampled_scale is None:
        cfg.downsampled_scale = cfg.horizontal_scale
    elif cfg.downsampled_scale < cfg.horizontal_scale:
        raise ValueError(
            "Downsampled scale must be larger than or equal to the horizontal scale:"
            f" {cfg.downsampled_scale} < {cfg.horizontal_scale}."
        )

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)

    width_downsampled = int(cfg.size[0] / cfg.downsampled_scale)
    length_downsampled = int(cfg.size[1] / cfg.downsampled_scale)

    height_min = int(cfg.noise_range[0] / cfg.vertical_scale)
    height_max = int(cfg.noise_range[1] / cfg.vertical_scale)
    height_step = int(cfg.noise_step / cfg.vertical_scale)

    height_range = np.arange(height_min, height_max + height_step, height_step)

    height_field_downsampled = np.random.choice(
        height_range, size=(width_downsampled, length_downsampled)
    )

    x = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_downsampled)
    y = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_downsampled)
    func = interpolate.RectBivariateSpline(x, y, height_field_downsampled)

    x_upsampled = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_pixels)
    y_upsampled = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_pixels)
    z_upsampled = func(x_upsampled, y_upsampled)

    return np.rint(z_upsampled).astype(np.int16)


@height_field_to_mesh
def pyramid_sloped_terrain(
    difficulty: float, cfg: hf_terrains_cfg.HfPyramidSlopedTerrainCfg
) -> np.ndarray:
    """Generate a terrain with a truncated pyramid structure.

    The terrain is a pyramid-shaped sloped surface with a slope of :obj:`slope` that trims into a flat platform
    at the center. The slope is defined as the ratio of the height change along the x axis to the width along the
    x axis. For example, a slope of 1.0 means that the height changes by 1 unit for every 1 unit of width.

    If the :obj:`cfg.inverted` flag is set to :obj:`True`, the terrain is inverted such that
    the platform is at the bottom.

    .. image:: ../../_static/terrains/height_field/pyramid_sloped_terrain.jpg
       :width: 40%

    .. image:: ../../_static/terrains/height_field/inverted_pyramid_sloped_terrain.jpg
       :width: 40%

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.
    """

    if cfg.inverted:
        slope = -cfg.slope_range[0] - difficulty * (
            cfg.slope_range[1] - cfg.slope_range[0]
        )
    else:
        slope = cfg.slope_range[0] + difficulty * (
            cfg.slope_range[1] - cfg.slope_range[0]
        )

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)

    height_max = int(slope * cfg.size[0] / 2 / cfg.vertical_scale)

    center_x = int(width_pixels / 2)
    center_y = int(length_pixels / 2)

    x = np.arange(0, width_pixels)
    y = np.arange(0, length_pixels)
    xx, yy = np.meshgrid(x, y, sparse=True)

    xx = (center_x - np.abs(center_x - xx)) / center_x
    yy = (center_y - np.abs(center_y - yy)) / center_y

    xx = xx.reshape(width_pixels, 1)
    yy = yy.reshape(1, length_pixels)

    hf_raw = np.zeros((width_pixels, length_pixels))
    hf_raw = height_max * xx * yy

    platform_width = int(cfg.platform_width / cfg.horizontal_scale / 2)

    x_pf = width_pixels // 2 - platform_width
    y_pf = length_pixels // 2 - platform_width
    z_pf = hf_raw[x_pf, y_pf]
    hf_raw = np.clip(hf_raw, min(0, z_pf), max(0, z_pf))

    return np.rint(hf_raw).astype(np.int16)


@height_field_to_mesh
def pyramid_stairs_terrain(
    difficulty: float, cfg: hf_terrains_cfg.HfPyramidStairsTerrainCfg
) -> np.ndarray:
    """Generate a terrain with a pyramid stair pattern.

    The terrain is a pyramid stair pattern which trims to a flat platform at the center of the terrain.

    If the :obj:`cfg.inverted` flag is set to :obj:`True`, the terrain is inverted such that
    the platform is at the bottom.

    .. image:: ../../_static/terrains/height_field/pyramid_stairs_terrain.jpg
       :width: 40%

    .. image:: ../../_static/terrains/height_field/inverted_pyramid_stairs_terrain.jpg
       :width: 40%

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.
    """

    step_height = cfg.step_height_range[0] + difficulty * (
        cfg.step_height_range[1] - cfg.step_height_range[0]
    )
    if cfg.inverted:
        step_height *= -1

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)

    step_width = int(cfg.step_width / cfg.horizontal_scale)
    step_height = int(step_height / cfg.vertical_scale)

    platform_width = int(cfg.platform_width / cfg.horizontal_scale)

    hf_raw = np.zeros((width_pixels, length_pixels))

    current_step_height = 0
    start_x, start_y = 0, 0
    stop_x, stop_y = width_pixels, length_pixels
    while (stop_x - start_x) > platform_width and (stop_y - start_y) > platform_width:
        start_x += step_width
        stop_x -= step_width

        start_y += step_width
        stop_y -= step_width

        current_step_height += step_height

        hf_raw[start_x:stop_x, start_y:stop_y] = current_step_height

    return np.rint(hf_raw).astype(np.int16)


@height_field_to_mesh
def discrete_obstacles_terrain(
    difficulty: float, cfg: hf_terrains_cfg.HfDiscreteObstaclesTerrainCfg
) -> np.ndarray:
    """Generate a terrain with randomly generated obstacles as pillars with positive and negative heights.

    The terrain is a flat platform at the center of the terrain with randomly generated obstacles as pillars
    with positive and negative height. The obstacles are randomly generated cuboids with a random width and
    height. They are placed randomly on the terrain with a minimum distance of :obj:`cfg.platform_width`
    from the center of the terrain.

    .. image:: ../../_static/terrains/height_field/discrete_obstacles_terrain.jpg
       :width: 40%
       :align: center

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.
    """

    obs_height = cfg.obstacle_height_range[0] + difficulty * (
        cfg.obstacle_height_range[1] - cfg.obstacle_height_range[0]
    )

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)

    obs_height = int(obs_height / cfg.vertical_scale)
    obs_width_min = int(cfg.obstacle_width_range[0] / cfg.horizontal_scale)
    obs_width_max = int(cfg.obstacle_width_range[1] / cfg.horizontal_scale)

    platform_width = int(cfg.platform_width / cfg.horizontal_scale)

    obs_width_range = np.arange(obs_width_min, obs_width_max, 4)
    obs_length_range = np.arange(obs_width_min, obs_width_max, 4)

    obs_x_range = np.arange(0, width_pixels, 4)
    obs_y_range = np.arange(0, length_pixels, 4)

    hf_raw = np.zeros((width_pixels, length_pixels))

    for _ in range(cfg.num_obstacles):
        if cfg.obstacle_height_mode == "choice":
            height = np.random.choice(
                [-obs_height, -obs_height // 2, obs_height // 2, obs_height]
            )
        elif cfg.obstacle_height_mode == "fixed":
            height = obs_height
        else:
            raise ValueError(
                f"Unknown obstacle height mode '{cfg.obstacle_height_mode}'. Must be 'choice' or 'fixed'."
            )
        width = int(np.random.choice(obs_width_range))
        length = int(np.random.choice(obs_length_range))

        x_start = int(np.random.choice(obs_x_range))
        y_start = int(np.random.choice(obs_y_range))

        if x_start + width > width_pixels:
            x_start = width_pixels - width
        if y_start + length > length_pixels:
            y_start = length_pixels - length

        hf_raw[x_start : x_start + width, y_start : y_start + length] = height

    x1 = (width_pixels - platform_width) // 2
    x2 = (width_pixels + platform_width) // 2
    y1 = (length_pixels - platform_width) // 2
    y2 = (length_pixels + platform_width) // 2
    hf_raw[x1:x2, y1:y2] = 0

    return np.rint(hf_raw).astype(np.int16)


@height_field_to_mesh
def wave_terrain(
    difficulty: float, cfg: hf_terrains_cfg.HfWaveTerrainCfg
) -> np.ndarray:
    r"""Generate a terrain with a wave pattern.

    The terrain is a flat platform at the center of the terrain with a wave pattern. The wave pattern
    is generated by adding sinusoidal waves based on the number of waves and the amplitude of the waves.

    The height of the terrain at a point :math:`(x, y)` is given by:

    .. math::

        h(x, y) =  A \left(\sin\left(\frac{2 \pi x}{\lambda}\right) + \cos\left(\frac{2 \pi y}{\lambda}\right) \right)

    where :math:`A` is the amplitude of the waves, :math:`\lambda` is the wavelength of the waves.

    .. image:: ../../_static/terrains/height_field/wave_terrain.jpg
       :width: 40%
       :align: center

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.

    Raises:
        ValueError: When the number of waves is non-positive.
    """

    if cfg.num_waves < 0:
        raise ValueError(
            f"Number of waves must be a positive integer. Got: {cfg.num_waves}."
        )

    amplitude = cfg.amplitude_range[0] + difficulty * (
        cfg.amplitude_range[1] - cfg.amplitude_range[0]
    )

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    amplitude_pixels = int(0.5 * amplitude / cfg.vertical_scale)

    wave_length = length_pixels / cfg.num_waves
    wave_number = 2 * np.pi / wave_length

    x = np.arange(0, width_pixels)
    y = np.arange(0, length_pixels)
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = xx.reshape(width_pixels, 1)
    yy = yy.reshape(1, length_pixels)

    hf_raw = np.zeros((width_pixels, length_pixels))

    hf_raw += amplitude_pixels * (np.cos(yy * wave_number) + np.sin(xx * wave_number))

    return np.rint(hf_raw).astype(np.int16)


@height_field_to_mesh
def stepping_stones_terrain(
    difficulty: float, cfg: hf_terrains_cfg.HfSteppingStonesTerrainCfg
) -> np.ndarray:
    """Generate a terrain with a stepping stones pattern.

    The terrain is a stepping stones pattern which trims to a flat platform at the center of the terrain.

    .. image:: ../../_static/terrains/height_field/stepping_stones_terrain.jpg
       :width: 40%
       :align: center

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.
    """

    stone_width = cfg.stone_width_range[1] - difficulty * (
        cfg.stone_width_range[1] - cfg.stone_width_range[0]
    )
    stone_distance = cfg.stone_distance_range[0] + difficulty * (
        cfg.stone_distance_range[1] - cfg.stone_distance_range[0]
    )

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)

    stone_distance = int(stone_distance / cfg.horizontal_scale)
    stone_width = int(stone_width / cfg.horizontal_scale)
    stone_height_max = int(cfg.stone_height_max / cfg.vertical_scale)

    holes_depth = int(cfg.holes_depth / cfg.vertical_scale)

    platform_width = int(cfg.platform_width / cfg.horizontal_scale)

    stone_height_range = np.arange(-stone_height_max - 1, stone_height_max, step=1)

    hf_raw = np.full((width_pixels, length_pixels), holes_depth)

    start_x, start_y = 0, 0

    if length_pixels >= width_pixels:
        while start_y < length_pixels:
            stop_y = min(length_pixels, start_y + stone_width)

            start_x = np.random.randint(0, stone_width)
            stop_x = max(0, start_x - stone_distance)

            hf_raw[0:stop_x, start_y:stop_y] = np.random.choice(stone_height_range)

            while start_x < width_pixels:
                stop_x = min(width_pixels, start_x + stone_width)
                hf_raw[start_x:stop_x, start_y:stop_y] = np.random.choice(
                    stone_height_range
                )
                start_x += stone_width + stone_distance

            start_y += stone_width + stone_distance
    elif width_pixels > length_pixels:
        while start_x < width_pixels:
            stop_x = min(width_pixels, start_x + stone_width)

            start_y = np.random.randint(0, stone_width)
            stop_y = max(0, start_y - stone_distance)

            hf_raw[start_x:stop_x, 0:stop_y] = np.random.choice(stone_height_range)

            while start_y < length_pixels:
                stop_y = min(length_pixels, start_y + stone_width)
                hf_raw[start_x:stop_x, start_y:stop_y] = np.random.choice(
                    stone_height_range
                )
                start_y += stone_width + stone_distance

            start_x += stone_width + stone_distance

    x1 = (width_pixels - platform_width) // 2
    x2 = (width_pixels + platform_width) // 2
    y1 = (length_pixels - platform_width) // 2
    y2 = (length_pixels + platform_width) // 2
    hf_raw[x1:x2, y1:y2] = 0

    return np.rint(hf_raw).astype(np.int16)


@height_field_to_mesh
def stairs_terrain(
    difficulty: float, cfg: hf_terrains_cfg.StairsTerrainCfg
) -> np.ndarray:
    step_height = cfg.step_height_range[0] + difficulty * (
        cfg.step_height_range[1] - cfg.step_height_range[0]
    )
    step_width = cfg.step_width_range[0] + difficulty * (
        cfg.step_width_range[1] - cfg.step_width_range[0]
    )

    if cfg.inverted:
        step_height *= -1

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)

    step_width = int(step_width / cfg.horizontal_scale)
    step_height = int(step_height / cfg.vertical_scale)

    platform_width = int(cfg.platform_width / cfg.horizontal_scale)

    hf_raw = np.zeros((width_pixels, length_pixels))

    current_step_height = 0
    start_x, start_y = 0, 0
    start_x = platform_width

    while start_x <= width_pixels - 2 * platform_width:
        current_step_height += step_height
        hf_raw[start_x : start_x + step_width, 0:length_pixels] = current_step_height
        start_x += step_width

    hf_raw[start_x : width_pixels - platform_width, 0:length_pixels] = (
        current_step_height
    )
    hf_raw[width_pixels - platform_width : width_pixels, 0:length_pixels] = 0

    return np.rint(hf_raw).astype(np.int16)


@height_field_to_mesh
def parkour_terrain(
    difficulty: float, cfg: hf_terrains_cfg.ParkourTerrainCfg
) -> np.ndarray:
    pit_depth_val = np.random.uniform(cfg.pit_depth[0], cfg.pit_depth[1])
    h_scale = cfg.horizontal_scale
    v_scale = cfg.vertical_scale
    start_x = cfg.start_x
    start_y = cfg.start_y
    num_goals = cfg.num_goals

    pit_depth_grid = -round(pit_depth_val / v_scale)

    length_y_grid = int(cfg.size[1] / h_scale)
    mid_y = length_y_grid // 2
    length_x_grid = int(cfg.size[0] / h_scale)

    hf_raw = np.zeros((length_x_grid, length_y_grid))

    stone_len = int(
        (
            (cfg.stone_len_range[0] - cfg.stone_len_range[1]) * difficulty
            + cfg.stone_len_range[1]
        )
        / h_scale
    )
    stone_width = int(
        (
            (cfg.stone_width_range[0] - cfg.stone_width_range[1]) * difficulty
            + cfg.stone_width_range[1]
        )
        / h_scale
    )
    gap_x = int(
        ((cfg.x_range[1] - cfg.x_range[0]) * difficulty + cfg.x_range[0]) / h_scale
    )
    gap_y = int(
        ((cfg.y_range[1] - cfg.y_range[0]) * difficulty + cfg.y_range[0]) / h_scale
    )
    platform_size_grid = int(cfg.platform_width / h_scale)
    incline_height_grid = int(cfg.incline_height / v_scale)

    hf_raw[
        start_x + platform_size_grid : start_x + length_x_grid,
        start_y : start_y + length_y_grid * 2,
    ] = pit_depth_grid

    dis_x = start_x + platform_size_grid - gap_x + stone_len // 2
    left_right_flag = np.random.randint(0, 2)

    for i in range(num_goals - 2):
        dis_x += gap_x
        pos_neg = 2 * (left_right_flag - 0.5)
        dis_y = mid_y + pos_neg * gap_y

        x_start = int(dis_x - stone_len // 2)
        x_end = x_start + stone_len
        y_start = int(dis_y - stone_width // 2)
        y_end = y_start + stone_width

        heights = (
            np.tile(
                np.linspace(-incline_height_grid, incline_height_grid, stone_width),
                (stone_len, 1),
            )
            * pos_neg
        )
        heights = heights.astype(int)

        if x_end > hf_raw.shape[0]:
            x_end = hf_raw.shape[0]
        if y_end > hf_raw.shape[1]:
            y_end = hf_raw.shape[1]

        actual_height = heights[: x_end - x_start, : y_end - y_start]
        hf_raw[x_start:x_end, y_start:y_end] = actual_height
        left_right_flag = 1 - left_right_flag

    final_dis_x = dis_x + gap_x
    return np.rint(hf_raw).astype(np.int16)


@height_field_to_mesh
def hurdle_terrain(
    difficulty: float, cfg: hf_terrains_cfg.HurdleTerrainCfg
) -> np.ndarray:
    x_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    y_pixels = int(cfg.size[1] / cfg.horizontal_scale)

    hf_raw = np.zeros((x_pixels, y_pixels))

    platform_size = int(cfg.platform_width / cfg.horizontal_scale)
    start_x = cfg.start_x
    start_y = cfg.start_y
    num_goals = cfg.num_goals

    mid_y = y_pixels // 2
    per_x = (x_pixels - platform_size) // num_goals

    hurdle_size = int(
        ((cfg.hurdle_range[1] - cfg.hurdle_range[0]) * difficulty + cfg.hurdle_range[0])
        / cfg.horizontal_scale
    )
    hurdle_height = int(
        (
            (cfg.hurdle_height_range[1] - cfg.hurdle_height_range[0]) * difficulty
            + cfg.hurdle_height_range[0]
        )
        / cfg.vertical_scale
    )

    flat_size = int(cfg.flat_size / cfg.horizontal_scale)
    dis_x = start_x + platform_size

    for i in range(num_goals):
        hf_raw[
            dis_x - hurdle_size // 2 : dis_x + hurdle_size // 2,
            start_y : start_y + mid_y * 2,
        ] = hurdle_height
        dis_x += flat_size + hurdle_size

    return np.rint(hf_raw).astype(np.int16)


@height_field_to_mesh
def bridge_terrain(
    difficulty: float, cfg: hf_terrains_cfg.BridgeTerrainCfg
) -> np.ndarray:
    x_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    y_pixels = int(cfg.size[1] / cfg.horizontal_scale)

    hf_raw = np.zeros((x_pixels, y_pixels))

    platform_size = int(cfg.platform_width / cfg.horizontal_scale)
    start_x = cfg.start_x
    start_y = cfg.start_y
    num_goals = cfg.num_goals

    mid_y = y_pixels // 2

    bridge_width = round(
        (
            (cfg.bridge_width_range[1] - cfg.bridge_width_range[0]) * difficulty
            + cfg.bridge_width_range[0]
        )
        / cfg.horizontal_scale
    )
    bridge_height = round(cfg.bridge_height / cfg.vertical_scale)

    hf_raw[start_x : start_x + platform_size, start_y : start_y + 2 * mid_y] = 0
    bridge_start_x = platform_size + start_x
    bridge_length = x_pixels
    bridge_end_x = start_x + bridge_length - platform_size

    left_y1 = 0
    left_y2 = int(mid_y - bridge_width // 2)
    right_y1 = int(mid_y + bridge_width // 2)
    right_y2 = mid_y * 2
    hf_raw[bridge_start_x:bridge_end_x, left_y1:left_y2] = -bridge_height
    hf_raw[bridge_start_x:bridge_end_x, right_y1:right_y2] = -bridge_height

    return np.rint(hf_raw).astype(np.int16)


@height_field_to_mesh
def platform_terrain(
    difficulty: float, cfg: hf_terrains_cfg.PlatformTerrainCfg
) -> np.ndarray:
    step_height = cfg.height_range[0] + difficulty * (
        cfg.height_range[1] - cfg.height_range[0]
    )
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)

    step_height = int(step_height / cfg.vertical_scale)

    platform_width = int(cfg.platform_width / cfg.horizontal_scale)
    flat_size = int(cfg.flat_size / cfg.horizontal_scale)

    hf_raw = np.zeros((width_pixels, length_pixels))

    start_x, start_y = 0, 0
    start_x = platform_width

    while start_x < (width_pixels - 2 * platform_width):
        hf_raw[start_x : start_x + flat_size, 0:length_pixels] = step_height
        start_x += flat_size * 2

    return np.rint(hf_raw).astype(np.int16)
