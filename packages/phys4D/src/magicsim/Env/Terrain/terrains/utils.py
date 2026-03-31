from __future__ import annotations

import numpy as np
import torch
import trimesh

import warp as wp

from isaaclab.utils.warp import raycast_mesh


def color_meshes_by_height(meshes: list[trimesh.Trimesh], **kwargs) -> trimesh.Trimesh:
    """
    Color the vertices of a trimesh object based on the z-coordinate (height) of each vertex,
    using the Turbo colormap. If the z-coordinates are all the same, the vertices will be colored
    with a single color.

    Args:
        meshes: A list of trimesh objects.

    Keyword Args:
        color: A list of 3 integers in the range [0,255] representing the RGB
            color of the mesh. Used when the z-coordinates of all vertices are the same.
            Defaults to [172, 216, 230].
        color_map: The name of the color map to be used. Defaults to "turbo".

    Returns:
        A trimesh object with the vertices colored based on the z-coordinate (height) of each vertex.
    """

    mesh = trimesh.util.concatenate(meshes)

    heights = mesh.vertices[:, 2]

    if np.max(heights) == np.min(heights):
        color = kwargs.pop("color", (172, 216, 230))
        color = np.asarray(color, dtype=np.uint8)

        mesh.visual.vertex_colors = color
    else:
        heights_normalized = (heights - np.min(heights)) / (
            np.max(heights) - np.min(heights)
        )

        heights_normalized = np.clip(heights_normalized, 0.1, 0.9)

        color_map = kwargs.pop("color_map", "turbo")
        colors = trimesh.visual.color.interpolate(
            heights_normalized, color_map=color_map
        )

        mesh.visual.vertex_colors = colors

    return mesh


def create_prim_from_mesh(prim_path: str, mesh: trimesh.Trimesh, **kwargs):
    """Create a USD prim with mesh defined from vertices and triangles.

    The function creates a USD prim with a mesh defined from vertices and triangles. It performs the
    following steps:

    - Create a USD Xform prim at the path :obj:`prim_path`.
    - Create a USD prim with a mesh defined from the input vertices and triangles at the path :obj:`{prim_path}/mesh`.
    - Assign a physics material to the mesh at the path :obj:`{prim_path}/physicsMaterial`.
    - Assign a visual material to the mesh at the path :obj:`{prim_path}/visualMaterial`.

    Args:
        prim_path: The path to the primitive to be created.
        mesh: The mesh to be used for the primitive.

    Keyword Args:
        translation: The translation of the terrain. Defaults to None.
        orientation: The orientation of the terrain. Defaults to None.
        visual_material: The visual material to apply. Defaults to None.
        physics_material: The physics material to apply. Defaults to None.
    """

    import isaacsim.core.utils.prims as prim_utils
    from pxr import UsdGeom

    import isaaclab.sim as sim_utils

    prim_utils.create_prim(prim_path, "Xform")

    prim = prim_utils.create_prim(
        f"{prim_path}/mesh",
        "Mesh",
        translation=kwargs.get("translation"),
        orientation=kwargs.get("orientation"),
        attributes={
            "points": mesh.vertices,
            "faceVertexIndices": mesh.faces.flatten(),
            "faceVertexCounts": np.asarray([3] * len(mesh.faces)),
            "subdivisionScheme": "bilinear",
        },
    )

    collider_cfg = sim_utils.CollisionPropertiesCfg(collision_enabled=True)
    sim_utils.define_collision_properties(prim.GetPrimPath(), collider_cfg)

    if mesh.visual.vertex_colors is not None:
        rgba_colors = np.asarray(mesh.visual.vertex_colors).astype(np.float32) / 255.0

        color_prim_attr = prim.GetAttribute("primvars:displayColor")
        color_prim_var = UsdGeom.Primvar(color_prim_attr)
        color_prim_var.SetInterpolation(UsdGeom.Tokens.vertex)
        color_prim_attr.Set(rgba_colors[:, :3])

        display_prim_attr = prim.GetAttribute("primvars:displayOpacity")
        display_prim_var = UsdGeom.Primvar(display_prim_attr)
        display_prim_var.SetInterpolation(UsdGeom.Tokens.vertex)
        display_prim_var.Set(rgba_colors[:, 3])

    if kwargs.get("visual_material") is not None:
        visual_material_cfg: sim_utils.VisualMaterialCfg = kwargs.get("visual_material")

        visual_material_cfg.func(f"{prim_path}/visualMaterial", visual_material_cfg)
        sim_utils.bind_visual_material(
            prim.GetPrimPath(), f"{prim_path}/visualMaterial"
        )

    if kwargs.get("physics_material") is not None:
        physics_material_cfg: sim_utils.RigidBodyMaterialCfg = kwargs.get(
            "physics_material"
        )

        physics_material_cfg.func(f"{prim_path}/physicsMaterial", physics_material_cfg)
        sim_utils.bind_physics_material(
            prim.GetPrimPath(), f"{prim_path}/physicsMaterial"
        )


def find_flat_patches(
    wp_mesh: wp.Mesh,
    num_patches: int,
    patch_radius: float | list[float],
    origin: np.ndarray | torch.Tensor | tuple[float, float, float],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    max_height_diff: float,
) -> torch.Tensor:
    """Finds flat patches of given radius in the input mesh.

    The function finds flat patches of given radius based on the search space defined by the input ranges.
    The search space is characterized by origin in the mesh frame, and the x, y, and z ranges. The x and y
    ranges are used to sample points in the 2D region around the origin, and the z range is used to filter
    patches based on the height of the points.

    The function performs rejection sampling to find the patches based on the following steps:

    1. Sample patch locations in the 2D region around the origin.
    2. Define a ring of points around each patch location to query the height of the points using ray-casting.
    3. Reject patches that are outside the z range or have a height difference that is too large.
    4. Keep sampling until all patches are valid.

    Args:
        wp_mesh: The warp mesh to find patches in.
        num_patches: The desired number of patches to find.
        patch_radius: The radii used to form patches. If a list is provided, multiple patch sizes are checked.
            This is useful to deal with holes or other artifacts in the mesh.
        origin: The origin defining the center of the search space. This is specified in the mesh frame.
        x_range: The range of X coordinates to sample from.
        y_range: The range of Y coordinates to sample from.
        z_range: The range of valid Z coordinates used for filtering patches.
        max_height_diff: The maximum allowable distance between the lowest and highest points
            on a patch to consider it as valid. If the difference is greater than this value,
            the patch is rejected.

    Returns:
        A tensor of shape (num_patches, 3) containing the flat patches. The patches are defined in the mesh frame.

    Raises:
        RuntimeError: If the function fails to find valid patches. This can happen if the input parameters
            are not suitable for finding valid patches and maximum number of iterations is reached.
    """

    device = wp.device_to_torch(wp_mesh.device)

    if isinstance(patch_radius, float):
        patch_radius = [patch_radius]

    if isinstance(origin, np.ndarray):
        origin = torch.from_numpy(origin).to(torch.float).to(device)
    elif isinstance(origin, torch.Tensor):
        origin = origin.to(device)
    else:
        origin = torch.tensor(origin, dtype=torch.float, device=device)

    x_range = (
        max(x_range[0] + origin[0].item(), wp_mesh.points.numpy()[:, 0].min()),
        min(x_range[1] + origin[0].item(), wp_mesh.points.numpy()[:, 0].max()),
    )
    y_range = (
        max(y_range[0] + origin[1].item(), wp_mesh.points.numpy()[:, 1].min()),
        min(y_range[1] + origin[1].item(), wp_mesh.points.numpy()[:, 1].max()),
    )
    z_range = (
        z_range[0] + origin[2].item(),
        z_range[1] + origin[2].item(),
    )

    angle = torch.linspace(0, 2 * np.pi, 10, device=device)
    query_x = []
    query_y = []
    for radius in patch_radius:
        query_x.append(radius * torch.cos(angle))
        query_y.append(radius * torch.sin(angle))
    query_x = torch.cat(query_x).unsqueeze(1)
    query_y = torch.cat(query_y).unsqueeze(1)

    query_points = torch.cat([query_x, query_y, torch.zeros_like(query_x)], dim=-1)

    points_ids = torch.arange(num_patches, device=device)

    flat_patches = torch.zeros(num_patches, 3, device=device)

    iter_count = 0
    while len(points_ids) > 0 and iter_count < 10000:
        pos_x = torch.empty(len(points_ids), device=device).uniform_(*x_range)
        pos_y = torch.empty(len(points_ids), device=device).uniform_(*y_range)
        flat_patches[points_ids, :2] = torch.stack([pos_x, pos_y], dim=-1)

        points = flat_patches[points_ids].unsqueeze(1) + query_points
        points[..., 2] = 100.0

        dirs = torch.zeros_like(points)
        dirs[..., 2] = -1.0

        ray_hits = raycast_mesh(points.view(-1, 3), dirs.view(-1, 3), wp_mesh)[0]
        heights = ray_hits.view(points.shape)[..., 2]

        flat_patches[points_ids, 2] = heights[..., -1]

        not_valid = torch.any(
            torch.logical_or(heights < z_range[0], heights > z_range[1]), dim=1
        )

        not_valid = torch.logical_or(
            not_valid, (heights.max(dim=1)[0] - heights.min(dim=1)[0]) > max_height_diff
        )

        points_ids = points_ids[not_valid]

        iter_count += 1

    if len(points_ids) > 0:
        raise RuntimeError(
            "Failed to find valid patches! Please check the input parameters."
            f"\n\tMaximum number of iterations reached: {iter_count}"
            f"\n\tNumber of invalid patches: {len(points_ids)}"
            f"\n\tMaximum height difference: {max_height_diff}"
        )

    return flat_patches - origin
