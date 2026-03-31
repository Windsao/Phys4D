




from __future__ import annotations

from typing import TYPE_CHECKING

import isaacsim.core.utils.prims as prim_utils
import omni.kit.commands
import omni.log
from pxr import Gf, Sdf, Usd


try:
    import Semantics
except ModuleNotFoundError:
    from pxr import Semantics

from isaacsim.core.utils.stage import get_current_stage

from isaaclab.sim import converters, schemas
from isaaclab.sim.utils import (
    bind_physics_material,
    bind_visual_material,
    clone,
    is_current_stage_in_memory,
    select_usd_variants,
)
from isaaclab.utils.assets import check_usd_path_with_timeout

if TYPE_CHECKING:
    from . import from_files_cfg


@clone
def spawn_from_usd(
    prim_path: str,
    cfg: from_files_cfg.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn an asset from a USD file and override the settings with the given config.

    In the case of a USD file, the asset is spawned at the default prim specified in the USD file.
    If a default prim is not specified, then the asset is spawned at the root prim.

    In case a prim already exists at the given prim path, then the function does not create a new prim
    or throw an error that the prim already exists. Instead, it just takes the existing prim and overrides
    the settings with the given config.

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        cfg: The configuration instance.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which
            case the translation specified in the USD file is used.
        orientation: The orientation in (w, x, y, z) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case the orientation specified in the USD file is used.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Returns:
        The prim of the spawned asset.

    Raises:
        FileNotFoundError: If the USD file does not exist at the given path.
    """

    return _spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation)


@clone
def spawn_from_urdf(
    prim_path: str,
    cfg: from_files_cfg.UrdfFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn an asset from a URDF file and override the settings with the given config.

    It uses the :class:`UrdfConverter` class to create a USD file from URDF. This file is then imported
    at the specified prim path.

    In case a prim already exists at the given prim path, then the function does not create a new prim
    or throw an error that the prim already exists. Instead, it just takes the existing prim and overrides
    the settings with the given config.

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        cfg: The configuration instance.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which
            case the translation specified in the generated USD file is used.
        orientation: The orientation in (w, x, y, z) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case the orientation specified in the generated USD file is used.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Returns:
        The prim of the spawned asset.

    Raises:
        FileNotFoundError: If the URDF file does not exist at the given path.
    """

    urdf_loader = converters.UrdfConverter(cfg)

    return _spawn_from_usd_file(prim_path, urdf_loader.usd_path, cfg, translation, orientation)


def spawn_ground_plane(
    prim_path: str,
    cfg: from_files_cfg.GroundPlaneCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawns a ground plane into the scene.

    This function loads the USD file containing the grid plane asset from Isaac Sim. It may
    not work with other assets for ground planes. In those cases, please use the `spawn_from_usd`
    function.

    Note:
        This function takes keyword arguments to be compatible with other spawners. However, it does not
        use any of the kwargs.

    Args:
        prim_path: The path to spawn the asset at.
        cfg: The configuration instance.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which
            case the translation specified in the USD file is used.
        orientation: The orientation in (w, x, y, z) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case the orientation specified in the USD file is used.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Returns:
        The prim of the spawned asset.

    Raises:
        ValueError: If the prim path already exists.
    """

    if not prim_utils.is_prim_path_valid(prim_path):
        prim_utils.create_prim(prim_path, usd_path=cfg.usd_path, translation=translation, orientation=orientation)
    else:
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")


    if cfg.physics_material is not None:
        cfg.physics_material.func(f"{prim_path}/physicsMaterial", cfg.physics_material)

        collision_prim_path = prim_utils.get_prim_path(
            prim_utils.get_first_matching_child_prim(
                prim_path, predicate=lambda x: prim_utils.get_prim_type_name(x) == "Plane"
            )
        )
        bind_physics_material(collision_prim_path, f"{prim_path}/physicsMaterial")



    if prim_utils.is_prim_path_valid(f"{prim_path}/Environment"):

        scale = (cfg.size[0] / 100.0, cfg.size[1] / 100.0, 1.0)

        prim_utils.set_prim_property(f"{prim_path}/Environment", "xformOp:scale", scale)



    if cfg.color is not None:


        if is_current_stage_in_memory():
            omni.log.warn(
                "Ground plane color modification is not supported while the stage is in memory. Skipping operation."
            )

        else:
            prop_path = f"{prim_path}/Looks/theGrid/Shader.inputs:diffuse_tint"


            omni.kit.commands.execute(
                "ChangePropertyCommand",
                prop_path=Sdf.Path(prop_path),
                value=Gf.Vec3f(*cfg.color),
                prev=None,
                type_to_create_if_not_exist=Sdf.ValueTypeNames.Color3f,
            )


    stage = get_current_stage()
    omni.kit.commands.execute("ToggleVisibilitySelectedPrims", selected_paths=[f"{prim_path}/SphereLight"], stage=stage)

    prim = prim_utils.get_prim_at_path(prim_path)

    if hasattr(cfg, "semantic_tags") and cfg.semantic_tags is not None:

        for semantic_type, semantic_value in cfg.semantic_tags:

            semantic_type_sanitized = semantic_type.replace(" ", "_")
            semantic_value_sanitized = semantic_value.replace(" ", "_")

            instance_name = f"{semantic_type_sanitized}_{semantic_value_sanitized}"
            sem = Semantics.SemanticsAPI.Apply(prim, instance_name)

            sem.CreateSemanticTypeAttr().Set(semantic_type)
            sem.CreateSemanticDataAttr().Set(semantic_value)


    prim_utils.set_prim_visibility(prim, cfg.visible)


    return prim


"""
Helper functions.
"""


def _spawn_from_usd_file(
    prim_path: str,
    usd_path: str,
    cfg: from_files_cfg.FileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn an asset from a USD file and override the settings with the given config.

    In case a prim already exists at the given prim path, then the function does not create a new prim
    or throw an error that the prim already exists. Instead, it just takes the existing prim and overrides
    the settings with the given config.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        usd_path: The path to the USD file to spawn the asset from.
        cfg: The configuration instance.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which
            case the translation specified in the generated USD file is used.
        orientation: The orientation in (w, x, y, z) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case the orientation specified in the generated USD file is used.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Returns:
        The prim of the spawned asset.

    Raises:
        FileNotFoundError: If the USD file does not exist at the given path.
    """

    if not check_usd_path_with_timeout(usd_path):
        if "4.5" in usd_path:
            usd_5_0_path = usd_path.replace("http", "https").replace("/4.5", "/5.0")
            if not check_usd_path_with_timeout(usd_5_0_path):
                raise FileNotFoundError(f"USD file not found at path at either: '{usd_path}' or '{usd_5_0_path}'.")
            usd_path = usd_5_0_path
        else:
            raise FileNotFoundError(f"USD file not found at path at: '{usd_path}'.")


    if not prim_utils.is_prim_path_valid(prim_path):

        prim_utils.create_prim(
            prim_path,
            usd_path=usd_path,
            translation=translation,
            orientation=orientation,
            scale=cfg.scale,
        )
    else:
        omni.log.warn(f"A prim already exists at prim path: '{prim_path}'.")


    if hasattr(cfg, "variants") and cfg.variants is not None:
        select_usd_variants(prim_path, cfg.variants)


    if cfg.rigid_props is not None:
        schemas.modify_rigid_body_properties(prim_path, cfg.rigid_props)

    if cfg.collision_props is not None:
        schemas.modify_collision_properties(prim_path, cfg.collision_props)

    if cfg.mass_props is not None:
        schemas.modify_mass_properties(prim_path, cfg.mass_props)


    if cfg.articulation_props is not None:
        schemas.modify_articulation_root_properties(prim_path, cfg.articulation_props)

    if cfg.fixed_tendons_props is not None:
        schemas.modify_fixed_tendon_properties(prim_path, cfg.fixed_tendons_props)
    if cfg.spatial_tendons_props is not None:
        schemas.modify_spatial_tendon_properties(prim_path, cfg.spatial_tendons_props)



    if cfg.joint_drive_props is not None:
        schemas.modify_joint_drive_properties(prim_path, cfg.joint_drive_props)


    if cfg.deformable_props is not None:
        schemas.modify_deformable_body_properties(prim_path, cfg.deformable_props)


    if cfg.visual_material is not None:
        if not cfg.visual_material_path.startswith("/"):
            material_path = f"{prim_path}/{cfg.visual_material_path}"
        else:
            material_path = cfg.visual_material_path

        cfg.visual_material.func(material_path, cfg.visual_material)

        bind_visual_material(prim_path, material_path)


    return prim_utils.get_prim_at_path(prim_path)
