




from __future__ import annotations

from typing import TYPE_CHECKING

import isaacsim.core.utils.prims as prim_utils
from isaacsim.core.utils.stage import get_current_stage
from pxr import PhysxSchema, Usd, UsdPhysics, UsdShade

from isaaclab.sim.utils import clone, safe_set_attribute_on_usd_schema

if TYPE_CHECKING:
    from . import physics_materials_cfg


@clone
def spawn_rigid_body_material(prim_path: str, cfg: physics_materials_cfg.RigidBodyMaterialCfg) -> Usd.Prim:
    """Create material with rigid-body physics properties.

    Rigid body materials are used to define the physical properties to meshes of a rigid body. These
    include the friction, restitution, and their respective combination modes. For more information on
    rigid body material, please refer to the `documentation on PxMaterial <https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/classPxBaseMaterial.html>`_.

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        cfg: The configuration for the physics material.

    Returns:
        The spawned rigid body material prim.

    Raises:
        ValueError:  When a prim already exists at the specified prim path and is not a material.
    """

    stage = get_current_stage()


    if not prim_utils.is_prim_path_valid(prim_path):
        _ = UsdShade.Material.Define(stage, prim_path)


    prim = prim_utils.get_prim_at_path(prim_path)

    if not prim.IsA(UsdShade.Material):
        raise ValueError(f"A prim already exists at path: '{prim_path}' but is not a material.")

    usd_physics_material_api = UsdPhysics.MaterialAPI(prim)
    if not usd_physics_material_api:
        usd_physics_material_api = UsdPhysics.MaterialAPI.Apply(prim)

    physx_material_api = PhysxSchema.PhysxMaterialAPI(prim)
    if not physx_material_api:
        physx_material_api = PhysxSchema.PhysxMaterialAPI.Apply(prim)


    cfg = cfg.to_dict()
    del cfg["func"]

    for attr_name in ["static_friction", "dynamic_friction", "restitution"]:
        value = cfg.pop(attr_name, None)
        safe_set_attribute_on_usd_schema(usd_physics_material_api, attr_name, value, camel_case=True)

    for attr_name, value in cfg.items():
        safe_set_attribute_on_usd_schema(physx_material_api, attr_name, value, camel_case=True)

    return prim


@clone
def spawn_deformable_body_material(prim_path: str, cfg: physics_materials_cfg.DeformableBodyMaterialCfg) -> Usd.Prim:
    """Create material with deformable-body physics properties.

    Deformable body materials are used to define the physical properties to meshes of a deformable body. These
    include the friction and deformable body properties. For more information on deformable body material,
    please refer to the documentation on `PxFEMSoftBodyMaterial`_.

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        cfg: The configuration for the physics material.

    Returns:
        The spawned deformable body material prim.

    Raises:
        ValueError:  When a prim already exists at the specified prim path and is not a material.

    .. _PxFEMSoftBodyMaterial: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/structPxFEMSoftBodyMaterialModel.html
    """

    stage = get_current_stage()


    if not prim_utils.is_prim_path_valid(prim_path):
        _ = UsdShade.Material.Define(stage, prim_path)


    prim = prim_utils.get_prim_at_path(prim_path)

    if not prim.IsA(UsdShade.Material):
        raise ValueError(f"A prim already exists at path: '{prim_path}' but is not a material.")

    physx_deformable_body_material_api = PhysxSchema.PhysxDeformableBodyMaterialAPI(prim)
    if not physx_deformable_body_material_api:
        physx_deformable_body_material_api = PhysxSchema.PhysxDeformableBodyMaterialAPI.Apply(prim)


    cfg = cfg.to_dict()
    del cfg["func"]

    for attr_name, value in cfg.items():
        safe_set_attribute_on_usd_schema(physx_deformable_body_material_api, attr_name, value, camel_case=True)

    return prim
