




from __future__ import annotations

from typing import TYPE_CHECKING

import isaacsim.core.utils.prims as prim_utils
import omni.kit.commands
import omni.log
from pxr import Sdf, Usd

from isaaclab.sim.utils import attach_stage_to_usd_context, clone
from isaaclab.utils import to_camel_case

if TYPE_CHECKING:
    from . import sensors_cfg


CUSTOM_PINHOLE_CAMERA_ATTRIBUTES = {
    "projection_type": ("cameraProjectionType", Sdf.ValueTypeNames.Token),
}
"""Custom attributes for pinhole camera model.

The dictionary maps the attribute name in the configuration to the attribute name in the USD prim.
"""


CUSTOM_FISHEYE_CAMERA_ATTRIBUTES = {
    "projection_type": ("cameraProjectionType", Sdf.ValueTypeNames.Token),
    "fisheye_nominal_width": ("fthetaWidth", Sdf.ValueTypeNames.Float),
    "fisheye_nominal_height": ("fthetaHeight", Sdf.ValueTypeNames.Float),
    "fisheye_optical_centre_x": ("fthetaCx", Sdf.ValueTypeNames.Float),
    "fisheye_optical_centre_y": ("fthetaCy", Sdf.ValueTypeNames.Float),
    "fisheye_max_fov": ("fthetaMaxFov", Sdf.ValueTypeNames.Float),
    "fisheye_polynomial_a": ("fthetaPolyA", Sdf.ValueTypeNames.Float),
    "fisheye_polynomial_b": ("fthetaPolyB", Sdf.ValueTypeNames.Float),
    "fisheye_polynomial_c": ("fthetaPolyC", Sdf.ValueTypeNames.Float),
    "fisheye_polynomial_d": ("fthetaPolyD", Sdf.ValueTypeNames.Float),
    "fisheye_polynomial_e": ("fthetaPolyE", Sdf.ValueTypeNames.Float),
    "fisheye_polynomial_f": ("fthetaPolyF", Sdf.ValueTypeNames.Float),
}
"""Custom attributes for fisheye camera model.

The dictionary maps the attribute name in the configuration to the attribute name in the USD prim.
"""


@clone
def spawn_camera(
    prim_path: str,
    cfg: sensors_cfg.PinholeCameraCfg | sensors_cfg.FisheyeCameraCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Create a USD camera prim with given projection type.

    The function creates various attributes on the camera prim that specify the camera's properties.
    These are later used by ``omni.replicator.core`` to render the scene with the given camera.

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        cfg: The configuration instance.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which case
            this is set to the origin.
        orientation: The orientation in (w, x, y, z) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case this is set to identity.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Returns:
        The created prim.

    Raises:
        ValueError: If a prim already exists at the given path.
    """

    if not prim_utils.is_prim_path_valid(prim_path):
        prim_utils.create_prim(prim_path, "Camera", translation=translation, orientation=orientation)
    else:
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")


    if cfg.lock_camera:


        attach_stage_to_usd_context(attaching_early=True)

        omni.kit.commands.execute(
            "ChangePropertyCommand",
            prop_path=Sdf.Path(f"{prim_path}.omni:kit:cameraLock"),
            value=True,
            prev=None,
            type_to_create_if_not_exist=Sdf.ValueTypeNames.Bool,
        )

    if cfg.projection_type == "pinhole":
        attribute_types = CUSTOM_PINHOLE_CAMERA_ATTRIBUTES
    else:
        attribute_types = CUSTOM_FISHEYE_CAMERA_ATTRIBUTES



    if cfg.horizontal_aperture_offset > 1e-4 or cfg.vertical_aperture_offset > 1e-4:
        omni.log.warn("Camera aperture offsets are not supported by Omniverse. These parameters will be ignored.")


    non_usd_cfg_param_names = [
        "func",
        "copy_from_source",
        "lock_camera",
        "visible",
        "semantic_tags",
        "from_intrinsic_matrix",
    ]

    prim = prim_utils.get_prim_at_path(prim_path)


    for attr_name, attr_type in attribute_types.values():

        if prim.GetAttribute(attr_name).Get() is None:

            prim.CreateAttribute(attr_name, attr_type)

    for param_name, param_value in cfg.__dict__.items():

        if param_value is None or param_name in non_usd_cfg_param_names:
            continue

        if param_name in attribute_types:

            prim_prop_name = attribute_types[param_name][0]
        else:

            prim_prop_name = to_camel_case(param_name, to="cC")

        prim.GetAttribute(prim_prop_name).Set(param_value)

    return prim_utils.get_prim_at_path(prim_path)
