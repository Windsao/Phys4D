




"""Sub-module with USD-related utilities."""

from __future__ import annotations

import contextlib
import functools
import inspect
import re
from collections.abc import Callable, Generator
from typing import TYPE_CHECKING, Any

import carb
import isaacsim.core.utils.stage as stage_utils
import omni
import omni.kit.commands
import omni.log
from isaacsim.core.cloner import Cloner
from isaacsim.core.utils.carb import get_carb_setting
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.version import get_version
from pxr import PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils


try:
    import Semantics
except ModuleNotFoundError:
    from pxr import Semantics

from isaaclab.utils.string import to_camel_case

from . import schemas

if TYPE_CHECKING:
    from .spawners.spawner_cfg import SpawnerCfg

"""
Attribute - Setters.
"""


def safe_set_attribute_on_usd_schema(schema_api: Usd.APISchemaBase, name: str, value: Any, camel_case: bool):
    """Set the value of an attribute on its USD schema if it exists.

    A USD API schema serves as an interface or API for authoring and extracting a set of attributes.
    They typically derive from the :class:`pxr.Usd.SchemaBase` class. This function checks if the
    attribute exists on the schema and sets the value of the attribute if it exists.

    Args:
        schema_api: The USD schema to set the attribute on.
        name: The name of the attribute.
        value: The value to set the attribute to.
        camel_case: Whether to convert the attribute name to camel case.

    Raises:
        TypeError: When the input attribute name does not exist on the provided schema API.
    """

    if value is None:
        return

    if camel_case:
        attr_name = to_camel_case(name, to="CC")
    else:
        attr_name = name


    attr = getattr(schema_api, f"Create{attr_name}Attr", None)

    if attr is not None:
        attr().Set(value)
    else:


        omni.log.error(f"Attribute '{attr_name}' does not exist on prim '{schema_api.GetPath()}'.")
        raise TypeError(f"Attribute '{attr_name}' does not exist on prim '{schema_api.GetPath()}'.")


def safe_set_attribute_on_usd_prim(prim: Usd.Prim, attr_name: str, value: Any, camel_case: bool):
    """Set the value of a attribute on its USD prim.

    The function creates a new attribute if it does not exist on the prim. This is because in some cases (such
    as with shaders), their attributes are not exposed as USD prim properties that can be altered. This function
    allows us to set the value of the attributes in these cases.

    Args:
        prim: The USD prim to set the attribute on.
        attr_name: The name of the attribute.
        value: The value to set the attribute to.
        camel_case: Whether to convert the attribute name to camel case.
    """

    if value is None:
        return

    if camel_case:
        attr_name = to_camel_case(attr_name, to="cC")

    if isinstance(value, bool):
        sdf_type = Sdf.ValueTypeNames.Bool
    elif isinstance(value, int):
        sdf_type = Sdf.ValueTypeNames.Int
    elif isinstance(value, float):
        sdf_type = Sdf.ValueTypeNames.Float
    elif isinstance(value, (tuple, list)) and len(value) == 3 and any(isinstance(v, float) for v in value):
        sdf_type = Sdf.ValueTypeNames.Float3
    elif isinstance(value, (tuple, list)) and len(value) == 2 and any(isinstance(v, float) for v in value):
        sdf_type = Sdf.ValueTypeNames.Float2
    else:
        raise NotImplementedError(
            f"Cannot set attribute '{attr_name}' with value '{value}'. Please modify the code to support this type."
        )



    attach_stage_to_usd_context(attaching_early=True)


    omni.kit.commands.execute(
        "ChangePropertyCommand",
        prop_path=Sdf.Path(f"{prim.GetPath()}.{attr_name}"),
        value=value,
        prev=None,
        type_to_create_if_not_exist=sdf_type,
        usd_context_name=prim.GetStage(),
    )


"""
Decorators.
"""


def apply_nested(func: Callable) -> Callable:
    """Decorator to apply a function to all prims under a specified prim-path.

    The function iterates over the provided prim path and all its children to apply input function
    to all prims under the specified prim path.

    If the function succeeds to apply to a prim, it will not look at the children of that prim.
    This is based on the physics behavior that nested schemas are not allowed. For example, a parent prim
    and its child prim cannot both have a rigid-body schema applied on them, or it is not possible to
    have nested articulations.

    While traversing the prims under the specified prim path, the function will throw a warning if it
    does not succeed to apply the function to any prim. This is because the user may have intended to
    apply the function to a prim that does not have valid attributes, or the prim may be an instanced prim.

    Args:
        func: The function to apply to all prims under a specified prim-path. The function
            must take the prim-path and other arguments. It should return a boolean indicating whether
            the function succeeded or not.

    Returns:
        The wrapped function that applies the function to all prims under a specified prim-path.

    Raises:
        ValueError: If the prim-path does not exist on the stage.
    """

    @functools.wraps(func)
    def wrapper(prim_path: str | Sdf.Path, *args, **kwargs):


        sig = inspect.signature(func)
        bound_args = sig.bind(prim_path, *args, **kwargs)

        stage = bound_args.arguments.get("stage")
        if stage is None:
            stage = get_current_stage()


        prim: Usd.Prim = stage.GetPrimAtPath(prim_path)

        if not prim.IsValid():
            raise ValueError(f"Prim at path '{prim_path}' is not valid.")

        count_success = 0
        instanced_prim_paths = []

        all_prims = [prim]
        while len(all_prims) > 0:

            child_prim = all_prims.pop(0)
            child_prim_path = child_prim.GetPath().pathString

            if child_prim.IsInstance():
                instanced_prim_paths.append(child_prim_path)
                continue

            success = func(child_prim_path, *args, **kwargs)


            if not success:
                all_prims += child_prim.GetChildren()
            else:
                count_success += 1

        if count_success == 0:
            omni.log.warn(
                f"Could not perform '{func.__name__}' on any prims under: '{prim_path}'."
                " This might be because of the following reasons:"
                "\n\t(1) The desired attribute does not exist on any of the prims."
                "\n\t(2) The desired attribute exists on an instanced prim."
                f"\n\t\tDiscovered list of instanced prim paths: {instanced_prim_paths}"
            )

    return wrapper


def clone(func: Callable) -> Callable:
    """Decorator for cloning a prim based on matching prim paths of the prim's parent.

    The decorator checks if the parent prim path matches any prim paths in the stage. If so, it clones the
    spawned prim at each matching prim path. For example, if the input prim path is: ``/World/Table_[0-9]/Bottle``,
    the decorator will clone the prim at each matching prim path of the parent prim: ``/World/Table_0/Bottle``,
    ``/World/Table_1/Bottle``, etc.

    Note:
        For matching prim paths, the decorator assumes that valid prims exist for all matching prim paths.
        In case no matching prim paths are found, the decorator raises a ``RuntimeError``.

    Args:
        func: The function to decorate.

    Returns:
        The decorated function that spawns the prim and clones it at each matching prim path.
        It returns the spawned source prim, i.e., the first prim in the list of matching prim paths.
    """

    @functools.wraps(func)
    def wrapper(prim_path: str | Sdf.Path, cfg: SpawnerCfg, *args, **kwargs):

        stage = get_current_stage()


        prim_path = str(prim_path)

        if not prim_path.startswith("/"):
            raise ValueError(f"Prim path '{prim_path}' is not global. It must start with '/'.")


        root_path, asset_path = prim_path.rsplit("/", 1)


        is_regex_expression = re.match(r"^[a-zA-Z0-9/_]+$", root_path) is None


        if is_regex_expression and root_path != "":
            source_prim_paths = find_matching_prim_paths(root_path)

            if len(source_prim_paths) == 0:
                raise RuntimeError(
                    f"Unable to find source prim path: '{root_path}'. Please create the prim before spawning."
                )
        else:
            source_prim_paths = [root_path]


        prim_paths = [f"{source_prim_path}/{asset_path}" for source_prim_path in source_prim_paths]

        prim = func(prim_paths[0], cfg, *args, **kwargs)

        if hasattr(cfg, "visible"):
            imageable = UsdGeom.Imageable(prim)
            if cfg.visible:
                imageable.MakeVisible()
            else:
                imageable.MakeInvisible()

        if hasattr(cfg, "semantic_tags") and cfg.semantic_tags is not None:

            for semantic_type, semantic_value in cfg.semantic_tags:

                semantic_type_sanitized = semantic_type.replace(" ", "_")
                semantic_value_sanitized = semantic_value.replace(" ", "_")

                instance_name = f"{semantic_type_sanitized}_{semantic_value_sanitized}"
                sem = Semantics.SemanticsAPI.Apply(prim, instance_name)

                sem.CreateSemanticTypeAttr()
                sem.CreateSemanticDataAttr()
                sem.GetSemanticTypeAttr().Set(semantic_type)
                sem.GetSemanticDataAttr().Set(semantic_value)

        if hasattr(cfg, "activate_contact_sensors") and cfg.activate_contact_sensors:
            schemas.activate_contact_sensors(prim_paths[0], cfg.activate_contact_sensors)

        if len(prim_paths) > 1:
            cloner = Cloner(stage=stage)

            isaac_sim_version = float(".".join(get_version()[2]))
            if isaac_sim_version < 5:

                cloner.clone(
                    prim_paths[0], prim_paths[1:], replicate_physics=False, copy_from_source=cfg.copy_from_source
                )
            else:

                clone_in_fabric = kwargs.get("clone_in_fabric", False)
                replicate_physics = kwargs.get("replicate_physics", False)
                cloner.clone(
                    prim_paths[0],
                    prim_paths[1:],
                    replicate_physics=replicate_physics,
                    copy_from_source=cfg.copy_from_source,
                    clone_in_fabric=clone_in_fabric,
                )

        return prim

    return wrapper


"""
Material bindings.
"""


@apply_nested
def bind_visual_material(
    prim_path: str | Sdf.Path,
    material_path: str | Sdf.Path,
    stage: Usd.Stage | None = None,
    stronger_than_descendants: bool = True,
):
    """Bind a visual material to a prim.

    This function is a wrapper around the USD command `BindMaterialCommand`_.

    .. note::
        The function is decorated with :meth:`apply_nested` to allow applying the function to a prim path
        and all its descendants.

    .. _BindMaterialCommand: https://docs.omniverse.nvidia.com/kit/docs/omni.usd/latest/omni.usd.commands/omni.usd.commands.BindMaterialCommand.html

    Args:
        prim_path: The prim path where to apply the material.
        material_path: The prim path of the material to apply.
        stage: The stage where the prim and material exist.
            Defaults to None, in which case the current stage is used.
        stronger_than_descendants: Whether the material should override the material of its descendants.
            Defaults to True.

    Raises:
        ValueError: If the provided prim paths do not exist on stage.
    """

    if stage is None:
        stage = get_current_stage()


    if not stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"Target prim '{material_path}' does not exist.")
    if not stage.GetPrimAtPath(material_path).IsValid():
        raise ValueError(f"Visual material '{material_path}' does not exist.")


    if stronger_than_descendants:
        binding_strength = "strongerThanDescendants"
    else:
        binding_strength = "weakerThanDescendants"


    success, _ = omni.kit.commands.execute(
        "BindMaterialCommand",
        prim_path=prim_path,
        material_path=material_path,
        strength=binding_strength,
        stage=stage,
    )

    return success


@apply_nested
def bind_physics_material(
    prim_path: str | Sdf.Path,
    material_path: str | Sdf.Path,
    stage: Usd.Stage | None = None,
    stronger_than_descendants: bool = True,
):
    """Bind a physics material to a prim.

    `Physics material`_ can be applied only to a prim with physics-enabled on them. This includes having
    collision APIs, or deformable body APIs, or being a particle system. In case the prim does not have
    any of these APIs, the function will not apply the material and return False.

    .. note::
        The function is decorated with :meth:`apply_nested` to allow applying the function to a prim path
        and all its descendants.

    .. _Physics material: https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sim.html#isaaclab.sim.SimulationCfg.physics_material

    Args:
        prim_path: The prim path where to apply the material.
        material_path: The prim path of the material to apply.
        stage: The stage where the prim and material exist.
            Defaults to None, in which case the current stage is used.
        stronger_than_descendants: Whether the material should override the material of its descendants.
            Defaults to True.

    Raises:
        ValueError: If the provided prim paths do not exist on stage.
    """

    if stage is None:
        stage = get_current_stage()


    if not stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"Target prim '{material_path}' does not exist.")
    if not stage.GetPrimAtPath(material_path).IsValid():
        raise ValueError(f"Physics material '{material_path}' does not exist.")

    prim = stage.GetPrimAtPath(prim_path)

    has_physics_scene_api = prim.HasAPI(PhysxSchema.PhysxSceneAPI)
    has_collider = prim.HasAPI(UsdPhysics.CollisionAPI)
    has_deformable_body = prim.HasAPI(PhysxSchema.PhysxDeformableBodyAPI)
    has_particle_system = prim.IsA(PhysxSchema.PhysxParticleSystem)
    if not (has_physics_scene_api or has_collider or has_deformable_body or has_particle_system):
        omni.log.verbose(
            f"Cannot apply physics material '{material_path}' on prim '{prim_path}'. It is neither a"
            " PhysX scene, collider, a deformable body, nor a particle system."
        )
        return False


    if prim.HasAPI(UsdShade.MaterialBindingAPI):
        material_binding_api = UsdShade.MaterialBindingAPI(prim)
    else:
        material_binding_api = UsdShade.MaterialBindingAPI.Apply(prim)


    material = UsdShade.Material(stage.GetPrimAtPath(material_path))

    if stronger_than_descendants:
        binding_strength = UsdShade.Tokens.strongerThanDescendants
    else:
        binding_strength = UsdShade.Tokens.weakerThanDescendants

    material_binding_api.Bind(material, bindingStrength=binding_strength, materialPurpose="physics")

    return True


"""
Exporting.
"""


def export_prim_to_file(
    path: str | Sdf.Path,
    source_prim_path: str | Sdf.Path,
    target_prim_path: str | Sdf.Path | None = None,
    stage: Usd.Stage | None = None,
):
    """Exports a prim from a given stage to a USD file.

    The function creates a new layer at the provided path and copies the prim to the layer.
    It sets the copied prim as the default prim in the target layer. Additionally, it updates
    the stage up-axis and meters-per-unit to match the current stage.

    Args:
        path: The filepath path to export the prim to.
        source_prim_path: The prim path to export.
        target_prim_path: The prim path to set as the default prim in the target layer.
            Defaults to None, in which case the source prim path is used.
        stage: The stage where the prim exists. Defaults to None, in which case the
            current stage is used.

    Raises:
        ValueError: If the prim paths are not global (i.e: do not start with '/').
    """

    if stage is None:
        stage = get_current_stage()



    path = str(path)
    source_prim_path = str(source_prim_path)
    if target_prim_path is not None:
        target_prim_path = str(target_prim_path)

    if not source_prim_path.startswith("/"):
        raise ValueError(f"Source prim path '{source_prim_path}' is not global. It must start with '/'.")
    if target_prim_path is not None and not target_prim_path.startswith("/"):
        raise ValueError(f"Target prim path '{target_prim_path}' is not global. It must start with '/'.")


    source_layer = stage.GetRootLayer()


    target_layer = Sdf.Find(path)
    if target_layer is None:
        target_layer = Sdf.Layer.CreateNew(path)

    target_stage = Usd.Stage.Open(target_layer)


    UsdGeom.SetStageUpAxis(target_stage, UsdGeom.GetStageUpAxis(stage))
    UsdGeom.SetStageMetersPerUnit(target_stage, UsdGeom.GetStageMetersPerUnit(stage))


    source_prim_path = Sdf.Path(source_prim_path)
    if target_prim_path is None:
        target_prim_path = source_prim_path


    Sdf.CreatePrimInLayer(target_layer, target_prim_path)
    Sdf.CopySpec(source_layer, source_prim_path, target_layer, target_prim_path)

    target_layer.defaultPrim = Sdf.Path(target_prim_path).name

    omni.usd.resolve_paths(source_layer.identifier, target_layer.identifier)

    target_layer.Save()


"""
USD Prim properties.
"""


def make_uninstanceable(prim_path: str | Sdf.Path, stage: Usd.Stage | None = None):
    """Check if a prim and its descendants are instanced and make them uninstanceable.

    This function checks if the prim at the specified prim path and its descendants are instanced.
    If so, it makes the respective prim uninstanceable by disabling instancing on the prim.

    This is useful when we want to modify the properties of a prim that is instanced. For example, if we
    want to apply a different material on an instanced prim, we need to make the prim uninstanceable first.

    Args:
        prim_path: The prim path to check.
        stage: The stage where the prim exists. Defaults to None, in which case the current stage is used.

    Raises:
        ValueError: If the prim path is not global (i.e: does not start with '/').
    """

    if stage is None:
        stage = get_current_stage()


    prim_path = str(prim_path)

    if not prim_path.startswith("/"):
        raise ValueError(f"Prim path '{prim_path}' is not global. It must start with '/'.")

    prim: Usd.Prim = stage.GetPrimAtPath(prim_path)

    if not prim.IsValid():
        raise ValueError(f"Prim at path '{prim_path}' is not valid.")

    all_prims = [prim]
    while len(all_prims) > 0:

        child_prim = all_prims.pop(0)

        if child_prim.IsInstance():

            child_prim.SetInstanceable(False)

        all_prims += child_prim.GetFilteredChildren(Usd.TraverseInstanceProxies())


def resolve_prim_pose(
    prim: Usd.Prim, ref_prim: Usd.Prim | None = None
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Resolve the pose of a prim with respect to another prim.

    Note:
        This function ignores scale and skew by orthonormalizing the transformation
        matrix at the final step. However, if any ancestor prim in the hierarchy
        has non-uniform scale, that scale will still affect the resulting position
        and orientation of the prim (because it's baked into the transform before
        scale removal).

        In other words: scale **is not removed hierarchically**. If you need
        completely scale-free poses, you must walk the transform chain and strip
        scale at each level. Please open an issue if you need this functionality.

    Args:
        prim: The USD prim to resolve the pose for.
        ref_prim: The USD prim to compute the pose with respect to.
            Defaults to None, in which case the world frame is used.

    Returns:
        A tuple containing the position (as a 3D vector) and the quaternion orientation
        in the (w, x, y, z) format.

    Raises:
        ValueError: If the prim or ref prim is not valid.
    """

    if not prim.IsValid():
        raise ValueError(f"Prim at path '{prim.GetPath().pathString}' is not valid.")

    xform = UsdGeom.Xformable(prim)
    prim_tf = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())


    prim_tf = prim_tf.GetOrthonormalized()

    if ref_prim is not None:

        if not ref_prim.IsValid():
            raise ValueError(f"Ref prim at path '{ref_prim.GetPath().pathString}' is not valid.")

        ref_xform = UsdGeom.Xformable(ref_prim)
        ref_tf = ref_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

        ref_tf = ref_tf.GetOrthonormalized()

        prim_tf = prim_tf * ref_tf.GetInverse()


    prim_pos = [*prim_tf.ExtractTranslation()]
    prim_quat = [prim_tf.ExtractRotationQuat().real, *prim_tf.ExtractRotationQuat().imaginary]
    return tuple(prim_pos), tuple(prim_quat)


def resolve_prim_scale(prim: Usd.Prim) -> tuple[float, float, float]:
    """Resolve the scale of a prim in the world frame.

    At an attribute level, a USD prim's scale is a scaling transformation applied to the prim with
    respect to its parent prim. This function resolves the scale of the prim in the world frame,
    by computing the local to world transform of the prim. This is equivalent to traversing up
    the prim hierarchy and accounting for the rotations and scales of the prims.

    For instance, if a prim has a scale of (1, 2, 3) and it is a child of a prim with a scale of (4, 5, 6),
    then the scale of the prim in the world frame is (4, 10, 18).

    Args:
        prim: The USD prim to resolve the scale for.

    Returns:
        The scale of the prim in the x, y, and z directions in the world frame.

    Raises:
        ValueError: If the prim is not valid.
    """

    if not prim.IsValid():
        raise ValueError(f"Prim at path '{prim.GetPath().pathString}' is not valid.")

    xform = UsdGeom.Xformable(prim)
    world_transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    return tuple([*(v.GetLength() for v in world_transform.ExtractRotationMatrix())])


"""
USD Stage traversal.
"""


def get_first_matching_child_prim(
    prim_path: str | Sdf.Path,
    predicate: Callable[[Usd.Prim], bool],
    stage: Usd.Stage | None = None,
    traverse_instance_prims: bool = True,
) -> Usd.Prim | None:
    """Recursively get the first USD Prim at the path string that passes the predicate function.

    This function performs a depth-first traversal of the prim hierarchy starting from
    :attr:`prim_path`, returning the first prim that satisfies the provided :attr:`predicate`.
    It optionally supports traversal through instance prims, which are normally skipped in standard USD
    traversals.

    USD instance prims are lightweight copies of prototype scene structures and are not included
    in default traversals unless explicitly handled. This function allows traversing into instances
    when :attr:`traverse_instance_prims` is set to :attr:`True`.

    .. versionchanged:: 2.3.0

        Added :attr:`traverse_instance_prims` to control whether to traverse instance prims.
        By default, instance prims are now traversed.

    Args:
        prim_path: The path of the prim in the stage.
        predicate: The function to test the prims against. It takes a prim as input and returns a boolean.
        stage: The stage where the prim exists. Defaults to None, in which case the current stage is used.
        traverse_instance_prims: Whether to traverse instance prims. Defaults to True.

    Returns:
        The first prim on the path that passes the predicate. If no prim passes the predicate, it returns None.

    Raises:
        ValueError: If the prim path is not global (i.e: does not start with '/').
    """

    if stage is None:
        stage = get_current_stage()


    prim_path = str(prim_path)

    if not prim_path.startswith("/"):
        raise ValueError(f"Prim path '{prim_path}' is not global. It must start with '/'.")

    prim = stage.GetPrimAtPath(prim_path)

    if not prim.IsValid():
        raise ValueError(f"Prim at path '{prim_path}' is not valid.")

    all_prims = [prim]
    while len(all_prims) > 0:

        child_prim = all_prims.pop(0)

        if predicate(child_prim):
            return child_prim

        if traverse_instance_prims:
            all_prims += child_prim.GetFilteredChildren(Usd.TraverseInstanceProxies())
        else:
            all_prims += child_prim.GetChildren()
    return None


def get_all_matching_child_prims(
    prim_path: str | Sdf.Path,
    predicate: Callable[[Usd.Prim], bool] = lambda _: True,
    depth: int | None = None,
    stage: Usd.Stage | None = None,
    traverse_instance_prims: bool = True,
) -> list[Usd.Prim]:
    """Performs a search starting from the root and returns all the prims matching the predicate.

    This function performs a depth-first traversal of the prim hierarchy starting from
    :attr:`prim_path`, returning all prims that satisfy the provided :attr:`predicate`. It optionally
    supports traversal through instance prims, which are normally skipped in standard USD traversals.

    USD instance prims are lightweight copies of prototype scene structures and are not included
    in default traversals unless explicitly handled. This function allows traversing into instances
    when :attr:`traverse_instance_prims` is set to :attr:`True`.

    .. versionchanged:: 2.3.0

        Added :attr:`traverse_instance_prims` to control whether to traverse instance prims.
        By default, instance prims are now traversed.

    Args:
        prim_path: The root prim path to start the search from.
        predicate: The predicate that checks if the prim matches the desired criteria. It takes a prim as input
            and returns a boolean. Defaults to a function that always returns True.
        depth: The maximum depth for traversal, should be bigger than zero if specified.
            Defaults to None (i.e: traversal happens till the end of the tree).
        stage: The stage where the prim exists. Defaults to None, in which case the current stage is used.
        traverse_instance_prims: Whether to traverse instance prims. Defaults to True.

    Returns:
        A list containing all the prims matching the predicate.

    Raises:
        ValueError: If the prim path is not global (i.e: does not start with '/').
    """

    if stage is None:
        stage = get_current_stage()


    prim_path = str(prim_path)

    if not prim_path.startswith("/"):
        raise ValueError(f"Prim path '{prim_path}' is not global. It must start with '/'.")

    prim = stage.GetPrimAtPath(prim_path)

    if not prim.IsValid():
        raise ValueError(f"Prim at path '{prim_path}' is not valid.")

    if depth is not None and depth <= 0:
        raise ValueError(f"Depth must be bigger than zero, got {depth}.")



    all_prims_queue = [(prim, 0)]
    output_prims = []
    while len(all_prims_queue) > 0:

        child_prim, current_depth = all_prims_queue.pop(0)

        if predicate(child_prim):
            output_prims.append(child_prim)

        if depth is None or current_depth < depth:

            if traverse_instance_prims:
                children = child_prim.GetFilteredChildren(Usd.TraverseInstanceProxies())
            else:
                children = child_prim.GetChildren()

            all_prims_queue += [(child, current_depth + 1) for child in children]

    return output_prims


def find_first_matching_prim(prim_path_regex: str, stage: Usd.Stage | None = None) -> Usd.Prim | None:
    """Find the first matching prim in the stage based on input regex expression.

    Args:
        prim_path_regex: The regex expression for prim path.
        stage: The stage where the prim exists. Defaults to None, in which case the current stage is used.

    Returns:
        The first prim that matches input expression. If no prim matches, returns None.

    Raises:
        ValueError: If the prim path is not global (i.e: does not start with '/').
    """

    if stage is None:
        stage = get_current_stage()


    if not prim_path_regex.startswith("/"):
        raise ValueError(f"Prim path '{prim_path_regex}' is not global. It must start with '/'.")

    pattern = f"^{prim_path_regex}$"
    compiled_pattern = re.compile(pattern)

    for prim in stage.Traverse():

        if compiled_pattern.match(prim.GetPath().pathString) is not None:
            return prim
    return None


def find_matching_prims(prim_path_regex: str, stage: Usd.Stage | None = None) -> list[Usd.Prim]:
    """Find all the matching prims in the stage based on input regex expression.

    Args:
        prim_path_regex: The regex expression for prim path.
        stage: The stage where the prim exists. Defaults to None, in which case the current stage is used.

    Returns:
        A list of prims that match input expression.

    Raises:
        ValueError: If the prim path is not global (i.e: does not start with '/').
    """

    if stage is None:
        stage = get_current_stage()


    if not prim_path_regex.startswith("/"):
        raise ValueError(f"Prim path '{prim_path_regex}' is not global. It must start with '/'.")

    tokens = prim_path_regex.split("/")[1:]
    tokens = [f"^{token}$" for token in tokens]

    all_prims = [stage.GetPseudoRoot()]
    output_prims = []
    for index, token in enumerate(tokens):
        token_compiled = re.compile(token)
        for prim in all_prims:
            for child in prim.GetAllChildren():
                if token_compiled.match(child.GetName()) is not None:
                    output_prims.append(child)
        if index < len(tokens) - 1:
            all_prims = output_prims
            output_prims = []
    return output_prims


def find_matching_prim_paths(prim_path_regex: str, stage: Usd.Stage | None = None) -> list[str]:
    """Find all the matching prim paths in the stage based on input regex expression.

    Args:
        prim_path_regex: The regex expression for prim path.
        stage: The stage where the prim exists. Defaults to None, in which case the current stage is used.

    Returns:
        A list of prim paths that match input expression.

    Raises:
        ValueError: If the prim path is not global (i.e: does not start with '/').
    """

    output_prims = find_matching_prims(prim_path_regex, stage)

    output_prim_paths = []
    for prim in output_prims:
        output_prim_paths.append(prim.GetPath().pathString)
    return output_prim_paths


def find_global_fixed_joint_prim(
    prim_path: str | Sdf.Path, check_enabled_only: bool = False, stage: Usd.Stage | None = None
) -> UsdPhysics.Joint | None:
    """Find the fixed joint prim under the specified prim path that connects the target to the simulation world.

    A joint is a connection between two bodies. A fixed joint is a joint that does not allow relative motion
    between the two bodies. When a fixed joint has only one target body, it is considered to attach the body
    to the simulation world.

    This function finds the fixed joint prim that has only one target under the specified prim path. If no such
    fixed joint prim exists, it returns None.

    Args:
        prim_path: The prim path to search for the fixed joint prim.
        check_enabled_only: Whether to consider only enabled fixed joints. Defaults to False.
            If False, then all joints (enabled or disabled) are considered.
        stage: The stage where the prim exists. Defaults to None, in which case the current stage is used.

    Returns:
        The fixed joint prim that has only one target. If no such fixed joint prim exists, it returns None.

    Raises:
        ValueError: If the prim path is not global (i.e: does not start with '/').
        ValueError: If the prim path does not exist on the stage.
    """

    if stage is None:
        stage = get_current_stage()


    if not prim_path.startswith("/"):
        raise ValueError(f"Prim path '{prim_path}' is not global. It must start with '/'.")


    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path '{prim_path}' is not valid.")

    fixed_joint_prim = None


    for prim in Usd.PrimRange(prim):


        joint_prim = UsdPhysics.Joint(prim)
        if joint_prim:

            if check_enabled_only and not joint_prim.GetJointEnabledAttr().Get():
                continue

            body_0_exist = joint_prim.GetBody0Rel().GetTargets() != []
            body_1_exist = joint_prim.GetBody1Rel().GetTargets() != []

            if not (body_0_exist and body_1_exist):
                fixed_joint_prim = joint_prim
                break

    return fixed_joint_prim


"""
USD Variants.
"""


def select_usd_variants(prim_path: str, variants: object | dict[str, str], stage: Usd.Stage | None = None):
    """Sets the variant selections from the specified variant sets on a USD prim.

    `USD Variants`_ are a very powerful tool in USD composition that allows prims to have different options on
    a single asset. This can be done by modifying variations of the same prim parameters per variant option in a set.
    This function acts as a script-based utility to set the variant selections for the specified variant sets on a
    USD prim.

    The function takes a dictionary or a config class mapping variant set names to variant selections. For instance,
    if we have a prim at ``"/World/Table"`` with two variant sets: "color" and "size", we can set the variant
    selections as follows:

    .. code-block:: python

        select_usd_variants(
            prim_path="/World/Table",
            variants={
                "color": "red",
                "size": "large",
            },
        )

    Alternatively, we can use a config class to define the variant selections:

    .. code-block:: python

        @configclass
        class TableVariants:
            color: Literal["blue", "red"] = "red"
            size: Literal["small", "large"] = "large"

        select_usd_variants(
            prim_path="/World/Table",
            variants=TableVariants(),
        )

    Args:
        prim_path: The path of the USD prim.
        variants: A dictionary or config class mapping variant set names to variant selections.
        stage: The USD stage. Defaults to None, in which case, the current stage is used.

    Raises:
        ValueError: If the prim at the specified path is not valid.

    .. _USD Variants: https://graphics.pixar.com/usd/docs/USD-Glossary.html#USDGlossary-Variant
    """

    if stage is None:
        stage = get_current_stage()


    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path '{prim_path}' is not valid.")

    if not isinstance(variants, dict):
        variants = variants.to_dict()

    existing_variant_sets = prim.GetVariantSets()
    for variant_set_name, variant_selection in variants.items():

        if not existing_variant_sets.HasVariantSet(variant_set_name):
            omni.log.warn(f"Variant set '{variant_set_name}' does not exist on prim '{prim_path}'.")
            continue

        variant_set = existing_variant_sets.GetVariantSet(variant_set_name)

        if variant_set.GetVariantSelection() != variant_selection:
            variant_set.SetVariantSelection(variant_selection)
            omni.log.info(
                f"Setting variant selection '{variant_selection}' for variant set '{variant_set_name}' on"
                f" prim '{prim_path}'."
            )


"""
Stage management.
"""


def attach_stage_to_usd_context(attaching_early: bool = False):
    """Attaches the current USD stage in memory to the USD context.

    This function should be called during or after scene is created and before stage is simulated or rendered.

    Note:
        If the stage is not in memory or rendering is not enabled, this function will return without attaching.

    Args:
        attaching_early: Whether to attach the stage to the usd context before stage is created. Defaults to False.
    """

    from isaacsim.core.simulation_manager import SimulationManager

    from isaaclab.sim.simulation_context import SimulationContext


    isaac_sim_version = float(".".join(get_version()[2]))
    if isaac_sim_version < 5:
        return


    if not is_current_stage_in_memory():
        return


    stage_id = get_current_stage_id()
    physx_sim_interface = omni.physx.get_physx_simulation_interface()
    physx_sim_interface.attach_stage(stage_id)


    carb_setting = carb.settings.get_settings()
    is_rendering_enabled = get_carb_setting(carb_setting, "/physics/fabricUpdateTransformations")


    if not is_rendering_enabled:
        return


    if attaching_early:
        omni.log.warn(
            "Attaching stage in memory to USD context early to support an operation which doesn't support stage in"
            " memory."
        )


    SimulationContext.instance().skip_next_stage_open_callback()


    SimulationManager.enable_stage_open_callback(False)


    SimulationContext.instance()._physics_context.enable_fabric(True)


    omni.usd.get_context().attach_stage_with_callback(stage_id)


    physx_sim_interface = omni.physx.get_physx_simulation_interface()
    physx_sim_interface.attach_stage(stage_id)


    SimulationManager.enable_stage_open_callback(True)


def is_current_stage_in_memory() -> bool:
    """Checks if the current stage is in memory.

    This function compares the stage id of the current USD stage with the stage id of the USD context stage.

    Returns:
        Whether the current stage is in memory.
    """


    stage_id = get_current_stage_id()


    context_stage = omni.usd.get_context().get_stage()
    with use_stage(context_stage):
        context_stage_id = get_current_stage_id()


    return stage_id != context_stage_id


@contextlib.contextmanager
def use_stage(stage: Usd.Stage) -> Generator[None, None, None]:
    """Context manager that sets a thread-local stage, if supported.

    In Isaac Sim < 5.0, this is a no-op to maintain compatibility.

    Args:
        stage: The stage to set temporarily.

    Yields:
        None
    """
    isaac_sim_version = float(".".join(get_version()[2]))
    if isaac_sim_version < 5:
        omni.log.warn("[Compat] Isaac Sim < 5.0 does not support thread-local stage contexts. Skipping use_stage().")
        yield
    else:
        with stage_utils.use_stage(stage):
            yield


def create_new_stage_in_memory() -> Usd.Stage:
    """Creates a new stage in memory, if supported.

    Returns:
        The new stage in memory.
    """
    isaac_sim_version = float(".".join(get_version()[2]))
    if isaac_sim_version < 5:
        omni.log.warn(
            "[Compat] Isaac Sim < 5.0 does not support creating a new stage in memory. Falling back to creating a new"
            " stage attached to USD context."
        )
        return stage_utils.create_new_stage()
    else:
        return stage_utils.create_new_stage_in_memory()


def get_current_stage_id() -> int:
    """Gets the current open stage id.

    This function is a reimplementation of :meth:`isaacsim.core.utils.stage.get_current_stage_id` for
    backwards compatibility to Isaac Sim < 5.0.

    Returns:
        The current open stage id.
    """
    stage = get_current_stage()
    stage_cache = UsdUtils.StageCache.Get()
    stage_id = stage_cache.GetId(stage).ToLongInt()
    if stage_id < 0:
        stage_id = stage_cache.Insert(stage).ToLongInt()
    return stage_id
