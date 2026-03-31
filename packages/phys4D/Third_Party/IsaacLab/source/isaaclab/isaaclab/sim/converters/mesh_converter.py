




import asyncio
import os

import omni
import omni.kit.commands
import omni.usd
from isaacsim.core.utils.extensions import enable_extension
from pxr import Gf, Tf, Usd, UsdGeom, UsdPhysics, UsdUtils

from isaaclab.sim.converters.asset_converter_base import AssetConverterBase
from isaaclab.sim.converters.mesh_converter_cfg import MeshConverterCfg
from isaaclab.sim.schemas import schemas
from isaaclab.sim.utils import export_prim_to_file


class MeshConverter(AssetConverterBase):
    """Converter for a mesh file in OBJ / STL / FBX format to a USD file.

    This class wraps around the `omni.kit.asset_converter`_ extension to provide a lazy implementation
    for mesh to USD conversion. It stores the output USD file in an instanceable format since that is
    what is typically used in all learning related applications.

    To make the asset instanceable, we must follow a certain structure dictated by how USD scene-graph
    instancing and physics work. The rigid body component must be added to each instance and not the
    referenced asset (i.e. the prototype prim itself). This is because the rigid body component defines
    properties that are specific to each instance and cannot be shared under the referenced asset. For
    more information, please check the `documentation <https://docs.isaacsim.omniverse.nvidia.com/latest/physics/simulation_fundamentals.html#rigid-body>`_.

    Due to the above, we follow the following structure:

    * ``{prim_path}`` - The root prim that is an Xform with the rigid body and mass APIs if configured.
    * ``{prim_path}/geometry`` - The prim that contains the mesh and optionally the materials if configured.
      If instancing is enabled, this prim will be an instanceable reference to the prototype prim.

    .. _omni.kit.asset_converter: https://docs.omniverse.nvidia.com/extensions/latest/ext_asset-converter.html

    .. caution::
        When converting STL files, Z-up convention is assumed, even though this is not the default for many CAD
        export programs. Asset orientation convention can either be modified directly in the CAD program's export
        process or an offset can be added within the config in Isaac Lab.

    """

    cfg: MeshConverterCfg
    """The configuration instance for mesh to USD conversion."""

    def __init__(self, cfg: MeshConverterCfg):
        """Initializes the class.

        Args:
            cfg: The configuration instance for mesh to USD conversion.
        """
        super().__init__(cfg=cfg)

    """
    Implementation specific methods.
    """

    def _convert_asset(self, cfg: MeshConverterCfg):
        """Generate USD from OBJ, STL or FBX.

        The USD file has Y-up axis and is scaled to meters.
        The asset hierarchy is arranged as follows:

        .. code-block:: none
            mesh_file_basename (default prim)
                |- /geometry/Looks
                |- /geometry/mesh

        Args:
            cfg: The configuration for conversion of mesh to USD.

        Raises:
            RuntimeError: If the conversion using the Omniverse asset converter fails.
        """

        mesh_file_basename, mesh_file_format = os.path.basename(cfg.asset_path).split(".")
        mesh_file_format = mesh_file_format.lower()


        if not Tf.IsValidIdentifier(mesh_file_basename):

            mesh_file_basename_original = mesh_file_basename
            mesh_file_basename = Tf.MakeValidIdentifier(mesh_file_basename)
            omni.log.warn(
                f"Input file name '{mesh_file_basename_original}' is an invalid identifier for the mesh prim path."
                f" Renaming it to '{mesh_file_basename}' for the conversion."
            )


        asyncio.get_event_loop().run_until_complete(
            self._convert_mesh_to_usd(in_file=cfg.asset_path, out_file=self.usd_path)
        )

        temp_stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(temp_stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(temp_stage, 1.0)
        UsdPhysics.SetStageKilogramsPerUnit(temp_stage, 1.0)

        base_prim = temp_stage.DefinePrim(f"/{mesh_file_basename}", "Xform")
        prim = temp_stage.DefinePrim(f"/{mesh_file_basename}/geometry", "Xform")
        prim.GetReferences().AddReference(self.usd_path)
        temp_stage.SetDefaultPrim(base_prim)
        temp_stage.Export(self.usd_path)


        stage = Usd.Stage.Open(self.usd_path)

        stage.Reload()

        stage_id = UsdUtils.StageCache.Get().Insert(stage)

        xform_prim = stage.GetDefaultPrim()
        geom_prim = stage.GetPrimAtPath(f"/{mesh_file_basename}/geometry")

        for child_mesh_prim in geom_prim.GetChildren():
            if child_mesh_prim.GetTypeName() == "Mesh":

                if cfg.collision_props is not None:

                    schemas.define_collision_properties(
                        prim_path=child_mesh_prim.GetPath(), cfg=cfg.collision_props, stage=stage
                    )

                if cfg.mesh_collision_props is not None:
                    schemas.define_mesh_collision_properties(
                        prim_path=child_mesh_prim.GetPath(), cfg=cfg.mesh_collision_props, stage=stage
                    )

        stage.SetDefaultPrim(xform_prim)

        omni.kit.commands.execute(
            "CreateDefaultXformOnPrimCommand",
            prim_path=xform_prim.GetPath(),
            **{"stage": stage},
        )


        geom_xform = UsdGeom.Xform(geom_prim)
        geom_xform.ClearXformOpOrder()


        rotate_attr = geom_prim.GetAttribute("xformOp:rotateXYZ")
        if rotate_attr:
            geom_prim.RemoveProperty(rotate_attr.GetName())


        translate_op = geom_xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        translate_op.Set(Gf.Vec3d(*cfg.translation))

        orient_op = geom_xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
        orient_op.Set(Gf.Quatd(*cfg.rotation))

        scale_op = geom_xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
        scale_op.Set(Gf.Vec3d(*cfg.scale))



        if cfg.make_instanceable:

            export_prim_to_file(
                path=os.path.join(self.usd_dir, self.usd_instanceable_meshes_path),
                source_prim_path=geom_prim.GetPath(),
                stage=stage,
            )

            geom_prim_path = geom_prim.GetPath().pathString
            omni.kit.commands.execute("DeletePrims", paths=[geom_prim_path], stage=stage)

            geom_undef_prim = stage.DefinePrim(geom_prim_path)
            geom_undef_prim.GetReferences().AddReference(self.usd_instanceable_meshes_path, primPath=geom_prim_path)
            geom_undef_prim.SetInstanceable(True)





        if cfg.mass_props is not None:
            schemas.define_mass_properties(prim_path=xform_prim.GetPath(), cfg=cfg.mass_props, stage=stage)

        if cfg.rigid_props is not None:
            schemas.define_rigid_body_properties(prim_path=xform_prim.GetPath(), cfg=cfg.rigid_props, stage=stage)


        stage.Save()
        if stage_id is not None:
            UsdUtils.StageCache.Get().Erase(stage_id)

    """
    Helper methods.
    """

    @staticmethod
    async def _convert_mesh_to_usd(in_file: str, out_file: str, load_materials: bool = True) -> bool:
        """Convert mesh from supported file types to USD.

        This function uses the Omniverse Asset Converter extension to convert a mesh file to USD.
        It is an asynchronous function and should be called using `asyncio.get_event_loop().run_until_complete()`.

        The converted asset is stored in the USD format in the specified output file.
        The USD file has Y-up axis and is scaled to cm.

        Args:
            in_file: The file to convert.
            out_file: The path to store the output file.
            load_materials: Set to True to enable attaching materials defined in the input file
                to the generated USD mesh. Defaults to True.

        Returns:
            True if the conversion succeeds.
        """
        enable_extension("omni.kit.asset_converter")

        import omni.kit.asset_converter
        import omni.usd


        converter_context = omni.kit.asset_converter.AssetConverterContext()


        converter_context.ignore_materials = not load_materials
        converter_context.ignore_animations = True
        converter_context.ignore_camera = True
        converter_context.ignore_light = True

        converter_context.merge_all_meshes = True


        converter_context.use_meter_as_world_unit = True
        converter_context.baking_scales = True

        converter_context.use_double_precision_to_usd_transform_op = True


        instance = omni.kit.asset_converter.get_instance()
        task = instance.create_converter_task(in_file, out_file, None, converter_context)

        success = await task.wait_until_finished()
        if not success:
            raise RuntimeError(f"Failed to convert {in_file} to USD. Error: {task.get_error_message()}")
        return success
