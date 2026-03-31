from __future__ import annotations

import omni.usd
import usdrt
from pxr import UsdGeom
from uipc.geometry import extract_surface, tetmesh

from isaaclab.sim.spawners import materials
from isaaclab.sim.spawners.spawner_cfg import DeformableObjectSpawnerCfg, RigidObjectSpawnerCfg
from isaaclab.utils import configclass

from tacex_uipc.utils import MeshGenerator


@configclass
class FileCfg(RigidObjectSpawnerCfg, DeformableObjectSpawnerCfg):
    """Configuration parameters for spawning an asset from a file.

    This class is a base class for spawning assets from files. It includes the common parameters
    for spawning assets from files, such as the path to the file and the function to use for spawning
    the asset.

    Note:
        By default, all properties are set to None. This means that no properties will be added or modified
        to the prim outside of the properties available by default when spawning the prim.

        If they are set to a value, then the properties are modified on the spawned prim in a nested manner.
        This is done by calling the respective function with the specified properties.
    """

    scale: tuple[float, float, float] | None = None
    """Scale of the asset. Defaults to None, in which case the scale is not modified."""










    visual_material_path: str = "material"
    """Path to the visual material to use for the prim. Defaults to "material".

    If the path is relative, then it will be relative to the prim's path.
    This parameter is ignored if `visual_material` is not None.
    """

    visual_material: materials.VisualMaterialCfg | None = None
    """Visual material properties to override the visual material properties in the URDF file.

    Note:
        If None, then no visual material will be added.
    """




















































































"""

Helper functions

"""


def create_prim_for_tet_data(prim_path, tet_points_world, tet_indices):

    stage = omni.usd.get_context().get_stage()
    prim = UsdGeom.Mesh.Define(stage, prim_path)


    uipc_tet_mesh = tetmesh(tet_points_world.copy(), tet_indices.copy())
    surf = extract_surface(uipc_tet_mesh)
    tet_surf_tri = surf.triangles().topo().view().reshape(-1).tolist()
    tet_surf_points_world = surf.positions().view().reshape(-1, 3)

    MeshGenerator.update_usd_mesh(prim=prim, surf_points=tet_surf_points_world, triangles=tet_surf_tri)


def create_prim_for_uipc_scene_object(uipc_sim, prim_path, uipc_scene_object):

    stage = omni.usd.get_context().get_stage()
    prim = UsdGeom.Mesh.Define(stage, prim_path)


    obj_id = uipc_scene_object.geometries().ids()[0]
    simplicial_complex_slot, _ = uipc_sim.scene.geometries().find(obj_id)


    surf = extract_surface(simplicial_complex_slot.geometry())
    tet_surf_tri = surf.triangles().topo().view().reshape(-1).tolist()
    tet_surf_points_world = surf.positions().view().reshape(-1, 3)

    MeshGenerator.update_usd_mesh(prim=prim, surf_points=tet_surf_points_world, triangles=tet_surf_tri)


    fabric_stage = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())
    fabric_prim = fabric_stage.GetPrimAtPath(usdrt.Sdf.Path(prim_path))


    if not fabric_prim.HasAttribute("Deformable"):
        fabric_prim.CreateAttribute("Deformable", usdrt.Sdf.ValueTypeNames.PrimTypeTag, True)


    rtxformable = usdrt.Rt.Xformable(fabric_prim)
    rtxformable.CreateFabricHierarchyWorldMatrixAttr()

    rtxformable.GetFabricHierarchyWorldMatrixAttr().Set(usdrt.Gf.Matrix4d())


    fabric_mesh_points_attr = fabric_prim.GetAttribute("points")
    fabric_mesh_points_attr.Set(usdrt.Vt.Vec3fArray(tet_surf_points_world))


    uipc_sim._fabric_meshes.append(fabric_prim)


    num_surf_points = tet_surf_points_world.shape[0]
    uipc_sim._surf_vertex_offsets.append(uipc_sim._surf_vertex_offsets[-1] + num_surf_points)
