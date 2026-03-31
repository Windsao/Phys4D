

from isaacsim.util.debug_draw import _debug_draw
from omni.physx.scripts import deformableUtils
from pxr import Sdf, Usd, UsdGeom

draw = _debug_draw.acquire_debug_draw_interface()

import numpy as np
import random

import wildmeshing as wm

from isaaclab.utils import configclass


@configclass
class TetMeshCfg:
    """

    References:

    """

    stop_quality: int = 10
    """
    Max AMIPS energy for stopping mesh optimization.

    Larger means less optimization and sooner stopping.
    """

    max_its: int = 80
    """
    Max number of mesh optimization iterations.
    """

    epsilon_r: float = 1e-2
    """ Relative envelope epsilon_r (definies the envelope size).

    -> Absolute epsilon = epsilon_r * diagonal_of_bbox.

    Smaller envelope preserves features better.
    Large Envelope + large edge_length = tetmesh with low res
    """

    edge_length_r: float = 1 / 2
    """ Relative target edge length l_r.

    -> Absolute l = l_r * diagonal_of_bbox.

    Smaller edge length gives denser mesh.
    """

    skip_simplify: bool = False

    coarsen: bool = True

    log_level: int = 6
    """
    Log level, ranges from 0 to 6  (0 = most verbose, 6 = off).
    """


@configclass
class TriMeshCfg:
    """

    References:

    """

    stop_quality: int = 10
    """
    Max AMIPS energy for stopping mesh optimization.

    Larger means less optimization and sooner stopping.
    """

    max_its: int = 80
    """
    Max number of mesh optimization iterations.
    """

    epsilon_r: float = 1e-2
    """ Relative envelope epsilon_r (definies the envelope size).
    -> Absolute epsilon = epsilon_r * diagonal_of_bbox.

    Smaller envelope pereserves features better.
    Larger Envelope + larger edge_length = tetmesh with low res
    """

    edge_length_r: float = 1 / 2
    """ Relative target edge length l_r.
    -> Absolute l = l_r * diagonal_of_bbox.

    Smaller edge length gives denser mesh.
    """

    skip_simplify: bool = False

    coarsen: bool = True

    log_level: int = 6
    """
    Log level, ranges from 0 to 6  (0 = most verbose, 6 = off).
    """


class MeshGenerator:
    def __init__(self, config: dict | TetMeshCfg = None) -> None:
        if config is None:
            config = TetMeshCfg().to_dict()
            print("No config for tet computation provided, using default settings.")
        elif type(config) is TetMeshCfg:
            config = config.to_dict()
        self.cfg = config
























        self.tetrahedralizer = wm.Tetrahedralizer(
            stop_quality=self.cfg["stop_quality"],
            max_its=self.cfg["max_its"],
            epsilon=self.cfg[
                "epsilon_r"
            ],
            edge_length_r=self.cfg[
                "edge_length_r"
            ],
            skip_simplify=self.cfg["skip_simplify"],
            coarsen=self.cfg["coarsen"],
        )
        self.tetrahedralizer.set_log_level(
            self.cfg["log_level"]
        )

















































    def generate_tet_mesh_for_prim(self, prim: UsdGeom.Mesh):

        points = np.array(prim.GetPointsAttr().Get())

        triangles = deformableUtils.triangulate_mesh(prim)

        tet_mesh_points, tet_indices, surf_points, surf_indices = self.compute_tet_mesh(points, triangles)











        return tet_mesh_points, tet_indices, surf_points, surf_indices

    def compute_tet_mesh(self, points, triangles):
        triangles = np.array(triangles, dtype=np.uint32).reshape(-1, 3)
        points = np.array(points, dtype=np.float64)























        self.tetrahedralizer.set_mesh(points, triangles)
        self.tetrahedralizer.tetrahedralize()
        tet_points, tet_indices, _ = self.tetrahedralizer.get_tet_mesh(
            correct_surface_orientation=True,
            manifold_surface=True,
            use_input_for_wn=False,
        )
        tet_indices = np.array(tet_indices).flatten().tolist()


        surf_points, surf_indices = self.tetrahedralizer.get_tracked_surfaces()
        surf_points = surf_points[0]
        surf_indices = np.array(surf_indices).flatten().tolist()
        return tet_points, tet_indices, surf_points, surf_indices

    @staticmethod
    def update_usd_mesh(prim: UsdGeom.Mesh, surf_points, triangles: list[int]):
        triangles = np.array(triangles).reshape(-1, 3)

        prim.GetPointsAttr().Set(surf_points)
        prim.GetFaceVertexCountsAttr().Set(
            [3] * triangles.shape[0]
        )
        prim.GetFaceVertexIndicesAttr().Set(triangles)
        prim.GetNormalsAttr().Set([])
        prim.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)



        colors = [
            (random.uniform(0.0, 0.0), random.uniform(0.0, 0.75), random.uniform(0.0, 0.75))
            for _ in range(triangles.shape[0] * 3)
        ]
        prim.CreateDisplayColorPrimvar(UsdGeom.Tokens.faceVarying).Set(colors)


        uv_coor = np.indices((int(triangles.size), 2)).transpose((1, 2, 0)).reshape((-1, 2))

        pv_api = UsdGeom.PrimvarsAPI(prim)
        if pv_api.HasPrimvar("primvars:st"):
            pv = pv_api.GetPrimvar("primvars:st")
            pv.SetInterpolation(UsdGeom.Tokens.faceVarying)



        else:

            pv = pv_api.CreatePrimvar(
                "primvars:st",
                Sdf.ValueTypeNames.TexCoord2fArray,
                UsdGeom.Tokens.faceVarying,

                uv_coor.size,
            )
            pv.Set(uv_coor)

    @staticmethod
    def update_usd_mesh_with_uipc_surface(prim: Usd.Prim):
        """Method to update render mesh topology based on the surface of the mesh in UIPC.

        Used for workaround so that textures can be applied to the meshes.
        """
        from uipc.geometry import extract_surface, flip_inward_triangles, label_surface, label_triangle_orient, tetmesh

        tet_points = prim.GetAttribute("tet_points").Get()
        tet_points = np.array(tet_points)

        tet_indices = prim.GetAttribute("tet_indices").Get()
        tet_indices = np.array(tet_indices).reshape(-1, 4)


        mesh = tetmesh(tet_points.copy(), tet_indices.copy())

        label_surface(mesh)
        label_triangle_orient(mesh)

        mesh = flip_inward_triangles(mesh)
        surf = extract_surface(mesh)

        surf_points = surf.positions().view().reshape(-1, 3)
        triangles = surf.triangles().topo().view().reshape(-1).tolist()


        attr_tet_surf_points = prim.CreateAttribute("tet_surf_points", Sdf.ValueTypeNames.Vector3fArray)
        attr_tet_surf_points.Set(surf_points)

        attr_tet_surf_indices = prim.CreateAttribute("tet_surf_indices", Sdf.ValueTypeNames.UIntArray)
        attr_tet_surf_indices.Set(triangles)


        prim = UsdGeom.Mesh(prim)

        triangles = np.array(triangles).reshape(-1, 3)
        prim.GetPointsAttr().Set(surf_points)
        prim.GetFaceVertexCountsAttr().Set(
            [3] * triangles.shape[0]
        )
        prim.GetFaceVertexIndicesAttr().Set(triangles)
        prim.GetNormalsAttr().Set([])
        prim.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)



        colors = [
            (random.uniform(0.0, 0.0), random.uniform(0.0, 0.75), random.uniform(0.0, 0.75))
            for _ in range(triangles.shape[0] * 3)
        ]
        prim.CreateDisplayColorPrimvar(UsdGeom.Tokens.faceVarying).Set(colors)


        uv_coor = np.indices((int(triangles.shape[0] * 1.5), 2)).transpose((1, 2, 0)).reshape((-1, 2))

        pv_api = UsdGeom.PrimvarsAPI(prim)
        if pv_api.HasPrimvar("primvars:st"):
            pv = pv_api.GetPrimvar("primvars:st")
            pv.SetInterpolation(UsdGeom.Tokens.faceVarying)
        else:
            pv = pv_api.CreatePrimvar(
                "primvars:st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying, uv_coor.size
            )
        pv.Set(uv_coor)
