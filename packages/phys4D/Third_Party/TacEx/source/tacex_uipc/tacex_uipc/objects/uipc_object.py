




from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log
import omni.physics.tensors.impl.api as physx
import omni.usd
import usdrt
import usdrt.UsdGeom
from isaacsim.core.prims import XFormPrim
from pxr import UsdGeom

try:
    from isaacsim.util.debug_draw import _debug_draw

    draw = _debug_draw.acquire_debug_draw_interface()
except ImportError:
    import warnings

    warnings.warn("_debug_draw failed to import", ImportWarning)
    draw = None

import numpy as np

import warp as wp
from uipc import builtin, view
from uipc.constitution import AffineBodyConstitution, ElasticModuli, StableNeoHookean
from uipc.geometry import extract_surface, flip_inward_triangles, label_surface, label_triangle_orient, tetmesh
from uipc.unit import MPa

import isaaclab.utils.string as string_utils
from isaaclab.assets import AssetBase, AssetBaseCfg
from isaaclab.utils import configclass

wp.init()


from tacex_uipc.utils import MeshGenerator, TetMeshCfg

from .uipc_object_deformable_data import UipcObjectDeformableData
from .uipc_object_rigid_data import UipcObjectRigidData

if TYPE_CHECKING:
    from tacex_uipc.sim import UipcIsaacAttachmentsCfg, UipcSim


@configclass
class UipcObjectCfg(AssetBaseCfg):
    mesh_cfg: TetMeshCfg = None


    mass_density: float = 1e3

    @configclass
    class AffineBodyConstitutionCfg:

        m_kappa: float = 100.0
        """Stiffness (hardness) of the object
        in [MPa]

        E.g. 100.0 MPa = hard-rubber-like material
        """

        kinematic: bool = False
        """Makes the DoF of the ABD body fixed.

        """

    @configclass
    class StableNeoHookeanCfg:

        youngs_modulus: float = 0.01
        """
        in [MPa]
        """

        poisson_rate: float = 0.49
        """ Poission Rate

        Has to be < 0.5.
        """

    constitution_cfg: AffineBodyConstitutionCfg | StableNeoHookeanCfg = None

    attachment_cfg: UipcIsaacAttachmentsCfg = None


class UipcObject(AssetBase):
    """A rigid object asset class.

    Rigid objects are assets comprising of rigid bodies. They can be used to represent dynamic objects
    such as boxes, spheres, etc. A rigid body is described by its pose, velocity and mass distribution.

    For an asset to be considered a rigid object, the root prim of the asset must have the `USD RigidBodyAPI`_
    applied to it. This API is used to define the simulation properties of the rigid body. On playing the
    simulation, the physics engine will automatically register the rigid body and create a corresponding
    rigid body handle. This handle can be accessed using the :attr:`root_physx_view` attribute.

    .. note::

        For users familiar with Isaac Sim, the PhysX view class API is not the exactly same as Isaac Sim view
        class API. Similar to Isaac Lab, Isaac Sim wraps around the PhysX view API. However, as of now (2023.1 release),
        we see a large difference in initializing the view classes in Isaac Sim. This is because the view classes
        in Isaac Sim perform additional USD-related operations which are slow and also not required.

    .. _`USD RigidBodyAPI`: https://openusd.org/dev/api/class_usd_physics_rigid_body_a_p_i.html
    """

    cfg: UipcObjectCfg
    """Configuration instance for the rigid object."""

    def __init__(self, cfg: UipcObjectCfg, uipc_sim: UipcSim):
        """Initialize the uipc object.

        Args:
            cfg: A configuration instance.
        """
        super().__init__(cfg)
        self._uipc_sim: UipcSim = uipc_sim

        prim_paths_expr = self.cfg.prim_path
        print(f"Initializing uipc objects {prim_paths_expr}...")
        self._prim_view = XFormPrim(prim_paths_expr=prim_paths_expr, name=f"{prim_paths_expr}", usd=False)
        self._prim_view.initialize()

        self.stage = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())

        self.uipc_scene_objects = []
        self.geo_slot_list = []

        self.uipc_meshes = []

        for (
            prim
        ) in (
            self._prim_view.prims
        ):

            prim_children = prim.GetChildren()
            usd_mesh = UsdGeom.Mesh(prim_children[0])
            usd_mesh_path = str(usd_mesh.GetPath())
            print("usd_mesh_path ", usd_mesh_path)

            if self.cfg.mesh_cfg is None:

                tet_points = np.array(prim_children[0].GetAttribute("tet_points").Get())
                tet_indices = prim_children[0].GetAttribute("tet_indices").Get()
                surf_points = np.array(prim_children[0].GetAttribute("tet_surf_points").Get())
                tet_surf_indices = prim_children[0].GetAttribute("tet_surf_indices").Get()

                if tet_indices is None:

                    raise Exception(f"No precomputed tet mesh data found for prim at {usd_mesh_path}")
            else:
                mesh_gen = MeshGenerator(config=self.cfg.mesh_cfg)
                if type(self.cfg.mesh_cfg) is TetMeshCfg:
                    tet_points, tet_indices, surf_points, tet_surf_indices = mesh_gen.generate_tet_mesh_for_prim(
                        usd_mesh
                    )


            tf_world = omni.usd.get_world_transform_matrix(usd_mesh)

            tet_points_world = np.array(tf_world).T @ np.vstack((tet_points.T, np.ones(tet_points.shape[0])))
            tet_points_world = tet_points_world[:-1].T

            self.init_world_transform = torch.tensor(np.array(tf_world).T.copy(), device=self.uipc_sim.cfg.device)


            tet_indices = np.array(tet_indices).reshape(-1, 4)
            tet_surf_indices = np.array(tet_surf_indices).reshape(-1, 3)


            mesh = tetmesh(tet_points_world.copy(), tet_indices.copy())

            label_surface(mesh)
            label_triangle_orient(mesh)

            mesh = flip_inward_triangles(mesh)
            self.uipc_meshes.append(mesh)


            surf = extract_surface(mesh)
            tet_surf_points_world = surf.positions().view().reshape(-1, 3)
            tet_surf_tri = surf.triangles().topo().view().reshape(-1).tolist()

            MeshGenerator.update_usd_mesh(prim=usd_mesh, surf_points=tet_surf_points_world, triangles=tet_surf_tri)



            self._create_constitutions(mesh)


            fabric_prim = self.stage.GetPrimAtPath(usdrt.Sdf.Path(usd_mesh_path))
            if not fabric_prim:
                print(f"Prim at path {usd_mesh_path} is not in Fabric")
            if not fabric_prim.HasAttribute("points"):
                print(f"Prim at path {usd_mesh_path} does not have points attribute")


            if not fabric_prim.HasAttribute("Deformable"):
                fabric_prim.CreateAttribute("Deformable", usdrt.Sdf.ValueTypeNames.PrimTypeTag, True)


            rtxformable = usdrt.Rt.Xformable(fabric_prim)
            rtxformable.CreateFabricHierarchyWorldMatrixAttr()

            rtxformable.GetFabricHierarchyWorldMatrixAttr().Set(usdrt.Gf.Matrix4d())


            fabric_mesh_points_attr = fabric_prim.GetAttribute("points")
            fabric_mesh_points_attr.Set(usdrt.Vt.Vec3fArray(tet_surf_points_world))

            self.fabric_prim = fabric_prim


            self._uipc_sim._fabric_meshes.append(fabric_prim)


            num_surf_points = tet_surf_points_world.shape[0]
            self._uipc_sim._surf_vertex_offsets.append(self._uipc_sim._surf_vertex_offsets[-1] + num_surf_points)


            num_vertex_points = mesh.positions().view().shape[0]
            self._vertex_count = num_vertex_points


            self._uipc_sim._system_vertex_offsets[self._system_name].append(
                self._uipc_sim._system_vertex_offsets[self._system_name][-1] + self._vertex_count
            )
            self.local_system_id = len(self._uipc_sim._system_vertex_offsets[self._system_name]) - 1


            self.global_system_id = 0

            self._data = None

    """
    Properties
    """

    @property
    def data(self) -> UipcObjectDeformableData | UipcObjectRigidData:
        return self._data

    @property
    def num_instances(self) -> int:
        return self._prim_view.count

    @property
    def num_bodies(self) -> int:
        """Number of bodies in the asset.

        This is always 1 since each object is a single rigid body.
        """
        return 1

    @property
    def body_names(self) -> list[str]:
        """Ordered names of bodies in the rigid object."""
        prim_paths = self.root_physx_view.prim_paths[: self.num_bodies]
        return [path.split("/")[-1] for path in prim_paths]

    @property
    def uipc_sim(self) -> physx.RigidBodyView:
        """uipc simulation instance of this uipc object."""
        return self._uipc_sim

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None):

        pass




    def write_data_to_sim(self):
        pass

    def update(self, dt: float):
        self._data.update(dt)

    """
    Operations - Finders.
    """

    def find_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        """Find bodies in the rigid body based on the name keys.

        Please check the :meth:`isaaclab.utils.string_utils.resolve_matching_names` function for more
        information on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the body names.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the body indices and names.
        """
        return string_utils.resolve_matching_names(name_keys, self.body_names, preserve_order)

    """
    Operations - Write to simulation.
    """

    def write_vertex_positions_to_sim(self, vertex_positions: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (w, x, y, z).

        Args:
            root_pose: Root poses in simulation frame. Shape is (len(env_ids), 7).
            env_ids: Environment indices. If None, then all indices are used.
        """














        print("")
        print(f"Write vertex pos for {self.cfg.prim_path} with obj id [{self.obj_id}]")

        print("num geo_slots ", len(self.geo_slot_list))
        print("global sys id ", self.global_system_id)
        geo_slot = self.geo_slot_list[0]
        geo = geo_slot.geometry()
        gvo = geo.meta().find(builtin.global_vertex_offset)
        print(f"global Vertex Offset: {gvo.view()}")
        global_vertex_offset = int(gvo.view()[0])
        local_vertex_offset = self._uipc_sim._system_vertex_offsets[self._system_name][self.local_system_id - 1]
        print("system ", self._system_name)
        print("local sys id ", self.local_system_id)
        print("local vertex offset ", local_vertex_offset)
        print("vertex count ", self._vertex_count)
        print("")
        if self._system_name == "uipc::backend::cuda::AffineBodyDynamics":
            self.uipc_sim.world.write_vertex_pos_to_sim(
                vertex_positions.cpu().numpy(),
                global_vertex_offset,
                self.local_system_id - 1,
                self._vertex_count,
                self._system_name,
            )
        else:
            self.uipc_sim.world.write_vertex_pos_to_sim(
                vertex_positions.cpu().numpy(),
                global_vertex_offset,
                local_vertex_offset,
                self._vertex_count,
                self._system_name,
            )

    """
    Internal helper.
    """

    def _initialize_impl(self):

        mesh = self.uipc_meshes[0]

        obj = self._uipc_sim.scene.objects().create(self.cfg.prim_path)
        self.uipc_scene_objects.append(obj)

        obj_geo_slot, _ = obj.geometries().create(mesh)
        self.obj_id = obj_geo_slot.id()
        print(f"obj id of {self.cfg.prim_path}: {self.obj_id} ")
        self.geo_slot_list.append(obj_geo_slot)


        geom = self._uipc_sim.scene.geometries()
        geo_slot, geo_slot_rest = geom.find(self.obj_id)
        self.init_vertex_pos = torch.tensor(
            geo_slot.geometry().positions().view().copy().reshape(-1, 3), device=self.device
        )


        omni.log.info(f"UIPC body initialized at: {self.cfg.prim_path}.")
        omni.log.info(f"Number of instances: {self.num_instances}")




        if type(self.constitution) is StableNeoHookean:
            self._data = UipcObjectDeformableData(self._uipc_sim, self, self.device)
        elif type(self.constitution) is AffineBodyConstitution:
            self._data = UipcObjectRigidData(self._uipc_sim, self, self.device)

        self._create_buffers()

        self._process_cfg()

        self.update(0.0)


        self._uipc_sim.uipc_objects.append(self)

    def _create_buffers(self):
        """Create buffers for storing data."""

        self._ALL_INDICES = torch.arange(self.num_instances, dtype=torch.long, device=self.device)








    def _process_cfg(self):
        """Post processing of configuration parameters."""



        default_root_state = (
            tuple(self.cfg.init_state.pos)
            + tuple(self.cfg.init_state.rot)


        )
        default_root_state = torch.tensor(default_root_state, dtype=torch.float, device=self.device)


    def _create_constitutions(self, mesh):

        constitution_types = {
            UipcObjectCfg.AffineBodyConstitutionCfg: AffineBodyConstitution,
            UipcObjectCfg.StableNeoHookeanCfg: StableNeoHookean,
        }
        self.constitution = constitution_types[type(self.cfg.constitution_cfg)]()

        if type(self.constitution) is StableNeoHookean:
            youngs = self.cfg.constitution_cfg.youngs_modulus
            poisson = self.cfg.constitution_cfg.poisson_rate
            moduli = ElasticModuli.youngs_poisson(youngs * MPa, poisson)

            self.constitution.apply_to(mesh, moduli, mass_density=self.cfg.mass_density)

            self._system_name = "uipc::backend::cuda::FiniteElementMethod"
        elif type(self.constitution) is AffineBodyConstitution:
            stiffness = self.cfg.constitution_cfg.m_kappa
            self.constitution.apply_to(mesh, stiffness * MPa, mass_density=self.cfg.mass_density)
            self._system_name = "uipc::backend::cuda::AffineBodyDynamics"


            if self.cfg.constitution_cfg.kinematic:
                is_fixed_attr = mesh.instances().find(builtin.is_fixed)
                view(is_fixed_attr)[0] = 1


        default_element = self._uipc_sim.scene.contact_tabular().default_element()
        default_element.apply_to(mesh)

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""

        super()._invalidate_initialize_callback(event)

        self._physics_sim_view = None
        self._root_physx_view = None
