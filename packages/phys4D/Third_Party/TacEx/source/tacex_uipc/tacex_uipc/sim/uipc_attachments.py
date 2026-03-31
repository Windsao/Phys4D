from __future__ import annotations

import inspect
import numpy as np
import torch
import weakref

import omni
from omni.physx import get_physx_interface, get_physx_scene_query_interface
from pxr import UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils

try:
    from isaacsim.util.debug_draw import _debug_draw

    draw = _debug_draw.acquire_debug_draw_interface()
except ImportError:
    import warnings

    warnings.warn("_debug_draw failed to import", ImportWarning)
    draw = None

from uipc import Animation, builtin, view
from uipc.constitution import SoftPositionConstraint
from uipc.geometry import GeometrySlot, SimplicialComplex

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.utils import configclass
from isaaclab.utils.math import transform_points

from tacex_uipc.objects import UipcObject


@configclass
class UipcIsaacAttachmentsCfg:
    constraint_strength_ratio: float = 100.0
    """
    E.g., 100.0 means the stiffness of the constraint is 100 times of the mass of the uipc object.
    """

    debug_vis: bool = False
    """Draw attachment offsets and aim_position via IsaacSim's _debug_draw api.

    """

    body_name: str = None
    """Name of the body in the rigid object that should be used for the attachment.

    Useful, e.g. when attaching to a part of an articulation.
    """

    compute_attachment_data: bool = True
    """False to use precomputed attachment data.

    Note: Precomputed attachment data is only valid if the corresponding precomputed tet mesh data exists.
    This means, that the uipc_object of the attachment class should also use precomputed data.
    """

    attachment_points_radius: float = 5e-4
    """Distance between tet points and isaac collider, which is used to determine the attachment points.

    If the collision mesh of the isaaclab_rigid_object is in the radius of a point, then the
    point is considered "attached" to the isaaclab_rigid_object.
    """


class UipcIsaacAttachments:
    cfg: UipcIsaacAttachmentsCfg


    def __init__(
        self, cfg: UipcIsaacAttachmentsCfg, uipc_object: UipcObject, isaaclab_rigid_object: RigidObject | Articulation
    ) -> None:

        cfg.validate()
        self.cfg = cfg.copy()

        self.uipc_object: UipcObject = uipc_object

        if UsdPhysics.RigidBodyAPI(self.uipc_object._prim_view.prims[0]):
            raise RuntimeWarning(
                f"Prim {self.uipc_object.cfg.prim_path} of UIPC object is a Isaac Rigid Body. This (usually) leads to"
                " unwanted behavior (e.g. when part of articulation)."
            )

        self.isaaclab_rigid_object: RigidObject | Articulation = isaaclab_rigid_object

        self.rigid_body_id = None

        self.uipc_object_vertex_indices = []
        self.attachment_points_init_positions = []


        self.aim_positions = np.zeros(0)



        if not self.cfg.compute_attachment_data:




            prim_children = self.uipc_object._prim_view.prims[0].GetChildren()
            prim = prim_children[0]
            usd_mesh = UsdGeom.Mesh(prim)
            attachment_offsets = np.array(prim.GetAttribute("attachment_offsets").Get())
            idx = prim.GetAttribute("attachment_indices").Get()

            if idx is None:
                raise Exception(
                    f"No precomputed attachment data found for prim at {str(usd_mesh.GetPath())}. Use set a cfg for the"
                    " attachment to compute attachment data."
                )
        else:

            attachment_points_radius = self.cfg.attachment_points_radius

            isaac_rigid_prim_path = self.isaaclab_rigid_object.cfg.prim_path
            if self.cfg.body_name is not None:
                isaac_rigid_prim_path += "/.*" + self.cfg.body_name
            print("isaac_rigid_prim ", isaac_rigid_prim_path)

            mesh = self.uipc_object.uipc_meshes[0]
            tet_points_world = mesh.positions().view()[:, :, 0]
            tet_indices = mesh.tetrahedra().topo().view()[:, :, 0]
            attachment_offsets, idx, rigid_prims, attachment_points_pos, obj_pos = self.compute_attachment_data(
                isaac_rigid_prim_path, tet_points_world, tet_indices, attachment_points_radius
            )


        self.attachment_offsets = attachment_offsets
        self.attachment_points_idx = idx
        self.num_attachment_points_per_obj = len(idx)

        self._num_instances = 1


        soft_position_constraint = SoftPositionConstraint()

        soft_position_constraint.apply_to(self.uipc_object.uipc_meshes[0], self.cfg.constraint_strength_ratio)


        self._is_initialized = False




        timeline_event_stream = omni.timeline.get_timeline_interface().get_timeline_event_stream()
        self._initialize_handle = timeline_event_stream.create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.PLAY),
            lambda event, obj=weakref.proxy(self): obj._initialize_callback(event),
            order=10,
        )
        self._invalidate_initialize_handle = timeline_event_stream.create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.STOP),
            lambda event, obj=weakref.proxy(self): obj._invalidate_initialize_callback(event),
            order=10,
        )

        self._debug_vis_handle = None

        self.set_debug_vis(self.cfg.debug_vis)

    def __del__(self):
        """Unsubscribe from the callbacks."""

        if self._initialize_handle:
            self._initialize_handle.unsubscribe()
            self._initialize_handle = None
        if self._invalidate_initialize_handle:
            self._invalidate_initialize_handle.unsubscribe()
            self._invalidate_initialize_handle = None

        if self._debug_vis_handle:
            self._debug_vis_handle.unsubscribe()
            self._debug_vis_handle = None

    """
    Properties
    """

    @property
    def is_initialized(self) -> bool:
        """Whether the asset is initialized.

        Returns True if the asset is initialized, False otherwise.
        """
        return self._is_initialized

    @property
    def num_instances(self) -> int:
        """Number of instances of the asset.

        This is equal to the number of asset instances per environment multiplied by the number of environments.
        """
        return self._num_instances

    @property
    def device(self) -> str:
        """Memory device for computation."""
        return self._device

    @property
    def has_debug_vis_implementation(self) -> bool:
        """Whether the asset has a debug visualization implemented."""

        source_code = inspect.getsource(self._set_debug_vis_impl)
        return "NotImplementedError" not in source_code

    """
    Operations.
    """

    def set_debug_vis(self, debug_vis: bool) -> bool:
        """Sets whether to visualize the asset data.

        Args:
            debug_vis: Whether to visualize the asset data.

        Returns:
            Whether the debug visualization was successfully set. False if the asset
            does not support debug visualization.
        """

        if not self.has_debug_vis_implementation:
            return False

        self._set_debug_vis_impl(debug_vis)

        if debug_vis:

            if self._debug_vis_handle is None:
                app_interface = omni.kit.app.get_app_interface()
                self._debug_vis_handle = app_interface.get_pre_update_event_stream().create_subscription_to_pop(
                    lambda event, obj=weakref.proxy(self): obj._debug_vis_callback(event)
                )
        else:

            if self._debug_vis_handle is not None:
                self._debug_vis_handle.unsubscribe()
                self._debug_vis_handle = None

        return True

    @staticmethod
    def compute_attachment_data(
        isaac_mesh_path, tet_points, tet_indices, sphere_radius=5e-4, max_dist=1e-5
    ):
        """

        Returns: attachment_offsets, attachment_indices and the found rigid prims to the isaac_mesh_path
        """
        print(f"Creating Uipc x Isaac attachments for {isaac_mesh_path}")

        get_physx_interface().force_load_physics_from_usd()









        matching_prims = sim_utils.find_matching_prims(isaac_mesh_path)
        if len(matching_prims) == 0:
            raise RuntimeError(
                f"Could not find prim with path {isaac_mesh_path}. The body_name in the cfg might not exist."
            )
        init_prim = matching_prims[0]

        pose = omni.usd.get_world_transform_matrix(init_prim)
        obj_position = pose.ExtractTranslation()
        obj_position = np.array([obj_position])

        q = pose.ExtractRotation().GetQuaternion()
        obj_orientation = [q.GetReal(), q.GetImaginary()[0], q.GetImaginary()[1], q.GetImaginary()[2]]
        obj_orientation = torch.tensor(np.array([obj_orientation]), device="cuda:0").float()

        idx = []
        attachment_points_positions = []
        attachment_offsets = []


        obj_pos = obj_position[0, :]










        vertex_positions = tet_points


        for i, v in enumerate(vertex_positions):

            ray_dir = [
                0,
                0,
                1,
            ]
            hitInfo = get_physx_scene_query_interface().sweep_sphere_closest(
                radius=sphere_radius, origin=v, dir=ray_dir, distance=max_dist, bothSides=True
            )
            if hitInfo["hit"]:

                if str(init_prim.GetPath()) in hitInfo["collision"]:
                    attachment_points_positions.append(v)

                    idx.append(i)


                    offset = v - obj_pos
                    offset = torch.tensor(offset, device="cuda:0").float()
                    offset = math_utils.quat_apply_inverse(obj_orientation[0].reshape((1, 4)), offset.reshape((1, 3)))[
                        0
                    ]
                    offset = offset.cpu().numpy()
                    attachment_offsets.append(offset)

        attachment_points_positions = np.array(attachment_points_positions).reshape(-1, 3)


        attachment_offsets = np.array(attachment_offsets).reshape(-1, 3)
        assert len(idx) == attachment_offsets.shape[0]














        return attachment_offsets, idx, matching_prims, attachment_points_positions, obj_pos

    """
    Internal helper.
    """

    def _initialize_impl(self):
        if self.cfg.body_name is not None:
            self.rigid_body_id, found_body_name = self.isaaclab_rigid_object.find_bodies(self.cfg.body_name)

        self._create_animation()

        sim: sim_utils.SimulationContext = sim_utils.SimulationContext.instance()
        sim.add_physics_callback(
            f"{self.uipc_object.cfg.prim_path}_X_{self.isaaclab_rigid_object.cfg.prim_path}_attachment_update",
            self._compute_aim_positions,
        )

    def _create_animation(self):
        animator = self.uipc_object._uipc_sim.scene.animator()

        def animate_tet(info: Animation.UpdateInfo):


            geo_slots: list[GeometrySlot] = info.geo_slots()
            geo: SimplicialComplex = geo_slots[0].geometry()



            is_constrained = geo.vertices().find(builtin.is_constrained)
            is_constrained_view = view(is_constrained)
            aim_position = geo.vertices().find(builtin.aim_position)
            aim_position_view = view(aim_position)


            is_constrained_view[self.attachment_points_idx] = 1

            aim_position_view[self.attachment_points_idx] = self.aim_positions.reshape(-1, 3, 1)

        animator.insert(self.uipc_object.uipc_scene_objects[0], animate_tet)

    def _compute_aim_positions(self, dt=0):


        if type(self.isaaclab_rigid_object) is Articulation:



            poses = self.isaaclab_rigid_object._root_physx_view.get_link_transforms().clone()
            poses[..., 3:7] = math_utils.convert_quat(poses[..., 3:7], to="wxyz")
            pose = poses[:, self.rigid_body_id, 0:7].clone()
        elif type(self.isaaclab_rigid_object) is RigidObject:

            pose = self.isaaclab_rigid_object._root_physx_view.root_state_w.view(-1, 1, 13)
            pose = pose[:, self.rigid_body_id, 0:7].clone()
        else:
            raise RuntimeError("Need an Articulation or a RigidBody object for the Isaac X UIPC attachment.")



        self.obj_pose = pose





        attachment_offsets = torch.tensor(
            self.attachment_offsets.reshape((self._num_instances, self.num_attachment_points_per_obj, 3)),
            device=self.device,
        ).float()

        aim_pos = transform_points(
            attachment_offsets, pos=pose[:, 0, 0:3], quat=pose[:, 0, 3:]
        )
        aim_pos = aim_pos.cpu().numpy()
        self.aim_positions = aim_pos.flatten().reshape(-1, 3)






        return self.aim_positions

    """
    Internal simulation callbacks.

    Same as AssetBase class from asset_base.py
    """

    def _initialize_callback(self, event):
        """Initializes the scene elements.

        Note:
            PhysX handles are only enabled once the simulator starts playing. Hence, this function needs to be
            called whenever the simulator "plays" from a "stop" state.
        """
        if not self._is_initialized:

            sim = sim_utils.SimulationContext.instance()
            if sim is None:
                raise RuntimeError("SimulationContext is not initialized! Please initialize SimulationContext first.")
            self._backend = sim.backend
            self._device = sim.device

            self._initialize_impl()

            self._is_initialized = True

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        self._is_initialized = False

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            try:
                from isaacsim.util.debug_draw import _debug_draw

                self._draw = _debug_draw.acquire_debug_draw_interface()
            except ImportError:
                import warnings

                warnings.warn("_debug_draw failed to import", ImportWarning)
                self._draw = None
                print("No debug_vis for attachment. Reason: Cannot import _debug_draw")

    def _debug_vis_callback(self, event):
        if self.aim_positions.shape[0] == 0:
            return




        self._draw.clear_points()
        self._draw.clear_lines()


        self._draw.draw_points(
            self.aim_positions, [(255, 0, 0, 0.5)] * self.aim_positions.shape[0], [60] * self.aim_positions.shape[0]
        )

        pose = self.obj_pose.clone()

        for i in range(self._num_instances):
            obj_center = pose[i, 0, 0:3].cpu().numpy()




            for j in range(i, self._num_instances * self.num_attachment_points_per_obj):
                self._draw.draw_lines([obj_center], [self.aim_positions[j]], [(255, 255, 0, 0.5)], [10])
