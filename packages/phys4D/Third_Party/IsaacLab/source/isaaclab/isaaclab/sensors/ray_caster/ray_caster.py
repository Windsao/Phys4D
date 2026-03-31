




from __future__ import annotations

import numpy as np
import re
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log
import omni.physics.tensors.impl.api as physx
import warp as wp
from isaacsim.core.prims import XFormPrim
from isaacsim.core.simulation_manager import SimulationManager
from pxr import UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.utils.math import convert_quat, quat_apply, quat_apply_yaw
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh

from ..sensor_base import SensorBase
from .ray_caster_data import RayCasterData

if TYPE_CHECKING:
    from .ray_caster_cfg import RayCasterCfg


class RayCaster(SensorBase):
    """A ray-casting sensor.

    The ray-caster uses a set of rays to detect collisions with meshes in the scene. The rays are
    defined in the sensor's local coordinate frame. The sensor can be configured to ray-cast against
    a set of meshes with a given ray pattern.

    The meshes are parsed from the list of primitive paths provided in the configuration. These are then
    converted to warp meshes and stored in the `warp_meshes` list. The ray-caster then ray-casts against
    these warp meshes using the ray pattern provided in the configuration.

    .. note::
        Currently, only static meshes are supported. Extending the warp mesh to support dynamic meshes
        is a work in progress.
    """

    cfg: RayCasterCfg
    """The configuration parameters."""

    def __init__(self, cfg: RayCasterCfg):
        """Initializes the ray-caster object.

        Args:
            cfg: The configuration parameters.
        """



        sensor_path = cfg.prim_path.split("/")[-1]
        sensor_path_is_regex = re.match(r"^[a-zA-Z0-9/_]+$", sensor_path) is None
        if sensor_path_is_regex:
            raise RuntimeError(
                f"Invalid prim path for the ray-caster sensor: {self.cfg.prim_path}."
                "\n\tHint: Please ensure that the prim path does not contain any regex patterns in the leaf."
            )

        super().__init__(cfg)

        self._data = RayCasterData()

        self.meshes: dict[str, wp.Mesh] = {}

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"Ray-caster @ '{self.cfg.prim_path}': \n"
            f"\tview type            : {self._view.__class__}\n"
            f"\tupdate period (s)    : {self.cfg.update_period}\n"
            f"\tnumber of meshes     : {len(self.meshes)}\n"
            f"\tnumber of sensors    : {self._view.count}\n"
            f"\tnumber of rays/sensor: {self.num_rays}\n"
            f"\ttotal number of rays : {self.num_rays * self._view.count}"
        )

    """
    Properties
    """

    @property
    def num_instances(self) -> int:
        return self._view.count

    @property
    def data(self) -> RayCasterData:

        self._update_outdated_buffers()

        return self._data

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None):

        super().reset(env_ids)

        if env_ids is None:
            env_ids = slice(None)
            num_envs_ids = self._view.count
        else:
            num_envs_ids = len(env_ids)

        r = torch.empty(num_envs_ids, 3, device=self.device)
        self.drift[env_ids] = r.uniform_(*self.cfg.drift_range)

        range_list = [self.cfg.ray_cast_drift_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
        ranges = torch.tensor(range_list, device=self.device)
        self.ray_cast_drift[env_ids] = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (num_envs_ids, 3), device=self.device
        )

    """
    Implementation.
    """

    def _initialize_impl(self):
        super()._initialize_impl()

        self._physics_sim_view = SimulationManager.get_physics_sim_view()



        found_supported_prim_class = False
        prim = sim_utils.find_first_matching_prim(self.cfg.prim_path)
        if prim is None:
            raise RuntimeError(f"Failed to find a prim at path expression: {self.cfg.prim_path}")

        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            self._view = self._physics_sim_view.create_articulation_view(self.cfg.prim_path.replace(".*", "*"))
            found_supported_prim_class = True
        elif prim.HasAPI(UsdPhysics.RigidBodyAPI):
            self._view = self._physics_sim_view.create_rigid_body_view(self.cfg.prim_path.replace(".*", "*"))
            found_supported_prim_class = True
        else:
            self._view = XFormPrim(self.cfg.prim_path, reset_xform_properties=False)
            found_supported_prim_class = True
            omni.log.warn(f"The prim at path {prim.GetPath().pathString} is not a physics prim! Using XFormPrim.")

        if not found_supported_prim_class:
            raise RuntimeError(f"Failed to find a valid prim view class for the prim paths: {self.cfg.prim_path}")


        self._initialize_warp_meshes()

        self._initialize_rays_impl()

    def _initialize_warp_meshes(self):

        if len(self.cfg.mesh_prim_paths) != 1:
            raise NotImplementedError(
                f"RayCaster currently only supports one mesh prim. Received: {len(self.cfg.mesh_prim_paths)}"
            )


        for mesh_prim_path in self.cfg.mesh_prim_paths:


            mesh_prim = sim_utils.get_first_matching_child_prim(
                mesh_prim_path, lambda prim: prim.GetTypeName() == "Plane"
            )

            if mesh_prim is None:

                mesh_prim = sim_utils.get_first_matching_child_prim(
                    mesh_prim_path, lambda prim: prim.GetTypeName() == "Mesh"
                )

                if mesh_prim is None or not mesh_prim.IsValid():
                    raise RuntimeError(f"Invalid mesh prim path: {mesh_prim_path}")

                mesh_prim = UsdGeom.Mesh(mesh_prim)

                points = np.asarray(mesh_prim.GetPointsAttr().Get())
                transform_matrix = np.array(omni.usd.get_world_transform_matrix(mesh_prim)).T
                points = np.matmul(points, transform_matrix[:3, :3].T)
                points += transform_matrix[:3, 3]
                indices = np.asarray(mesh_prim.GetFaceVertexIndicesAttr().Get())
                wp_mesh = convert_to_warp_mesh(points, indices, device=self.device)

                omni.log.info(
                    f"Read mesh prim: {mesh_prim.GetPath()} with {len(points)} vertices and {len(indices)} faces."
                )
            else:
                mesh = make_plane(size=(2e6, 2e6), height=0.0, center_zero=True)
                wp_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=self.device)

                omni.log.info(f"Created infinite plane mesh prim: {mesh_prim.GetPath()}.")

            self.meshes[mesh_prim_path] = wp_mesh


        if all([mesh_prim_path not in self.meshes for mesh_prim_path in self.cfg.mesh_prim_paths]):
            raise RuntimeError(
                f"No meshes found for ray-casting! Please check the mesh prim paths: {self.cfg.mesh_prim_paths}"
            )

    def _initialize_rays_impl(self):

        self.ray_starts, self.ray_directions = self.cfg.pattern_cfg.func(self.cfg.pattern_cfg, self._device)
        self.num_rays = len(self.ray_directions)

        offset_pos = torch.tensor(list(self.cfg.offset.pos), device=self._device)
        offset_quat = torch.tensor(list(self.cfg.offset.rot), device=self._device)
        self.ray_directions = quat_apply(offset_quat.repeat(len(self.ray_directions), 1), self.ray_directions)
        self.ray_starts += offset_pos

        self.ray_starts = self.ray_starts.repeat(self._view.count, 1, 1)
        self.ray_directions = self.ray_directions.repeat(self._view.count, 1, 1)

        self.drift = torch.zeros(self._view.count, 3, device=self.device)
        self.ray_cast_drift = torch.zeros(self._view.count, 3, device=self.device)

        self._data.pos_w = torch.zeros(self._view.count, 3, device=self._device)
        self._data.quat_w = torch.zeros(self._view.count, 4, device=self._device)
        self._data.ray_hits_w = torch.zeros(self._view.count, self.num_rays, 3, device=self._device)

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Fills the buffers of the sensor data."""

        if isinstance(self._view, XFormPrim):
            pos_w, quat_w = self._view.get_world_poses(env_ids)
        elif isinstance(self._view, physx.ArticulationView):
            pos_w, quat_w = self._view.get_root_transforms()[env_ids].split([3, 4], dim=-1)
            quat_w = convert_quat(quat_w, to="wxyz")
        elif isinstance(self._view, physx.RigidBodyView):
            pos_w, quat_w = self._view.get_transforms()[env_ids].split([3, 4], dim=-1)
            quat_w = convert_quat(quat_w, to="wxyz")
        else:
            raise RuntimeError(f"Unsupported view type: {type(self._view)}")

        pos_w = pos_w.clone()
        quat_w = quat_w.clone()

        pos_w += self.drift[env_ids]

        self._data.pos_w[env_ids] = pos_w
        self._data.quat_w[env_ids] = quat_w


        if self.cfg.attach_yaw_only is not None:
            msg = (
                "Raycaster attribute 'attach_yaw_only' property will be deprecated in a future release."
                " Please use the parameter 'ray_alignment' instead."
            )

            if self.cfg.attach_yaw_only:
                self.cfg.ray_alignment = "yaw"
                msg += " Setting ray_alignment to 'yaw'."
            else:
                self.cfg.ray_alignment = "base"
                msg += " Setting ray_alignment to 'base'."

            omni.log.warn(msg)

        if self.cfg.ray_alignment == "world":

            pos_w[:, 0:2] += self.ray_cast_drift[env_ids, 0:2]

            ray_starts_w = self.ray_starts[env_ids]
            ray_starts_w += pos_w.unsqueeze(1)
            ray_directions_w = self.ray_directions[env_ids]
        elif self.cfg.ray_alignment == "yaw":

            pos_w[:, 0:2] += quat_apply_yaw(quat_w, self.ray_cast_drift[env_ids])[:, 0:2]

            ray_starts_w = quat_apply_yaw(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids])
            ray_starts_w += pos_w.unsqueeze(1)
            ray_directions_w = self.ray_directions[env_ids]
        elif self.cfg.ray_alignment == "base":

            pos_w[:, 0:2] += quat_apply(quat_w, self.ray_cast_drift[env_ids])[:, 0:2]

            ray_starts_w = quat_apply(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids])
            ray_starts_w += pos_w.unsqueeze(1)
            ray_directions_w = quat_apply(quat_w.repeat(1, self.num_rays), self.ray_directions[env_ids])
        else:
            raise RuntimeError(f"Unsupported ray_alignment type: {self.cfg.ray_alignment}.")



        self._data.ray_hits_w[env_ids] = raycast_mesh(
            ray_starts_w,
            ray_directions_w,
            max_dist=self.cfg.max_distance,
            mesh=self.meshes[self.cfg.mesh_prim_paths[0]],
        )[0]


        self._data.ray_hits_w[env_ids, :, 2] += self.ray_cast_drift[env_ids, 2].unsqueeze(-1)

    def _set_debug_vis_impl(self, debug_vis: bool):


        if debug_vis:
            if not hasattr(self, "ray_visualizer"):
                self.ray_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)

            self.ray_visualizer.set_visibility(True)
        else:
            if hasattr(self, "ray_visualizer"):
                self.ray_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):

        viz_points = self._data.ray_hits_w.reshape(-1, 3)
        viz_points = viz_points[~torch.any(torch.isinf(viz_points), dim=1)]

        self.ray_visualizer.visualize(viz_points)

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""

        super()._invalidate_initialize_callback(event)

        self._view = None
