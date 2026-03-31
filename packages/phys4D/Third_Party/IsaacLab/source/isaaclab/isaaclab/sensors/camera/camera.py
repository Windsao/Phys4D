




from __future__ import annotations

import json
import numpy as np
import re
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import carb
import isaacsim.core.utils.stage as stage_utils
import omni.kit.commands
import omni.usd
from isaacsim.core.prims import XFormPrim
from isaacsim.core.version import get_version
from pxr import Sdf, UsdGeom

import isaaclab.sim as sim_utils
import isaaclab.utils.sensors as sensor_utils
from isaaclab.utils import to_camel_case
from isaaclab.utils.array import convert_to_torch
from isaaclab.utils.math import (
    convert_camera_frame_orientation_convention,
    create_rotation_matrix_from_view,
    quat_from_matrix,
)

from ..sensor_base import SensorBase
from .camera_data import CameraData

if TYPE_CHECKING:
    from .camera_cfg import CameraCfg


class Camera(SensorBase):
    r"""The camera sensor for acquiring visual data.

    This class wraps over the `UsdGeom Camera`_ for providing a consistent API for acquiring visual data.
    It ensures that the camera follows the ROS convention for the coordinate system.

    Summarizing from the `replicator extension`_, the following sensor types are supported:

    - ``"rgb"``: A 3-channel rendered color image.
    - ``"rgba"``: A 4-channel rendered color image with alpha channel.
    - ``"distance_to_camera"``: An image containing the distance to camera optical center.
    - ``"distance_to_image_plane"``: An image containing distances of 3D points from camera plane along camera's z-axis.
    - ``"depth"``: The same as ``"distance_to_image_plane"``.
    - ``"normals"``: An image containing the local surface normal vectors at each pixel.
    - ``"motion_vectors"``: An image containing the motion vector data at each pixel.
    - ``"semantic_segmentation"``: The semantic segmentation data.
    - ``"instance_segmentation_fast"``: The instance segmentation data.
    - ``"instance_id_segmentation_fast"``: The instance id segmentation data.

    .. note::
        Currently the following sensor types are not supported in a "view" format:

        - ``"instance_segmentation"``: The instance segmentation data. Please use the fast counterparts instead.
        - ``"instance_id_segmentation"``: The instance id segmentation data. Please use the fast counterparts instead.
        - ``"bounding_box_2d_tight"``: The tight 2D bounding box data (only contains non-occluded regions).
        - ``"bounding_box_2d_tight_fast"``: The tight 2D bounding box data (only contains non-occluded regions).
        - ``"bounding_box_2d_loose"``: The loose 2D bounding box data (contains occluded regions).
        - ``"bounding_box_2d_loose_fast"``: The loose 2D bounding box data (contains occluded regions).
        - ``"bounding_box_3d"``: The 3D view space bounding box data.
        - ``"bounding_box_3d_fast"``: The 3D view space bounding box data.

    .. _replicator extension: https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/annotators_details.html#annotator-output
    .. _USDGeom Camera: https://graphics.pixar.com/usd/docs/api/class_usd_geom_camera.html

    """

    cfg: CameraCfg
    """The configuration parameters."""

    UNSUPPORTED_TYPES: set[str] = {
        "instance_id_segmentation",
        "instance_segmentation",
        "bounding_box_2d_tight",
        "bounding_box_2d_loose",
        "bounding_box_3d",
        "bounding_box_2d_tight_fast",
        "bounding_box_2d_loose_fast",
        "bounding_box_3d_fast",
    }
    """The set of sensor types that are not supported by the camera class."""

    def __init__(self, cfg: CameraCfg):
        """Initializes the camera sensor.

        Args:
            cfg: The configuration parameters.

        Raises:
            RuntimeError: If no camera prim is found at the given path.
            ValueError: If the provided data types are not supported by the camera.
        """



        sensor_path = cfg.prim_path.split("/")[-1]
        sensor_path_is_regex = re.match(r"^[a-zA-Z0-9/_]+$", sensor_path) is None
        if sensor_path_is_regex:
            raise RuntimeError(
                f"Invalid prim path for the camera sensor: {self.cfg.prim_path}."
                "\n\tHint: Please ensure that the prim path does not contain any regex patterns in the leaf."
            )

        self._check_supported_data_types(cfg)

        super().__init__(cfg)



        carb_settings_iface = carb.settings.get_settings()
        carb_settings_iface.set_bool("/isaaclab/render/rtx_sensors", True)


        if self.cfg.spawn is not None:

            rot = torch.tensor(self.cfg.offset.rot, dtype=torch.float32, device="cpu").unsqueeze(0)
            rot_offset = convert_camera_frame_orientation_convention(
                rot, origin=self.cfg.offset.convention, target="opengl"
            )
            rot_offset = rot_offset.squeeze(0).cpu().numpy()

            if self.cfg.spawn.vertical_aperture is None:
                self.cfg.spawn.vertical_aperture = self.cfg.spawn.horizontal_aperture * self.cfg.height / self.cfg.width

            self.cfg.spawn.func(
                self.cfg.prim_path, self.cfg.spawn, translation=self.cfg.offset.pos, orientation=rot_offset
            )

        matching_prims = sim_utils.find_matching_prims(self.cfg.prim_path)
        if len(matching_prims) == 0:
            raise RuntimeError(f"Could not find prim with path {self.cfg.prim_path}.")


        self._sensor_prims: list[UsdGeom.Camera] = list()

        self._data = CameraData()


        isaac_sim_version = get_version()

        if int(isaac_sim_version[2]) == 4 and int(isaac_sim_version[3]) == 5:
            if "semantic_segmentation" in self.cfg.data_types or "instance_segmentation_fast" in self.cfg.data_types:
                omni.log.warn(
                    "Isaac Sim 4.5 introduced a bug in Camera and TiledCamera when outputting instance and semantic"
                    " segmentation outputs for instanceable assets. As a workaround, the instanceable flag on assets"
                    " will be disabled in the current workflow and may lead to longer load times and increased memory"
                    " usage."
                )
                with Sdf.ChangeBlock():
                    for prim in self.stage.Traverse():
                        prim.SetInstanceable(False)

    def __del__(self):
        """Unsubscribes from callbacks and detach from the replicator registry."""

        super().__del__()

        for _, annotators in self._rep_registry.items():
            for annotator, render_product_path in zip(annotators, self._render_product_paths):
                annotator.detach([render_product_path])
                annotator = None

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""

        return (
            f"Camera @ '{self.cfg.prim_path}': \n"
            f"\tdata types   : {list(self.data.output.keys())} \n"
            f"\tsemantic filter : {self.cfg.semantic_filter}\n"
            f"\tcolorize semantic segm.   : {self.cfg.colorize_semantic_segmentation}\n"
            f"\tcolorize instance segm.   : {self.cfg.colorize_instance_segmentation}\n"
            f"\tcolorize instance id segm.: {self.cfg.colorize_instance_id_segmentation}\n"
            f"\tupdate period (s): {self.cfg.update_period}\n"
            f"\tshape        : {self.image_shape}\n"
            f"\tnumber of sensors : {self._view.count}"
        )

    """
    Properties
    """

    @property
    def num_instances(self) -> int:
        return self._view.count

    @property
    def data(self) -> CameraData:

        self._update_outdated_buffers()

        return self._data

    @property
    def frame(self) -> torch.tensor:
        """Frame number when the measurement took place."""
        return self._frame

    @property
    def render_product_paths(self) -> list[str]:
        """The path of the render products for the cameras.

        This can be used via replicator interfaces to attach to writes or external annotator registry.
        """
        return self._render_product_paths

    @property
    def image_shape(self) -> tuple[int, int]:
        """A tuple containing (height, width) of the camera sensor."""
        return (self.cfg.height, self.cfg.width)

    """
    Configuration
    """

    def set_intrinsic_matrices(
        self, matrices: torch.Tensor, focal_length: float | None = None, env_ids: Sequence[int] | None = None
    ):
        """Set parameters of the USD camera from its intrinsic matrix.

        The intrinsic matrix is used to set the following parameters to the USD camera:

        - ``focal_length``: The focal length of the camera.
        - ``horizontal_aperture``: The horizontal aperture of the camera.
        - ``vertical_aperture``: The vertical aperture of the camera.
        - ``horizontal_aperture_offset``: The horizontal offset of the camera.
        - ``vertical_aperture_offset``: The vertical offset of the camera.

        .. warning::

            Due to limitations of Omniverse camera, we need to assume that the camera is a spherical lens,
            i.e. has square pixels, and the optical center is centered at the camera eye. If this assumption
            is not true in the input intrinsic matrix, then the camera will not set up correctly.

        Args:
            matrices: The intrinsic matrices for the camera. Shape is (N, 3, 3).
            focal_length: Perspective focal length (in cm) used to calculate pixel size. Defaults to None. If None,
                focal_length will be calculated 1 / width.
            env_ids: A sensor ids to manipulate. Defaults to None, which means all sensor indices.
        """

        if env_ids is None:
            env_ids = self._ALL_INDICES

        if isinstance(matrices, torch.Tensor):
            matrices = matrices.cpu().numpy()
        else:
            matrices = np.asarray(matrices, dtype=float)

        for i, intrinsic_matrix in zip(env_ids, matrices):

            height, width = self.image_shape

            params = sensor_utils.convert_camera_intrinsics_to_usd(
                intrinsic_matrix=intrinsic_matrix.reshape(-1), height=height, width=width, focal_length=focal_length
            )


            sensor_prim = self._sensor_prims[i]

            for param_name, param_value in params.items():

                param_name = to_camel_case(param_name, to="CC")

                param_attr = getattr(sensor_prim, f"Get{param_name}Attr")




                omni.usd.set_prop_val(param_attr(), param_value)

        self._update_intrinsic_matrices(env_ids)

    """
    Operations - Set pose.
    """

    def set_world_poses(
        self,
        positions: torch.Tensor | None = None,
        orientations: torch.Tensor | None = None,
        env_ids: Sequence[int] | None = None,
        convention: Literal["opengl", "ros", "world"] = "ros",
    ):
        r"""Set the pose of the camera w.r.t. the world frame using specified convention.

        Since different fields use different conventions for camera orientations, the method allows users to
        set the camera poses in the specified convention. Possible conventions are:

        - :obj:`"opengl"` - forward axis: -Z - up axis +Y - Offset is applied in the OpenGL (Usd.Camera) convention
        - :obj:`"ros"`    - forward axis: +Z - up axis -Y - Offset is applied in the ROS convention
        - :obj:`"world"`  - forward axis: +X - up axis +Z - Offset is applied in the World Frame convention

        See :meth:`isaaclab.sensors.camera.utils.convert_camera_frame_orientation_convention` for more details
        on the conventions.

        Args:
            positions: The cartesian coordinates (in meters). Shape is (N, 3).
                Defaults to None, in which case the camera position in not changed.
            orientations: The quaternion orientation in (w, x, y, z). Shape is (N, 4).
                Defaults to None, in which case the camera orientation in not changed.
            env_ids: A sensor ids to manipulate. Defaults to None, which means all sensor indices.
            convention: The convention in which the poses are fed. Defaults to "ros".

        Raises:
            RuntimeError: If the camera prim is not set. Need to call :meth:`initialize` method first.
        """

        if env_ids is None:
            env_ids = self._ALL_INDICES

        if positions is not None:
            if isinstance(positions, np.ndarray):
                positions = torch.from_numpy(positions).to(device=self._device)
            elif not isinstance(positions, torch.Tensor):
                positions = torch.tensor(positions, device=self._device)

        if orientations is not None:
            if isinstance(orientations, np.ndarray):
                orientations = torch.from_numpy(orientations).to(device=self._device)
            elif not isinstance(orientations, torch.Tensor):
                orientations = torch.tensor(orientations, device=self._device)
            orientations = convert_camera_frame_orientation_convention(orientations, origin=convention, target="opengl")

        self._view.set_world_poses(positions, orientations, env_ids)

    def set_world_poses_from_view(
        self, eyes: torch.Tensor, targets: torch.Tensor, env_ids: Sequence[int] | None = None
    ):
        """Set the poses of the camera from the eye position and look-at target position.

        Args:
            eyes: The positions of the camera's eye. Shape is (N, 3).
            targets: The target locations to look at. Shape is (N, 3).
            env_ids: A sensor ids to manipulate. Defaults to None, which means all sensor indices.

        Raises:
            RuntimeError: If the camera prim is not set. Need to call :meth:`initialize` method first.
            NotImplementedError: If the stage up-axis is not "Y" or "Z".
        """

        if env_ids is None:
            env_ids = self._ALL_INDICES

        up_axis = stage_utils.get_stage_up_axis()

        orientations = quat_from_matrix(create_rotation_matrix_from_view(eyes, targets, up_axis, device=self._device))
        self._view.set_world_poses(eyes, orientations, env_ids)

    """
    Operations
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        if not self._is_initialized:
            raise RuntimeError(
                "Camera could not be initialized. Please ensure --enable_cameras is used to enable rendering."
            )

        super().reset(env_ids)


        if env_ids is None:
            env_ids = self._ALL_INDICES


        self._update_poses(env_ids)

        self._frame[env_ids] = 0

    """
    Implementation.
    """

    def _initialize_impl(self):
        """Initializes the sensor handles and internal buffers.

        This function creates handles and registers the provided data types with the replicator registry to
        be able to access the data from the sensor. It also initializes the internal buffers to store the data.

        Raises:
            RuntimeError: If the number of camera prims in the view does not match the number of environments.
            RuntimeError: If replicator was not found.
        """
        carb_settings_iface = carb.settings.get_settings()
        if not carb_settings_iface.get("/isaaclab/cameras_enabled"):
            raise RuntimeError(
                "A camera was spawned without the --enable_cameras flag. Please use --enable_cameras to enable"
                " rendering."
            )

        import omni.replicator.core as rep
        from omni.syntheticdata.scripts.SyntheticData import SyntheticData


        super()._initialize_impl()

        self._view = XFormPrim(self.cfg.prim_path, reset_xform_properties=False)
        self._view.initialize()

        if self._view.count != self._num_envs:
            raise RuntimeError(
                f"Number of camera prims in the view ({self._view.count}) does not match"
                f" the number of environments ({self._num_envs})."
            )


        self._ALL_INDICES = torch.arange(self._view.count, device=self._device, dtype=torch.long)

        self._frame = torch.zeros(self._view.count, device=self._device, dtype=torch.long)


        self._render_product_paths: list[str] = list()
        self._rep_registry: dict[str, list[rep.annotators.Annotator]] = {name: list() for name in self.cfg.data_types}


        for cam_prim_path in self._view.prim_paths:

            cam_prim = self.stage.GetPrimAtPath(cam_prim_path)

            if not cam_prim.IsA(UsdGeom.Camera):
                raise RuntimeError(f"Prim at path '{cam_prim_path}' is not a Camera.")

            sensor_prim = UsdGeom.Camera(cam_prim)
            self._sensor_prims.append(sensor_prim)



            render_prod_path = rep.create.render_product(cam_prim_path, resolution=(self.cfg.width, self.cfg.height))
            if not isinstance(render_prod_path, str):
                render_prod_path = render_prod_path.path
            self._render_product_paths.append(render_prod_path)


            if isinstance(self.cfg.semantic_filter, list):
                semantic_filter_predicate = ":*; ".join(self.cfg.semantic_filter) + ":*"
            elif isinstance(self.cfg.semantic_filter, str):
                semantic_filter_predicate = self.cfg.semantic_filter
            else:
                raise ValueError(f"Semantic types must be a list or a string. Received: {self.cfg.semantic_filter}.")


            SyntheticData.Get().set_instance_mapping_semantic_filter(semantic_filter_predicate)




            for name in self.cfg.data_types:



                if name == "semantic_segmentation":
                    init_params = {
                        "colorize": self.cfg.colorize_semantic_segmentation,
                        "mapping": json.dumps(self.cfg.semantic_segmentation_mapping),
                    }
                elif name == "instance_segmentation_fast":
                    init_params = {"colorize": self.cfg.colorize_instance_segmentation}
                elif name == "instance_id_segmentation_fast":
                    init_params = {"colorize": self.cfg.colorize_instance_id_segmentation}
                else:
                    init_params = None


                if "cuda" in self._device:
                    device_name = self._device.split(":")[0]
                else:
                    device_name = "cpu"


                special_cases = {"rgba": "rgb", "depth": "distance_to_image_plane"}

                annotator_name = special_cases.get(name, name)

                rep_annotator = rep.AnnotatorRegistry.get_annotator(annotator_name, init_params, device=device_name)


                rep_annotator.attach(render_prod_path)

                self._rep_registry[name].append(rep_annotator)


        self._create_buffers()
        self._update_intrinsic_matrices(self._ALL_INDICES)

    def _update_buffers_impl(self, env_ids: Sequence[int]):

        self._frame[env_ids] += 1

        if self.cfg.update_latest_camera_pose:
            self._update_poses(env_ids)


        if len(self._data.output) == 0:


            self._create_annotator_data()
        else:

            for name, annotators in self._rep_registry.items():

                for index in env_ids:

                    output = annotators[index].get_data()

                    data, info = self._process_annotator_output(name, output)

                    self._data.output[name][index] = data

                    self._data.info[index][name] = info




                if name == "distance_to_camera":
                    self._data.output[name][self._data.output[name] > self.cfg.spawn.clipping_range[1]] = torch.inf

                if (
                    name == "distance_to_camera" or name == "distance_to_image_plane"
                ) and self.cfg.depth_clipping_behavior != "none":
                    self._data.output[name][torch.isinf(self._data.output[name])] = (
                        0.0 if self.cfg.depth_clipping_behavior == "zero" else self.cfg.spawn.clipping_range[1]
                    )

    """
    Private Helpers
    """

    def _check_supported_data_types(self, cfg: CameraCfg):
        """Checks if the data types are supported by the ray-caster camera."""


        common_elements = set(cfg.data_types) & Camera.UNSUPPORTED_TYPES
        if common_elements:

            fast_common_elements = []
            for item in common_elements:
                if "instance_segmentation" in item or "instance_id_segmentation" in item:
                    fast_common_elements.append(item + "_fast")

            raise ValueError(
                f"Camera class does not support the following sensor types: {common_elements}."
                "\n\tThis is because these sensor types output numpy structured data types which"
                "can't be converted to torch tensors easily."
                "\n\tHint: If you need to work with these sensor types, we recommend using their fast counterparts."
                f"\n\t\tFast counterparts: {fast_common_elements}"
            )

    def _create_buffers(self):
        """Create buffers for storing data."""


        self._data.pos_w = torch.zeros((self._view.count, 3), device=self._device)
        self._data.quat_w_world = torch.zeros((self._view.count, 4), device=self._device)

        self._data.intrinsic_matrices = torch.zeros((self._view.count, 3, 3), device=self._device)
        self._data.image_shape = self.image_shape




        self._data.output = {}
        self._data.info = [{name: None for name in self.cfg.data_types} for _ in range(self._view.count)]

    def _update_intrinsic_matrices(self, env_ids: Sequence[int]):
        """Compute camera's matrix of intrinsic parameters.

        Also called calibration matrix. This matrix works for linear depth images. We assume square pixels.

        Note:
            The calibration matrix projects points in the 3D scene onto an imaginary screen of the camera.
            The coordinates of points on the image plane are in the homogeneous representation.
        """

        for i in env_ids:

            sensor_prim = self._sensor_prims[i]


            focal_length = sensor_prim.GetFocalLengthAttr().Get()
            horiz_aperture = sensor_prim.GetHorizontalApertureAttr().Get()


            height, width = self.image_shape

            f_x = (width * focal_length) / horiz_aperture
            f_y = f_x
            c_x = width * 0.5
            c_y = height * 0.5

            self._data.intrinsic_matrices[i, 0, 0] = f_x
            self._data.intrinsic_matrices[i, 0, 2] = c_x
            self._data.intrinsic_matrices[i, 1, 1] = f_y
            self._data.intrinsic_matrices[i, 1, 2] = c_y
            self._data.intrinsic_matrices[i, 2, 2] = 1

    def _update_poses(self, env_ids: Sequence[int]):
        """Computes the pose of the camera in the world frame with ROS convention.

        This methods uses the ROS convention to resolve the input pose. In this convention,
        we assume that the camera front-axis is +Z-axis and up-axis is -Y-axis.

        Returns:
            A tuple of the position (in meters) and quaternion (w, x, y, z).
        """

        if len(self._sensor_prims) == 0:
            raise RuntimeError("Camera prim is None. Please call 'sim.play()' first.")


        poses, quat = self._view.get_world_poses(env_ids)
        self._data.pos_w[env_ids] = poses
        self._data.quat_w_world[env_ids] = convert_camera_frame_orientation_convention(
            quat, origin="opengl", target="world"
        )

    def _create_annotator_data(self):
        """Create the buffers to store the annotator data.

        We create a buffer for each annotator and store the data in a dictionary. Since the data
        shape is not known beforehand, we create a list of buffers and concatenate them later.

        This is an expensive operation and should be called only once.
        """

        for name, annotators in self._rep_registry.items():

            data_all_cameras = list()

            for index in self._ALL_INDICES:

                output = annotators[index].get_data()

                data, info = self._process_annotator_output(name, output)

                data_all_cameras.append(data)

                self._data.info[index][name] = info

            self._data.output[name] = torch.stack(data_all_cameras, dim=0)




            if name == "distance_to_camera":
                self._data.output[name][self._data.output[name] > self.cfg.spawn.clipping_range[1]] = torch.inf

            if (
                name == "distance_to_camera" or name == "distance_to_image_plane"
            ) and self.cfg.depth_clipping_behavior != "none":
                self._data.output[name][torch.isinf(self._data.output[name])] = (
                    0.0 if self.cfg.depth_clipping_behavior == "zero" else self.cfg.spawn.clipping_range[1]
                )

    def _process_annotator_output(self, name: str, output: Any) -> tuple[torch.tensor, dict | None]:
        """Process the annotator output.

        This function is called after the data has been collected from all the cameras.
        """

        if isinstance(output, dict):
            data = output["data"]
            info = output["info"]
        else:
            data = output
            info = None

        data = convert_to_torch(data, device=self.device)




        height, width = self.image_shape
        if name == "semantic_segmentation":
            if self.cfg.colorize_semantic_segmentation:
                data = data.view(torch.uint8).reshape(height, width, -1)
            else:
                data = data.view(height, width, 1)
        elif name == "instance_segmentation_fast":
            if self.cfg.colorize_instance_segmentation:
                data = data.view(torch.uint8).reshape(height, width, -1)
            else:
                data = data.view(height, width, 1)
        elif name == "instance_id_segmentation_fast":
            if self.cfg.colorize_instance_id_segmentation:
                data = data.view(torch.uint8).reshape(height, width, -1)
            else:
                data = data.view(height, width, 1)

        elif name == "distance_to_camera" or name == "distance_to_image_plane" or name == "depth":
            data = data.view(height, width, 1)


        elif name == "rgb" or name == "normals":
            data = data[..., :3]

        elif name == "motion_vectors":
            data = data[..., :2]


        return data, info

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""

        super()._invalidate_initialize_callback(event)

        self._view = None
