from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence
from matplotlib import pyplot as plt
from typing import TYPE_CHECKING

import cv2
import omni.kit.commands
import omni.usd
from isaacsim.core.prims import XFormPrim
from pxr import Sdf

from isaaclab.sensors import SensorBase, TiledCamera, TiledCameraCfg

from .gelsight_sensor_data import GelSightSensorData
from .simulation_approaches.gelsight_simulator import GelSightSimulator








if TYPE_CHECKING:
    from .gelsight_sensor_cfg import GelSightSensorCfg


class GelSightSensor(SensorBase):
    cfg: GelSightSensorCfg

    def __init__(self, cfg: GelSightSensorCfg, gelpad_obj=None):

        super().__init__(cfg)

        self._prim_view = None


        self.camera = None


        self.gelpad_obj = gelpad_obj

        self._indentation_depth: torch.tensor = None


        self.optical_simulator: GelSightSimulator = None
        self.marker_motion_simulator: GelSightSimulator = None
        self.compute_indentation_depth_func = None


        self._data = GelSightSensorData()
        self._data.output = dict.fromkeys(self.cfg.data_types, None)


        self._is_spawned = False


        if self.cfg.optical_sim_cfg is not None:

            self.optical_simulator = self.cfg.optical_sim_cfg.simulation_approach_class(
                sensor=self,
                cfg=self.cfg.optical_sim_cfg,
            )

        if self.cfg.marker_motion_sim_cfg is not None:
            if (self.optical_simulator is not None) and (
                self.cfg.optical_sim_cfg.simulation_approach_class
                == self.cfg.marker_motion_sim_cfg.simulation_approach_class
            ):

                self.marker_motion_simulator = self.optical_simulator
            else:
                self.marker_motion_simulator = self.cfg.marker_motion_sim_cfg.simulation_approach_class(
                    sensor=self, cfg=self.cfg.marker_motion_sim_cfg
                )

        self._set_debug_vis_flag = False
        self._debug_vis_is_initialized = False

    def __del__(self):
        """Unsubscribes from callbacks."""

        super().__del__()
















    """
    Properties
    """

    @property
    def data(self) -> GelSightSensorData:
        """Data related to Camera sensor."""

        self._update_outdated_buffers()
        return self._data

    @property
    def frame(self) -> torch.tensor:
        """Frame number when the measurement took place."""
        return self._frame

    @property
    def tactile_image_shape(self) -> tuple[int, int, int]:
        """Shape of the simulated tactile RGB image, i.e. (channels, height, width)."""
        return (self.cfg.optical_sim_cfg.tactile_img_res[1], self.cfg.optical_sim_cfg.tactile_img_res[0], 3)

    @property
    def camera_resolution(self) -> tuple[int, int]:
        """The resolution (width x height) of the camera used by this sensor."""
        return self.cfg.sensor_camera_cfg.resolution[0], self.cfg.sensor_camera_cfg.resolution[1]

    @property
    def indentation_depth(self):
        """How deep objects are inside the gel pad of the sensor.

        Units: [mm]
        """
        return self._indentation_depth

    @property
    def prim_view(self):
        return self._prim_view

    """
    Operations
    """


    def reset(self, env_ids: Sequence[int] | None = None):

        super().reset(env_ids)


        if env_ids is None:
            env_ids = self._ALL_INDICES


        if self.camera is not None:
            self.camera.reset()






        self._indentation_depth[env_ids] = 0


        self._data.output["height_map"][env_ids] = 0










        if "camera_depth" in self._data.output:
            self._data.output["camera_depth"][env_ids] = 0


        if (self.optical_simulator is not None) and ("tactile_rgb" in self._data.output):
            self._data.output["tactile_rgb"][:] = self.optical_simulator.optical_simulation()
            self.optical_simulator.reset()

        if (self.marker_motion_simulator is not None) and ("marker_motion" in self._data.output):

            self._data.output["marker_motion"][:] = self.marker_motion_simulator.marker_motion_simulation()

            self._data.output["init_marker_pos"] = ([0], [0])

            self.marker_motion_simulator.reset()


        self._frame[env_ids] = 0





    def _initialize_impl(self):
        """Initializes the sensor handles and internal buffers."""
        print(f"Initializing GelSight Sensor `{self.cfg.prim_path}`...")


        super()._initialize_impl()

        self._prim_view = XFormPrim(prim_paths_expr=self.cfg.prim_path, name=f"{self.cfg.prim_path}", usd=False)
        self._prim_view.initialize()

        if self._prim_view.count != self._num_envs:
            raise RuntimeError(
                f"Number of sensor prims in the view ({self._prim_view.count}) does not match"
                f" the number of environments ({self._num_envs})."
            )


        if self.cfg.device is not None:
            self._device = self.cfg.device


        self._ALL_INDICES = torch.arange(self._num_envs, device=self._device, dtype=torch.long)

        self._frame = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)

        self._indentation_depth = torch.zeros((self._num_envs), device=self._device)

        if self.cfg.sensor_camera_cfg is not None:
            self.camera_cfg: TiledCameraCfg = TiledCameraCfg(
                prim_path=self.cfg.prim_path + self.cfg.sensor_camera_cfg.prim_path_appendix,
                update_period=self.cfg.sensor_camera_cfg.update_period,
                height=self.cfg.sensor_camera_cfg.resolution[1],
                width=self.cfg.sensor_camera_cfg.resolution[0],
                data_types=self.cfg.sensor_camera_cfg.data_types,
                update_latest_camera_pose=True,
                spawn=None,





            )
            self.camera = TiledCamera(cfg=self.camera_cfg)

















            self.camera._initialize_impl()
            self.camera._is_initialized = True

        self._data.output["height_map"] = torch.zeros(
            (self._num_envs, self.camera_cfg.height, self.camera_cfg.width), device=self.cfg.device
        )








        if self.optical_simulator is not None:
            self.optical_simulator._initialize_impl()

        if self.marker_motion_simulator is not None:
            self.marker_motion_simulator._initialize_impl()


        if "camera_depth" in self.cfg.data_types:
            self._data.output["camera_depth"] = torch.zeros(
                (self._num_envs, self.camera_resolution[1], self.camera_resolution[0], 1), device=self.cfg.device
            )
        if "camera_rgb" in self.cfg.data_types:
            self._data.output["camera_rgb"] = torch.zeros(
                (self._num_envs, self.camera_resolution[1], self.camera_resolution[0], 3), device=self.cfg.device
            )
        if "tactile_rgb" in self.cfg.data_types:
            self._data.output["tactile_rgb"] = torch.zeros(
                (
                    self._num_envs,
                    self.cfg.optical_sim_cfg.tactile_img_res[1],
                    self.cfg.optical_sim_cfg.tactile_img_res[0],
                    3,
                ),
                device=self.cfg.device,
            )
        if "marker_motion" in self.cfg.data_types:









            self._data.output["marker_motion"] = torch.zeros(
                (
                    self._num_envs,
                    2,
                    self.cfg.marker_motion_sim_cfg.marker_params.num_markers,
                    2,
                ),
                device=self.cfg.device,
            )


        if (self.cfg.compute_indentation_depth_class) == "optical_sim" and (self.optical_simulator is not None):
            self.compute_indentation_depth_func = self.optical_simulator.compute_indentation_depth
        elif (self.cfg.compute_indentation_depth_class == "marker_motion_sim") and (
            self.marker_motion_simulator is not None
        ):
            self.compute_indentation_depth_func = self.marker_motion_simulator.compute_indentation_depth
        else:
            self.compute_indentation_depth_func = None


        self._ALL_INDICES = torch.arange(self._num_envs, device=self._device, dtype=torch.long)

        self._frame = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)


        self.reset()


        self._initialize_debug_vis(self._initialize_debug_vis_flag)





    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Updates the internal buffer with the latest data from the sensor.

        This function reads ...

        """






        self._frame[env_ids] += 1


        if self.camera is not None:
            self.camera._timestamp = self._timestamp
            self.camera.update(dt=0, force_recompute=True)

        if self.compute_indentation_depth_func is not None:

            self._get_height_map()

            self._indentation_depth[:] = self.compute_indentation_depth_func()

        if "camera_depth" in self._data.output:
            self._get_camera_depth()

        if "camera_rgb" in self._data.output:
            self._data.output["camera_rgb"][:] = self.camera.data.output["rgb"]

        if (self.optical_simulator is not None) and ("tactile_rgb" in self.cfg.data_types):

            self._data.output["tactile_rgb"][:] = self.optical_simulator.optical_simulation()

        if (self.marker_motion_simulator is not None) and ("marker_motion" in self.cfg.data_types):
            self._data.output["marker_motion"][:] = self.marker_motion_simulator.marker_motion_simulation()

    def _set_debug_vis_impl(self, debug_vis: bool):


        self._initialize_debug_vis_flag: bool = debug_vis

    def _initialize_debug_vis(self, debug_vis: bool):
        """Creates an USD attribute for the sensor assets, which can visualize the tactile image.

        Select the GelSight sensor case whose output you want to see in the Isaac Sim GUI,
        i.e. the `gelsight_mini_case` Xform (not the mesh!).
        Scroll down in the properties panel to "Raw Usd Properties" and click "Extra Properties".
        There is an attribute called "show_tactile_image".
        Toggle it on to show the sensor output in the GUI.

        If only optical simulation is used, then only an optical img is displayed.
        If only the marker simulatios is used, then only an image displaying the marker positions is displayed.
        If both, optical and marker simulation, are used, then the images are overlaid.

        > Method has to be called after the prim_view was initialized.
        """

        if debug_vis:

            for prim in self._prim_view.prims:

                if "camera_depth" in self.cfg.data_types:
                    attr = prim.CreateAttribute("debug_camera_depth", Sdf.ValueTypeNames.Bool)
                    attr.Set(False)
                if "camera_rgb" in self.cfg.data_types:
                    attr = prim.CreateAttribute("debug_camera_rgb", Sdf.ValueTypeNames.Bool)
                    attr.Set(False)
                if "tactile_rgb" in self.cfg.data_types:
                    attr = prim.CreateAttribute("debug_tactile_rgb", Sdf.ValueTypeNames.Bool)
                    attr.Set(False)
                if "marker_motion" in self.cfg.data_types:
                    attr = prim.CreateAttribute("debug_marker_motion", Sdf.ValueTypeNames.Bool)
                    attr.Set(False)

            if not hasattr(self, "_windows"):

                self._windows = {}
                self._img_providers = {}

                if "camera_depth" in self.cfg.data_types:
                    self._windows["camera_depth"] = {}
                    self._img_providers["camera_depth"] = {}
                if "camera_rgb" in self.cfg.data_types:
                    self._windows["camera_rgb"] = {}
                    self._img_providers["camera_rgb"] = {}

            if "tactile_rgb" in self.cfg.data_types:
                self.optical_simulator._set_debug_vis_impl(debug_vis)

            if "marker_motion" in self.cfg.data_types:
                self.marker_motion_simulator._set_debug_vis_impl(debug_vis)

            self._debug_vis_is_initialized = True

    def _debug_vis_callback(self, event):
        if not self._debug_vis_is_initialized:
            return


        for i, prim in enumerate(
            self._prim_view.prims
        ):
            if "camera_rgb" in self.cfg.data_types:
                show_img = prim.GetAttribute("debug_camera_rgb").Get()
                if show_img:
                    if str(i) not in self._windows["camera_rgb"]:

                        window = omni.ui.Window(
                            self._prim_view.prim_paths[i] + "/camera_rgb",
                            height=self.cfg.sensor_camera_cfg.resolution[1],
                            width=self.cfg.sensor_camera_cfg.resolution[0],
                        )
                        self._windows["camera_rgb"][str(i)] = window

                        self._img_providers["camera_rgb"][
                            str(i)
                        ] = omni.ui.ByteImageProvider()

                    frame = self._data.output["camera_rgb"][i].cpu().numpy()


                    frame = frame.astype(np.uint8)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
                    height, width, channels = frame.shape
                    with self._windows["camera_rgb"][str(i)].frame:

                        self._img_providers["camera_rgb"][str(i)].set_bytes_data(
                            frame.flatten().data, [width, height]
                        )
                        omni.ui.ImageWithProvider(
                            self._img_providers["camera_rgb"][str(i)]
                        )
                elif str(i) in self._windows["camera_rgb"]:

                    self._windows["camera_rgb"].pop(str(i)).destroy()
                    self._img_providers["camera_rgb"].pop(str(i)).destroy()

            if "camera_depth" in self.cfg.data_types:
                show_img = prim.GetAttribute("debug_camera_depth").Get()
                if show_img:
                    if str(i) not in self._windows["camera_depth"]:

                        window = omni.ui.Window(
                            self._prim_view.prim_paths[i] + "/camera_depth",
                            height=self.cfg.sensor_camera_cfg.resolution[1],
                            width=self.cfg.sensor_camera_cfg.resolution[0],
                        )
                        self._windows["camera_depth"][str(i)] = window

                        self._img_providers["camera_depth"][
                            str(i)
                        ] = omni.ui.ByteImageProvider()

                    frame = self._data.output["camera_depth"][i].cpu().numpy()



                    frame = np.dstack((frame, frame, frame)).astype(np.uint8)
                    frame = cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


                    frame = frame.astype(np.uint8)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
                    height, width, channels = frame.shape
                    with self._windows["camera_depth"][str(i)].frame:

                        self._img_providers["camera_depth"][str(i)].set_bytes_data(
                            frame.flatten().data, [width, height]
                        )
                        omni.ui.ImageWithProvider(
                            self._img_providers["camera_depth"][str(i)]
                        )
                elif str(i) in self._windows["camera_depth"]:

                    self._windows["camera_depth"].pop(str(i)).destroy()
                    self._img_providers["camera_depth"].pop(str(i)).destroy()

        if "tactile_rgb" in self.cfg.data_types:
            self.optical_simulator._debug_vis_callback(event)

        if "marker_motion" in self.cfg.data_types:
            self.marker_motion_simulator._debug_vis_callback(event)

    """
    Private Helper methods
    """




































    def _get_camera_depth(self):
        if self.camera is not None:
            depth_output = self.camera.data.output["depth"][
                :, :, :, 0
            ]

            depth_output[torch.isinf(depth_output)] = self.cfg.sensor_camera_cfg.clipping_range[1]

            self._data.output["camera_depth"] = depth_output.reshape(
                (self._num_envs, 1, self.camera_resolution[1], self.camera_resolution[0])
            )
            self._data.output["camera_depth"] *= 1000.0


            normalized = self._data.output["camera_depth"].view(self._data.output["camera_depth"].size(0), -1)
            normalized -= self.cfg.sensor_camera_cfg.clipping_range[0] * 1000
            normalized /= self.cfg.sensor_camera_cfg.clipping_range[1] * 1000
            normalized = (normalized * 255).type(dtype=torch.uint8)
            self._data.output["camera_depth"] = normalized.reshape(
                (self._num_envs, self.camera_resolution[1], self.camera_resolution[0], 1)
            )

        return self._data.output["camera_depth"]

    def _get_height_map(self):
        if self.camera is not None:
            self._data.output["height_map"][:] = self.camera.data.output["depth"][
                :, :, :, 0
            ]

            self._data.output["height_map"][torch.isinf(self._data.output["height_map"])] = (
                self.cfg.sensor_camera_cfg.clipping_range[1]
            )

            self._data.output["height_map"] *= 1000

            return self._data.output["height_map"]
        else:



            pass

    def _show_height_map_inside_gui(self, index):
        plt.close()
        height_map = self._data.output["height_map"][index].cpu().numpy()
        np.save("height_map.npy", height_map)

        X = np.arange(0, height_map.shape[0])
        Y = np.arange(0, height_map.shape[1])
        X, Y = np.meshgrid(X, Y)
        Z = height_map
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(X, Y, Z.T)

        print("saving height_map img")
        plt.savefig(f"height_map{index}.png")

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""

        super()._invalidate_initialize_callback(event)

        self._prim_view = None

        self.camera._invalidate_initialize_callback(event)
        self.camera.__del__()

        if hasattr(self, "_windows"):
            self._windows = None
            self._img_providers = None
