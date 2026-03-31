




from __future__ import annotations

import copy
import numpy as np
import torch
import weakref
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.kit.app
import omni.timeline

from isaaclab.assets.articulation.articulation import Articulation

if TYPE_CHECKING:
    from isaaclab.envs import DirectRLEnv, ManagerBasedEnv, ViewerCfg


class ViewportCameraController:
    """This class handles controlling the camera associated with a viewport in the simulator.

    It can be used to set the viewpoint camera to track different origin types:

    - **world**: the center of the world (static)
    - **env**: the center of an environment (static)
    - **asset_root**: the root of an asset in the scene (e.g. tracking a robot moving in the scene)

    On creation, the camera is set to track the origin type specified in the configuration.

    For the :attr:`asset_root` origin type, the camera is updated at each rendering step to track the asset's
    root position. For this, it registers a callback to the post update event stream from the simulation app.
    """

    def __init__(self, env: ManagerBasedEnv | DirectRLEnv, cfg: ViewerCfg):
        """Initialize the ViewportCameraController.

        Args:
            env: The environment.
            cfg: The configuration for the viewport camera controller.

        Raises:
            ValueError: If origin type is configured to be "env" but :attr:`cfg.env_index` is out of bounds.
            ValueError: If origin type is configured to be "asset_root" but :attr:`cfg.asset_name` is unset.

        """

        self._env = env
        self._cfg = copy.deepcopy(cfg)

        self.default_cam_eye = np.array(self._cfg.eye, dtype=float)
        self.default_cam_lookat = np.array(self._cfg.lookat, dtype=float)


        if self.cfg.origin_type == "env":

            self.set_view_env_index(self.cfg.env_index)

            self.update_view_to_env()
        elif self.cfg.origin_type == "asset_root" or self.cfg.origin_type == "asset_body":



            if self.cfg.asset_name is None:
                raise ValueError(f"No asset name provided for viewer with origin type: '{self.cfg.origin_type}'.")
            if self.cfg.origin_type == "asset_body":
                if self.cfg.body_name is None:
                    raise ValueError(f"No body name provided for viewer with origin type: '{self.cfg.origin_type}'.")
        else:

            self.update_view_to_world()


        app_interface = omni.kit.app.get_app_interface()
        app_event_stream = app_interface.get_post_update_event_stream()
        self._viewport_camera_update_handle = app_event_stream.create_subscription_to_pop(
            lambda event, obj=weakref.proxy(self): obj._update_tracking_callback(event)
        )

    def __del__(self):
        """Unsubscribe from the callback."""

        if hasattr(self, "_viewport_camera_update_handle") and self._viewport_camera_update_handle is not None:
            self._viewport_camera_update_handle.unsubscribe()
            self._viewport_camera_update_handle = None

    """
    Properties
    """

    @property
    def cfg(self) -> ViewerCfg:
        """The configuration for the viewer."""
        return self._cfg

    """
    Public Functions
    """

    def set_view_env_index(self, env_index: int):
        """Sets the environment index for the camera view.

        Args:
            env_index: The index of the environment to set the camera view to.

        Raises:
            ValueError: If the environment index is out of bounds. It should be between 0 and num_envs - 1.
        """

        if env_index < 0 or env_index >= self._env.num_envs:
            raise ValueError(
                f"Out of range value for attribute 'env_index': {env_index}."
                f" Expected a value between 0 and {self._env.num_envs - 1} for the current environment."
            )

        self.cfg.env_index = env_index


        if self.cfg.origin_type == "env":
            self.update_view_to_env()

    def update_view_to_world(self):
        """Updates the viewer's origin to the origin of the world which is (0, 0, 0)."""

        self.cfg.origin_type = "world"

        self.viewer_origin = torch.zeros(3)

        self.update_view_location()

    def update_view_to_env(self):
        """Updates the viewer's origin to the origin of the selected environment."""

        self.cfg.origin_type = "env"

        self.viewer_origin = self._env.scene.env_origins[self.cfg.env_index]

        self.update_view_location()

    def update_view_to_asset_root(self, asset_name: str):
        """Updates the viewer's origin based upon the root of an asset in the scene.

        Args:
            asset_name: The name of the asset in the scene. The name should match the name of the
                asset in the scene.

        Raises:
            ValueError: If the asset is not in the scene.
        """

        if self.cfg.asset_name != asset_name:
            asset_entities = [*self._env.scene.rigid_objects.keys(), *self._env.scene.articulations.keys()]
            if asset_name not in asset_entities:
                raise ValueError(f"Asset '{asset_name}' is not in the scene. Available entities: {asset_entities}.")

        self.cfg.asset_name = asset_name

        self.cfg.origin_type = "asset_root"

        self.viewer_origin = self._env.scene[self.cfg.asset_name].data.root_pos_w[self.cfg.env_index]

        self.update_view_location()

    def update_view_to_asset_body(self, asset_name: str, body_name: str):
        """Updates the viewer's origin based upon the body of an asset in the scene.

        Args:
            asset_name: The name of the asset in the scene. The name should match the name of the
                asset in the scene.
            body_name: The name of the body in the asset.

        Raises:
            ValueError: If the asset is not in the scene or the body is not valid.
        """

        if self.cfg.asset_name != asset_name:
            asset_entities = [*self._env.scene.rigid_objects.keys(), *self._env.scene.articulations.keys()]
            if asset_name not in asset_entities:
                raise ValueError(f"Asset '{asset_name}' is not in the scene. Available entities: {asset_entities}.")

        asset: Articulation = self._env.scene[asset_name]
        if body_name not in asset.body_names:
            raise ValueError(
                f"'{body_name}' is not a body of Asset '{asset_name}'. Available bodies: {asset.body_names}."
            )

        body_id, _ = asset.find_bodies(body_name)

        self.cfg.asset_name = asset_name

        self.cfg.origin_type = "asset_body"

        self.viewer_origin = self._env.scene[self.cfg.asset_name].data.body_pos_w[self.cfg.env_index, body_id].view(3)

        self.update_view_location()

    def update_view_location(self, eye: Sequence[float] | None = None, lookat: Sequence[float] | None = None):
        """Updates the camera view pose based on the current viewer origin and the eye and lookat positions.

        Args:
            eye: The eye position of the camera. If None, the current eye position is used.
            lookat: The lookat position of the camera. If None, the current lookat position is used.
        """

        if eye is not None:
            self.default_cam_eye = np.asarray(eye, dtype=float)
        if lookat is not None:
            self.default_cam_lookat = np.asarray(lookat, dtype=float)

        viewer_origin = self.viewer_origin.detach().cpu().numpy()
        cam_eye = viewer_origin + self.default_cam_eye
        cam_target = viewer_origin + self.default_cam_lookat


        self._env.sim.set_camera_view(eye=cam_eye, target=cam_target)

    """
    Private Functions
    """

    def _update_tracking_callback(self, event):
        """Updates the camera view at each rendering step."""


        if self.cfg.origin_type == "asset_root" and self.cfg.asset_name is not None:
            self.update_view_to_asset_root(self.cfg.asset_name)
        if self.cfg.origin_type == "asset_body" and self.cfg.asset_name is not None and self.cfg.body_name is not None:
            self.update_view_to_asset_body(self.cfg.asset_name, self.cfg.body_name)
