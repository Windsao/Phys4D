"""
Please use this environment if you need to get observation or save data in a synchronized manner.
This is a base environment class for MagicSim that serves as a foundation for sync systhetic data generation and sync low-level rl training
We handle data capture and observation in this environment.
! Warning !: This is a synchronized env, meaning that each subenv will step and reset simultaneously.
"""

from typing import Any, Dict, Sequence
from magicsim.Env.Environment.SyncBaseEnv import SyncBaseEnv
import torch
from magicsim.Env.Sensor.CameraManager import CameraManager
from magicsim.Env.Environment.Isaac.IsaacRLEnv import IsaacRLEnv
import omni.usd
from pxr import UsdGeom


class SyncCollectEnv(SyncBaseEnv):
    """
    This environment is used for collecting data in a synchronized manner.
    It inherits from SyncBaseEnv and provides the basic structure for the environment.
    In this class we will implement base data collection and observation get logic
    We implement all collection related logic here including area of VLM 3DV who may use flying camera here.
    """

    def __init__(self, config, cli_args, logger):
        super().__init__(config, cli_args, logger)
        self.capture_config = config.camera
        self.camera_config = config.camera
        self.camera_manager = CameraManager(
            self.num_envs,
            self.camera_config,
            self.device,
            seeds_per_env=self.env_seed_list,
        )
        if self.camera_config.get("enable_tiled", False):
            from magicsim.Env.Capture.TiledCaptureManager import TiledCaptureManager

            self.capture_manager = TiledCaptureManager(
                self.num_envs,
                self.capture_config,
                self.camera_manager,
                self.device,
            )
        else:
            from magicsim.Env.Capture.CaptureManager import CaptureManager

            self.capture_manager = CaptureManager(
                self.num_envs,
                self.capture_config,
                self.camera_manager,
                self.device,
            )

    def _setup_scene(self, sim: IsaacRLEnv):
        """
        Initialize the environment.
        This function will be called before simulation context create
        !!! Please put everything that can not be dynamicly imported here!!!
        """
        self.camera_manager.initialize(sim)
        self.capture_manager.initialize(sim)
        super()._setup_scene(sim)
        self._attach_nav_rooms_to_envs()

    def _post_setup_scene(self, sim: IsaacRLEnv):
        """
        This function will be called after official cloner work but before simulation start.
        Initialize cameras and annotators here.
        """
        super()._post_setup_scene(sim)

        self.camera_manager.post_init()

    def step(self, actions: Dict = None):
        """
        1/ step camera manager to update camera pose (if needed).
        2/ Step the simulation backend
        3/ capture data.
        """
        self.camera_manager.step(actions)
        super().step()

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        """
        Reset the environment and capture manager.
        """
        super().reset(seed=seed, options=options)
        self._update_seed_managers()

        self.camera_manager.reset()
        self.capture_manager.reset()

    def reset_idx(
        self,
        env_ids: Sequence[int] = None,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        """
        Reset specific environments.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.sim.device)
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.sim.device, dtype=torch.int32)

        env_id_list = env_ids.detach().cpu().tolist()

        super().reset_idx(env_ids=env_id_list, seed=seed, options=options)

        self._update_seed_managers()

        self.capture_manager.reset_idx(env_ids=env_id_list)
        self.camera_manager.reset_idx(env_ids=env_id_list)
        self._attach_nav_rooms_to_envs()

    def close(self):
        """
        Close the environment and release resources.
        """

        super().close()

    def _update_seed_managers(self):
        """
        Synchronize per-environment seeds with managers that rely on randomness.
        """
        seeds = self.env_seeds
        if hasattr(self, "camera_manager") and self.camera_manager is not None:
            self.camera_manager.update_env_seeds(seeds)

    def _attach_nav_rooms_to_envs(self):
        """Create per-env references to NavManager rooms so they become visible under env hierarchy."""
        if getattr(self, "_nav_rooms_attached", False):
            return
        if not hasattr(self, "nav_manager") or self.nav_manager is None:
            return

        stage = omni.usd.get_context().get_stage()
        nav_root = "/World/NavRoom"
        env_root = "/World/envs"

        for env_id in range(self.num_envs):
            src_path = f"{nav_root}/Room_{env_id}"
            dst_env_root = f"{env_root}/env_{env_id}"
            dst_path = f"{dst_env_root}/NavRoom"

            src_prim = stage.GetPrimAtPath(src_path)
            env_prim = stage.GetPrimAtPath(dst_env_root)

            if (
                not src_prim
                or not src_prim.IsValid()
                or not env_prim
                or not env_prim.IsValid()
            ):
                continue

            if stage.GetPrimAtPath(dst_path):
                continue

            dst_prim = stage.DefinePrim(dst_path, "Xform")
            UsdGeom.Xformable(dst_prim)
            dst_prim.GetReferences().AddInternalReference(src_path)

        self._nav_rooms_attached = True
