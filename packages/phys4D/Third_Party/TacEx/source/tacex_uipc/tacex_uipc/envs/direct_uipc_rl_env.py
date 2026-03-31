




from __future__ import annotations

import builtins
import gymnasium as gym
import inspect
import math
import numpy as np
import torch
import weakref
from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import MISSING
from typing import Any

import isaacsim.core.utils.torch as torch_utils
import omni.kit.app
import omni.log
from isaacsim.core.simulation_manager import SimulationManager

from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.common import VecEnvObs, VecEnvStepReturn
from isaaclab.envs.ui import ViewportCameraController
from isaaclab.envs.utils.spaces import sample_space, spec_to_gym_space
from isaaclab.managers import EventManager


from isaaclab.sim import SimulationContext
from isaaclab.utils.noise import NoiseModel
from isaaclab.utils.timer import Timer

from tacex_uipc.sim import UipcSim

from .uipc_interactive_scene import UipcInteractiveScene


class UipcRLEnv(DirectRLEnv):
    def __init__(self, cfg: DirectRLEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the environment.

        Args:
            cfg: The configuration object for the environment.
            render_mode: The render mode for the environment. Defaults to None, which
                is similar to ``"human"``.

        Raises:
            RuntimeError: If a simulation context already exists. The environment must always create one
                since it configures the simulation context and controls the simulation.
        """

        cfg.validate()

        self.cfg = cfg

        self.render_mode = render_mode

        self._is_closed = False


        if self.cfg.seed is not None:
            self.cfg.seed = self.seed(self.cfg.seed)
        else:
            omni.log.warn("Seed not set for the environment. The environment creation may not be deterministic.")


        if SimulationContext.instance() is None:
            self.sim: SimulationContext = SimulationContext(self.cfg.sim)
        else:
            raise RuntimeError("Simulation context already exists. Cannot create a new one.")


        if "cuda" in self.device:
            torch.cuda.set_device(self.device)


        if hasattr(self.cfg, "uipc_sim") and self.cfg.uipc_sim is not None:

            self.uipc_sim: UipcSim = UipcSim(self.cfg.uipc_sim)
        else:
            self.uipc_sim = None


        print("[INFO]: Base environment:")
        print(f"\tEnvironment device    : {self.device}")
        print(f"\tEnvironment seed      : {self.cfg.seed}")
        print(f"\tPhysics step-size     : {self.physics_dt}")
        print(f"\tRendering step-size   : {self.physics_dt * self.cfg.sim.render_interval}")
        print(f"\tEnvironment step-size : {self.step_dt}")

        if self.cfg.sim.render_interval < self.cfg.decimation:
            msg = (
                f"The render interval ({self.cfg.sim.render_interval}) is smaller than the decimation "
                f"({self.cfg.decimation}). Multiple render calls will happen for each environment step."
                "If this is not intended, set the render interval to be equal to the decimation."
            )
            omni.log.warn(msg)


        with Timer("[INFO]: Time taken for scene creation", "scene_creation"):
            self.scene: UipcInteractiveScene = UipcInteractiveScene(self.cfg.scene)
            self._setup_scene()
        print("[INFO]: Scene manager: ", self.scene)





        if self.sim.render_mode >= self.sim.RenderMode.PARTIAL_RENDERING:
            self.viewport_camera_controller = ViewportCameraController(self, self.cfg.viewer)
        else:
            self.viewport_camera_controller = None




        if self.cfg.events:
            self.event_manager = EventManager(self.cfg.events, self)


            if "prestartup" in self.event_manager.available_modes:
                self.event_manager.apply(mode="prestartup")




        if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:
            print("[INFO]: Starting the simulation. This may take a few seconds. Please wait...")
            with Timer("[INFO]: Time taken for simulation start", "simulation_start"):
                self.sim.reset()



                self.scene.update(dt=self.physics_dt)

        if self.uipc_sim is not None:
            self.uipc_sim.setup_sim()


        source_code = inspect.getsource(self._set_debug_vis_impl)
        self.has_debug_vis_implementation = "NotImplementedError" not in source_code
        self._debug_vis_handle = None




        if self.sim.has_gui() and self.cfg.ui_window_class_type is not None:
            self._window = self.cfg.ui_window_class_type(self, window_name="IsaacLab")
        else:

            self._window = None


        self.extras = {}



        self._sim_step_counter = 0

        self.common_step_counter = 0

        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.reset_terminated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.reset_time_outs = torch.zeros_like(self.reset_terminated)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.sim.device)


        self._configure_gym_env_spaces()


        if self.cfg.action_noise_model:
            self._action_noise_model: NoiseModel = self.cfg.action_noise_model.class_type(
                self.cfg.action_noise_model, num_envs=self.num_envs, device=self.device
            )
        if self.cfg.observation_noise_model:
            self._observation_noise_model: NoiseModel = self.cfg.observation_noise_model.class_type(
                self.cfg.observation_noise_model, num_envs=self.num_envs, device=self.device
            )


        if self.cfg.events:
            print("[INFO] Event Manager: ", self.event_manager)

            if "startup" in self.event_manager.available_modes:
                self.event_manager.apply(mode="startup")



        self.metadata["render_fps"] = 1 / self.step_dt


        print("[INFO]: Completed setting up the environment...")

    def __del__(self):
        """Cleanup for the environment."""
        self.close()

    """
    Properties.
    """

    @property
    def num_envs(self) -> int:
        """The number of instances of the environment that are running."""
        return self.scene.num_envs

    @property
    def physics_dt(self) -> float:
        """The physics time-step (in s).

        This is the lowest time-decimation at which the simulation is happening.
        """
        return self.cfg.sim.dt

    @property
    def step_dt(self) -> float:
        """The environment stepping time-step (in s).

        This is the time-step at which the environment steps forward.
        """
        return self.cfg.sim.dt * self.cfg.decimation

    @property
    def device(self):
        """The device on which the environment is running."""
        return self.sim.device

    @property
    def max_episode_length_s(self) -> float:
        """Maximum episode length in seconds."""
        return self.cfg.episode_length_s

    @property
    def max_episode_length(self):
        """The maximum episode length in steps adjusted from s."""
        return math.ceil(self.max_episode_length_s / (self.cfg.sim.dt * self.cfg.decimation))

    """
    Operations.
    """

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[VecEnvObs, dict]:
        """Resets all the environments and returns observations.

        This function calls the :meth:`_reset_idx` function to reset all the environments.
        However, certain operations, such as procedural terrain generation, that happened during initialization
        are not repeated.

        Args:
            seed: The seed to use for randomization. Defaults to None, in which case the seed is not set.
            options: Additional information to specify how the environment is reset. Defaults to None.

                Note:
                    This argument is used for compatibility with Gymnasium environment definition.

        Returns:
            A tuple containing the observations and extras.
        """

        if seed is not None:
            self.seed(seed)


        indices = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        self._reset_idx(indices)


        self.scene.write_data_to_sim()
        self.sim.forward()


        if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
            self.sim.render()

        if self.cfg.wait_for_textures and self.sim.has_rtx_sensors():
            while SimulationManager.assets_loading():
                self.sim.render()


        return self._get_observations(), self.extras

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics.

        The environment steps forward at a fixed time-step, while the physics simulation is decimated at a
        lower time-step. This is to ensure that the simulation is stable. These two time-steps can be configured
        independently using the :attr:`DirectRLEnvCfg.decimation` (number of simulation steps per environment step)
        and the :attr:`DirectRLEnvCfg.sim.physics_dt` (physics time-step). Based on these parameters, the environment
        time-step is computed as the product of the two.

        This function performs the following steps:

        1. Pre-process the actions before stepping through the physics.
        2. Apply the actions to the simulator and step through the physics in a decimated manner.
        3. Compute the reward and done signals.
        4. Reset environments that have terminated or reached the maximum episode length.
        5. Apply interval events if they are enabled.
        6. Compute observations.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        action = action.to(self.device)

        if self.cfg.action_noise_model:
            action = self._action_noise_model(action)


        self._pre_physics_step(action)



        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()


        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1

            self._apply_action()

            self.scene.write_data_to_sim()

            self.sim.step(render=False)








            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                if self.uipc_sim is not None:
                    self.uipc_sim.update_render_meshes()
                self.sim.render()


            self.scene.update(dt=self.physics_dt)



        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()


        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)

            self.scene.write_data_to_sim()
            self.sim.forward()

            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                if self.uipc_sim is not None:
                    self.uipc_sim.update_render_meshes()
                self.sim.render()


        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)


        self.obs_buf = self._get_observations()



        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model.apply(self.obs_buf["policy"])


        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    @staticmethod
    def seed(seed: int = -1) -> int:
        """Set the seed for the environment.

        Args:
            seed: The seed for random generator. Defaults to -1.

        Returns:
            The seed used for random generator.
        """

        try:
            import omni.replicator.core as rep

            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            pass

        return torch_utils.set_seed(seed)

    def render(self, recompute: bool = False) -> np.ndarray | None:
        """Run rendering without stepping through the physics.

        By convention, if mode is:

        - **human**: Render to the current display and return nothing. Usually for human consumption.
        - **rgb_array**: Return an numpy.ndarray with shape (x, y, 3), representing RGB values for an
          x-by-y pixel image, suitable for turning into a video.

        Args:
            recompute: Whether to force a render even if the simulator has already rendered the scene.
                Defaults to False.

        Returns:
            The rendered image as a numpy array if mode is "rgb_array". Otherwise, returns None.

        Raises:
            RuntimeError: If mode is set to "rgb_data" and simulation render mode does not support it.
                In this case, the simulation render mode must be set to ``RenderMode.PARTIAL_RENDERING``
                or ``RenderMode.FULL_RENDERING``.
            NotImplementedError: If an unsupported rendering mode is specified.
        """


        if not self.sim.has_rtx_sensors() and not recompute:
            if self.uipc_sim is not None:
                self.uipc_sim.update_render_meshes()
            self.sim.render()

        if self.render_mode == "human" or self.render_mode is None:
            return None
        elif self.render_mode == "rgb_array":

            if self.sim.render_mode.value < self.sim.RenderMode.PARTIAL_RENDERING.value:
                raise RuntimeError(
                    f"Cannot render '{self.render_mode}' when the simulation render mode is"
                    f" '{self.sim.render_mode.name}'. Please set the simulation render mode to:"
                    f"'{self.sim.RenderMode.PARTIAL_RENDERING.name}' or '{self.sim.RenderMode.FULL_RENDERING.name}'."
                    " If running headless, make sure --enable_cameras is set."
                )

            if not hasattr(self, "_rgb_annotator"):
                import omni.replicator.core as rep


                self._render_product = rep.create.render_product(
                    self.cfg.viewer.cam_prim_path, self.cfg.viewer.resolution
                )

                self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
                self._rgb_annotator.attach([self._render_product])

            rgb_data = self._rgb_annotator.get_data()

            rgb_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)


            if rgb_data.size == 0:
                return np.zeros((self.cfg.viewer.resolution[1], self.cfg.viewer.resolution[0], 3), dtype=np.uint8)
            else:
                return rgb_data[:, :, :3]
        else:
            raise NotImplementedError(
                f"Render mode '{self.render_mode}' is not supported. Please use: {self.metadata['render_modes']}."
            )

    def close(self):
        """Cleanup for the environment."""
        if not self._is_closed:


            if self.cfg.events:
                del self.event_manager
            del self.scene
            if self.viewport_camera_controller is not None:
                del self.viewport_camera_controller

            self.sim.clear_all_callbacks()
            self.sim.clear_instance()

            if self._window is not None:
                self._window = None

            self._is_closed = True

    """
    Operations - Debug Visualization.
    """

    def set_debug_vis(self, debug_vis: bool) -> bool:
        """Toggles the environment debug visualization.

        Args:
            debug_vis: Whether to visualize the environment debug visualization.

        Returns:
            Whether the debug visualization was successfully set. False if the environment
            does not support debug visualization.
        """

        if not self.has_debug_vis_implementation:
            return False

        self._set_debug_vis_impl(debug_vis)

        if debug_vis:

            if self._debug_vis_handle is None:
                app_interface = omni.kit.app.get_app_interface()
                self._debug_vis_handle = app_interface.get_post_update_event_stream().create_subscription_to_pop(
                    lambda event, obj=weakref.proxy(self): obj._debug_vis_callback(event)
                )
        else:

            if self._debug_vis_handle is not None:
                self._debug_vis_handle.unsubscribe()
                self._debug_vis_handle = None

        return True

    """
    Helper functions.
    """

    def _configure_gym_env_spaces(self):
        """Configure the action and observation spaces for the Gym environment."""

        if self.cfg.num_actions is not None:
            omni.log.warn("DirectRLEnvCfg.num_actions is deprecated. Use DirectRLEnvCfg.action_space instead.")
            if isinstance(self.cfg.action_space, type(MISSING)):
                self.cfg.action_space = self.cfg.num_actions
        if self.cfg.num_observations is not None:
            omni.log.warn(
                "DirectRLEnvCfg.num_observations is deprecated. Use DirectRLEnvCfg.observation_space instead."
            )
            if isinstance(self.cfg.observation_space, type(MISSING)):
                self.cfg.observation_space = self.cfg.num_observations
        if self.cfg.num_states is not None:
            omni.log.warn("DirectRLEnvCfg.num_states is deprecated. Use DirectRLEnvCfg.state_space instead.")
            if isinstance(self.cfg.state_space, type(MISSING)):
                self.cfg.state_space = self.cfg.num_states


        self.single_observation_space = gym.spaces.Dict()
        self.single_observation_space["policy"] = spec_to_gym_space(self.cfg.observation_space)
        self.single_action_space = spec_to_gym_space(self.cfg.action_space)


        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space["policy"], self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)


        self.state_space = None
        if self.cfg.state_space:
            self.single_observation_space["critic"] = spec_to_gym_space(self.cfg.state_space)
            self.state_space = gym.vector.utils.batch_space(self.single_observation_space["critic"], self.num_envs)


        self.actions = sample_space(self.single_action_space, self.sim.device, batch_size=self.num_envs, fill_value=0)

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset environments based on specified indices.

        Args:
            env_ids: List of environment ids which must be reset
        """
        self.scene.reset(env_ids)


        if self.cfg.events:
            if "reset" in self.event_manager.available_modes:
                env_step_count = self._sim_step_counter // self.cfg.decimation
                self.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=env_step_count)


        if self.cfg.action_noise_model:
            self._action_noise_model.reset(env_ids)
        if self.cfg.observation_noise_model:
            self._observation_noise_model.reset(env_ids)


        self.episode_length_buf[env_ids] = 0

    """
    Implementation-specific functions.
    """

    def _setup_scene(self):
        """Setup the scene for the environment.

        This function is responsible for creating the scene objects and setting up the scene for the environment.
        The scene creation can happen through :class:`isaaclab.scene.InteractiveSceneCfg` or through
        directly creating the scene objects and registering them with the scene manager.

        We leave the implementation of this function to the derived classes. If the environment does not require
        any explicit scene setup, the function can be left empty.
        """
        pass

    @abstractmethod
    def _pre_physics_step(self, actions: torch.Tensor):
        """Pre-process actions before stepping through the physics.

        This function is responsible for pre-processing the actions before stepping through the physics.
        It is called before the physics stepping (which is decimated).

        Args:
            actions: The actions to apply on the environment. Shape is (num_envs, action_dim).
        """
        raise NotImplementedError(f"Please implement the '_pre_physics_step' method for {self.__class__.__name__}.")

    @abstractmethod
    def _apply_action(self):
        """Apply actions to the simulator.

        This function is responsible for applying the actions to the simulator. It is called at each
        physics time-step.
        """
        raise NotImplementedError(f"Please implement the '_apply_action' method for {self.__class__.__name__}.")

    @abstractmethod
    def _get_observations(self) -> VecEnvObs:
        """Compute and return the observations for the environment.

        Returns:
            The observations for the environment.
        """
        raise NotImplementedError(f"Please implement the '_get_observations' method for {self.__class__.__name__}.")

    def _get_states(self) -> VecEnvObs | None:
        """Compute and return the states for the environment.

        The state-space is used for asymmetric actor-critic architectures. It is configured
        using the :attr:`DirectRLEnvCfg.state_space` parameter.

        Returns:
            The states for the environment. If the environment does not have a state-space, the function
            returns a None.
        """
        return None

    @abstractmethod
    def _get_rewards(self) -> torch.Tensor:
        """Compute and return the rewards for the environment.

        Returns:
            The rewards for the environment. Shape is (num_envs,).
        """
        raise NotImplementedError(f"Please implement the '_get_rewards' method for {self.__class__.__name__}.")

    @abstractmethod
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute and return the done flags for the environment.

        Returns:
            A tuple containing the done flags for termination and time-out.
            Shape of individual tensors is (num_envs,).
        """
        raise NotImplementedError(f"Please implement the '_get_dones' method for {self.__class__.__name__}.")

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization into visualization objects.

        This function is responsible for creating the visualization objects if they don't exist
        and input ``debug_vis`` is True. If the visualization objects exist, the function should
        set their visibility into the stage.
        """
        raise NotImplementedError(f"Debug visualization is not implemented for {self.__class__.__name__}.")
