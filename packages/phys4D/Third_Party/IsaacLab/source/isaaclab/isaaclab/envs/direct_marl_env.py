




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
from typing import Any, ClassVar

import isaacsim.core.utils.torch as torch_utils
import omni.kit.app
import omni.log
import omni.physx
from isaacsim.core.version import get_version

from isaaclab.managers import EventManager
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext
from isaaclab.sim.utils import attach_stage_to_usd_context, use_stage
from isaaclab.utils.noise import NoiseModel
from isaaclab.utils.timer import Timer

from .common import ActionType, AgentID, EnvStepReturn, ObsType, StateType
from .direct_marl_env_cfg import DirectMARLEnvCfg
from .ui import ViewportCameraController
from .utils.spaces import sample_space, spec_to_gym_space


class DirectMARLEnv(gym.Env):
    """The superclass for the direct workflow to design multi-agent environments.

    This class implements the core functionality for multi-agent reinforcement learning (MARL)
    environments. It is designed to be used with any RL library. The class is designed
    to be used with vectorized environments, i.e., the environment is expected to be run
    in parallel with multiple sub-environments.

    The design of this class is based on the PettingZoo Parallel API.
    While the environment itself is implemented as a vectorized environment, we do not
    inherit from :class:`pettingzoo.ParallelEnv` or :class:`gym.vector.VectorEnv`. This is mainly
    because the class adds various attributes and methods that are inconsistent with them.

    Note:
        For vectorized environments, it is recommended to **only** call the :meth:`reset`
        method once before the first call to :meth:`step`, i.e. after the environment is created.
        After that, the :meth:`step` function handles the reset of terminated sub-environments.
        This is because the simulator does not support resetting individual sub-environments
        in a vectorized environment.

    """

    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": [None, "human", "rgb_array"],
        "isaac_sim_version": get_version(),
    }
    """Metadata for the environment."""

    def __init__(self, cfg: DirectMARLEnvCfg, render_mode: str | None = None, **kwargs):
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

            with use_stage(self.sim.get_initial_stage()):
                self.scene = InteractiveScene(self.cfg.scene)
                self._setup_scene()
                attach_stage_to_usd_context()
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


                with use_stage(self.sim.get_initial_stage()):
                    self.sim.reset()



                self.scene.update(dt=self.physics_dt)


        source_code = inspect.getsource(self._set_debug_vis_impl)
        self.has_debug_vis_implementation = "NotImplementedError" not in source_code
        self._debug_vis_handle = None




        if self.sim.has_gui() and self.cfg.ui_window_class_type is not None:
            self._window = self.cfg.ui_window_class_type(self, window_name="IsaacLab")
        else:

            self._window = None


        self.extras = {agent: {} for agent in self.cfg.possible_agents}



        self._sim_step_counter = 0

        self.common_step_counter = 0

        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.sim.device)


        self._configure_env_spaces()


        if self.cfg.action_noise_model:
            self._action_noise_model: dict[AgentID, NoiseModel] = {
                agent: noise_model.class_type(noise_model, num_envs=self.num_envs, device=self.device)
                for agent, noise_model in self.cfg.action_noise_model.items()
                if noise_model is not None
            }
        if self.cfg.observation_noise_model:
            self._observation_noise_model: dict[AgentID, NoiseModel] = {
                agent: noise_model.class_type(noise_model, num_envs=self.num_envs, device=self.device)
                for agent, noise_model in self.cfg.observation_noise_model.items()
                if noise_model is not None
            }


        if self.cfg.events:

            print("[INFO] Event Manager: ", self.event_manager)

            if "startup" in self.event_manager.available_modes:
                self.event_manager.apply(mode="startup")


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
    def num_agents(self) -> int:
        """Number of current agents.

        The number of current agents may change as the environment progresses (e.g.: agents can be added or removed).
        """
        return len(self.agents)

    @property
    def max_num_agents(self) -> int:
        """Number of all possible agents the environment can generate.

        This value remains constant as the environment progresses.
        """
        return len(self.possible_agents)

    @property
    def unwrapped(self) -> DirectMARLEnv:
        """Get the unwrapped environment underneath all the layers of wrappers."""
        return self

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
    Space methods
    """

    def observation_space(self, agent: AgentID) -> gym.Space:
        """Get the observation space for the specified agent.

        Returns:
            The agent's observation space.
        """
        return self.observation_spaces[agent]

    def action_space(self, agent: AgentID) -> gym.Space:
        """Get the action space for the specified agent.

        Returns:
            The agent's action space.
        """
        return self.action_spaces[agent]

    """
    Operations.
    """

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[AgentID, ObsType], dict[AgentID, dict]]:
        """Resets all the environments and returns observations.

        Args:
            seed: The seed to use for randomization. Defaults to None, in which case the seed is not set.
            options: Additional information to specify how the environment is reset. Defaults to None.

                Note:
                    This argument is used for compatibility with Gymnasium environment definition.

        Returns:
            A tuple containing the observations and extras (keyed by the agent ID).
        """

        if seed is not None:
            self.seed(seed)


        indices = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        self._reset_idx(indices)


        self.obs_dict = self._get_observations()
        self.agents = [agent for agent in self.possible_agents if agent in self.obs_dict]


        return self.obs_dict, self.extras

    def step(self, actions: dict[AgentID, ActionType]) -> EnvStepReturn:
        """Execute one time-step of the environment's dynamics.

        The environment steps forward at a fixed time-step, while the physics simulation is decimated at a
        lower time-step. This is to ensure that the simulation is stable. These two time-steps can be configured
        independently using the :attr:`DirectMARLEnvCfg.decimation` (number of simulation steps per environment step)
        and the :attr:`DirectMARLEnvCfg.sim.physics_dt` (physics time-step). Based on these parameters, the environment
        time-step is computed as the product of the two.

        This function performs the following steps:

        1. Pre-process the actions before stepping through the physics.
        2. Apply the actions to the simulator and step through the physics in a decimated manner.
        3. Compute the reward and done signals.
        4. Reset environments that have terminated or reached the maximum episode length.
        5. Apply interval events if they are enabled.
        6. Compute observations.

        Args:
            actions: The actions to apply on the environment (keyed by the agent ID).
                Shape of individual tensors is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras (keyed by the agent ID).
        """
        actions = {agent: action.to(self.device) for agent, action in actions.items()}


        if self.cfg.action_noise_model:
            for agent, action in actions.items():
                if agent in self._action_noise_model:
                    actions[agent] = self._action_noise_model[agent](action)

        self._pre_physics_step(actions)



        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()


        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1

            self._apply_action()

            self.scene.write_data_to_sim()

            self.sim.step(render=False)



            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()

            self.scene.update(dt=self.physics_dt)



        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.terminated_dict, self.time_out_dict = self._get_dones()
        self.reset_buf[:] = math.prod(self.terminated_dict.values()) | math.prod(self.time_out_dict.values())
        self.reward_dict = self._get_rewards()


        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)


        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)


        self.obs_dict = self._get_observations()
        self.agents = [agent for agent in self.possible_agents if agent in self.obs_dict]



        if self.cfg.observation_noise_model:
            for agent, obs in self.obs_dict.items():
                if agent in self._observation_noise_model:
                    self.obs_dict[agent] = self._observation_noise_model[agent](obs)


        return self.obs_dict, self.reward_dict, self.terminated_dict, self.time_out_dict, self.extras

    def state(self) -> StateType | None:
        """Returns the state for the environment.

        The state-space is used for centralized training or asymmetric actor-critic architectures. It is configured
        using the :attr:`DirectMARLEnvCfg.state_space` parameter.

        Returns:
            The states for the environment, or None if :attr:`DirectMARLEnvCfg.state_space` parameter is zero.
        """
        if not self.cfg.state_space:
            return None


        if isinstance(self.cfg.state_space, int) and self.cfg.state_space < 0:
            self.state_buf = torch.cat(
                [self.obs_dict[agent].reshape(self.num_envs, -1) for agent in self.cfg.possible_agents], dim=-1
            )

        else:
            self.state_buf = self._get_states()
        return self.state_buf

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
        - **rgb_array**: Return a numpy.ndarray with shape (x, y, 3), representing RGB values for an
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


            if float(".".join(get_version()[2])) >= 5:
                if self.cfg.sim.create_stage_in_memory:

                    omni.physx.get_physx_simulation_interface().detach_stage()
                    self.sim.stop()
                    self.sim.clear()

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

    def _configure_env_spaces(self):
        """Configure the spaces for the environment."""
        self.agents = self.cfg.possible_agents
        self.possible_agents = self.cfg.possible_agents


        if self.cfg.num_actions is not None:
            omni.log.warn("DirectMARLEnvCfg.num_actions is deprecated. Use DirectMARLEnvCfg.action_spaces instead.")
            if isinstance(self.cfg.action_spaces, type(MISSING)):
                self.cfg.action_spaces = self.cfg.num_actions
        if self.cfg.num_observations is not None:
            omni.log.warn(
                "DirectMARLEnvCfg.num_observations is deprecated. Use DirectMARLEnvCfg.observation_spaces instead."
            )
            if isinstance(self.cfg.observation_spaces, type(MISSING)):
                self.cfg.observation_spaces = self.cfg.num_observations
        if self.cfg.num_states is not None:
            omni.log.warn("DirectMARLEnvCfg.num_states is deprecated. Use DirectMARLEnvCfg.state_space instead.")
            if isinstance(self.cfg.state_space, type(MISSING)):
                self.cfg.state_space = self.cfg.num_states


        self.observation_spaces = {
            agent: spec_to_gym_space(self.cfg.observation_spaces[agent]) for agent in self.cfg.possible_agents
        }
        self.action_spaces = {
            agent: spec_to_gym_space(self.cfg.action_spaces[agent]) for agent in self.cfg.possible_agents
        }


        if not self.cfg.state_space:
            self.state_space = None
        if isinstance(self.cfg.state_space, int) and self.cfg.state_space < 0:
            self.state_space = gym.spaces.flatten_space(
                gym.spaces.Tuple([self.observation_spaces[agent] for agent in self.cfg.possible_agents])
            )
        else:
            self.state_space = spec_to_gym_space(self.cfg.state_space)


        self.actions = {
            agent: sample_space(self.action_spaces[agent], self.sim.device, batch_size=self.num_envs, fill_value=0)
            for agent in self.cfg.possible_agents
        }

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
            for noise_model in self._action_noise_model.values():
                noise_model.reset(env_ids)
        if self.cfg.observation_noise_model:
            for noise_model in self._observation_noise_model.values():
                noise_model.reset(env_ids)


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
    def _pre_physics_step(self, actions: dict[AgentID, ActionType]):
        """Pre-process actions before stepping through the physics.

        This function is responsible for pre-processing the actions before stepping through the physics.
        It is called before the physics stepping (which is decimated).

        Args:
            actions: The actions to apply on the environment (keyed by the agent ID).
                Shape of individual tensors is (num_envs, action_dim).
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
    def _get_observations(self) -> dict[AgentID, ObsType]:
        """Compute and return the observations for the environment.

        Returns:
            The observations for the environment (keyed by the agent ID).
        """
        raise NotImplementedError(f"Please implement the '_get_observations' method for {self.__class__.__name__}.")

    @abstractmethod
    def _get_states(self) -> StateType:
        """Compute and return the states for the environment.

        This method is only called (and therefore has to be implemented) when the :attr:`DirectMARLEnvCfg.state_space`
        parameter is not a number less than or equal to zero.

        Returns:
            The states for the environment.
        """
        raise NotImplementedError(f"Please implement the '_get_states' method for {self.__class__.__name__}.")

    @abstractmethod
    def _get_rewards(self) -> dict[AgentID, torch.Tensor]:
        """Compute and return the rewards for the environment.

        Returns:
            The rewards for the environment (keyed by the agent ID).
            Shape of individual tensors is (num_envs,).
        """
        raise NotImplementedError(f"Please implement the '_get_rewards' method for {self.__class__.__name__}.")

    @abstractmethod
    def _get_dones(self) -> tuple[dict[AgentID, torch.Tensor], dict[AgentID, torch.Tensor]]:
        """Compute and return the done flags for the environment.

        Returns:
            A tuple containing the done flags for termination and time-out (keyed by the agent ID).
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
