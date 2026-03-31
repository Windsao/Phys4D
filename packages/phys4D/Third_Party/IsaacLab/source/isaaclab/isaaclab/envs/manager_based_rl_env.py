





from __future__ import annotations

import gymnasium as gym
import math
import numpy as np
import torch
from collections.abc import Sequence
from typing import Any, ClassVar

from isaacsim.core.version import get_version

from isaaclab.managers import CommandManager, CurriculumManager, RewardManager, TerminationManager
from isaaclab.ui.widgets import ManagerLiveVisualizer

from .common import VecEnvStepReturn
from .manager_based_env import ManagerBasedEnv
from .manager_based_rl_env_cfg import ManagerBasedRLEnvCfg


class ManagerBasedRLEnv(ManagerBasedEnv, gym.Env):
    """The superclass for the manager-based workflow reinforcement learning-based environments.

    This class inherits from :class:`ManagerBasedEnv` and implements the core functionality for
    reinforcement learning-based environments. It is designed to be used with any RL
    library. The class is designed to be used with vectorized environments, i.e., the
    environment is expected to be run in parallel with multiple sub-environments. The
    number of sub-environments is specified using the ``num_envs``.

    Each observation from the environment is a batch of observations for each sub-
    environments. The method :meth:`step` is also expected to receive a batch of actions
    for each sub-environment.

    While the environment itself is implemented as a vectorized environment, we do not
    inherit from :class:`gym.vector.VectorEnv`. This is mainly because the class adds
    various methods (for wait and asynchronous updates) which are not required.
    Additionally, each RL library typically has its own definition for a vectorized
    environment. Thus, to reduce complexity, we directly use the :class:`gym.Env` over
    here and leave it up to library-defined wrappers to take care of wrapping this
    environment for their agents.

    Note:
        For vectorized environments, it is recommended to **only** call the :meth:`reset`
        method once before the first call to :meth:`step`, i.e. after the environment is created.
        After that, the :meth:`step` function handles the reset of terminated sub-environments.
        This is because the simulator does not support resetting individual sub-environments
        in a vectorized environment.

    """

    is_vector_env: ClassVar[bool] = True
    """Whether the environment is a vectorized environment."""
    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": [None, "human", "rgb_array"],
        "isaac_sim_version": get_version(),
    }
    """Metadata for the environment."""

    cfg: ManagerBasedRLEnvCfg
    """Configuration for the environment."""

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the environment.

        Args:
            cfg: The configuration for the environment.
            render_mode: The render mode for the environment. Defaults to None, which
                is similar to ``"human"``.
        """

        self.common_step_counter = 0


        self.episode_length_buf = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device, dtype=torch.long)


        super().__init__(cfg=cfg)

        self.render_mode = render_mode



        self.metadata["render_fps"] = 1 / self.step_dt

        print("[INFO]: Completed setting up the environment...")

    """
    Properties.
    """

    @property
    def max_episode_length_s(self) -> float:
        """Maximum episode length in seconds."""
        return self.cfg.episode_length_s

    @property
    def max_episode_length(self) -> int:
        """Maximum episode length in environment steps."""
        return math.ceil(self.max_episode_length_s / self.step_dt)

    """
    Operations - Setup.
    """

    def load_managers(self):



        self.command_manager: CommandManager = CommandManager(self.cfg.commands, self)
        print("[INFO] Command Manager: ", self.command_manager)


        super().load_managers()



        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        print("[INFO] Termination Manager: ", self.termination_manager)

        self.reward_manager = RewardManager(self.cfg.rewards, self)
        print("[INFO] Reward Manager: ", self.reward_manager)

        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        print("[INFO] Curriculum Manager: ", self.curriculum_manager)


        self._configure_gym_env_spaces()


        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")

    def setup_manager_visualizers(self):
        """Creates live visualizers for manager terms."""

        self.manager_visualizers = {
            "action_manager": ManagerLiveVisualizer(manager=self.action_manager),
            "observation_manager": ManagerLiveVisualizer(manager=self.observation_manager),
            "command_manager": ManagerLiveVisualizer(manager=self.command_manager),
            "termination_manager": ManagerLiveVisualizer(manager=self.termination_manager),
            "reward_manager": ManagerLiveVisualizer(manager=self.reward_manager),
            "curriculum_manager": ManagerLiveVisualizer(manager=self.curriculum_manager),
        }

    """
    Operations - MDP
    """

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics and reset terminated environments.

        Unlike the :class:`ManagerBasedEnv.step` class, the function performs the following operations:

        1. Process the actions.
        2. Perform physics stepping.
        3. Perform rendering if gui is enabled.
        4. Update the environment counters and compute the rewards and terminations.
        5. Reset the environments that terminated.
        6. Compute the observations.
        7. Return the observations, rewards, resets and extras.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """

        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()



        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()


        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1

            self.action_manager.apply_action()

            self.scene.write_data_to_sim()

            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()



            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()

            self.scene.update(dt=self.physics_dt)



        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs

        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:

            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()


        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:

            self.recorder_manager.record_pre_reset(reset_env_ids)

            self._reset_idx(reset_env_ids)


            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()


            self.recorder_manager.record_post_reset(reset_env_ids)


        self.command_manager.compute(dt=self.step_dt)

        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)


        self.obs_buf = self.observation_manager.compute(update_history=True)


        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

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
        if not self._is_closed:

            del self.command_manager
            del self.reward_manager
            del self.termination_manager
            del self.curriculum_manager

            super().close()

    """
    Helper functions.
    """

    def _configure_gym_env_spaces(self):
        """Configure the action and observation spaces for the Gym environment."""

        self.single_observation_space = gym.spaces.Dict()
        for group_name, group_term_names in self.observation_manager.active_terms.items():

            has_concatenated_obs = self.observation_manager.group_obs_concatenate[group_name]
            group_dim = self.observation_manager.group_obs_dim[group_name]


            if has_concatenated_obs:
                self.single_observation_space[group_name] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=group_dim)
            else:
                group_term_cfgs = self.observation_manager._group_obs_term_cfgs[group_name]
                term_dict = {}
                for term_name, term_dim, term_cfg in zip(group_term_names, group_dim, group_term_cfgs):
                    low = -np.inf if term_cfg.clip is None else term_cfg.clip[0]
                    high = np.inf if term_cfg.clip is None else term_cfg.clip[1]
                    term_dict[term_name] = gym.spaces.Box(low=low, high=high, shape=term_dim)
                self.single_observation_space[group_name] = gym.spaces.Dict(term_dict)

        action_dim = sum(self.action_manager.action_term_dim)
        self.single_action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(action_dim,))


        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset environments based on specified indices.

        Args:
            env_ids: List of environment ids which must be reset
        """

        self.curriculum_manager.compute(env_ids=env_ids)

        self.scene.reset(env_ids)

        if "reset" in self.event_manager.available_modes:
            env_step_count = self._sim_step_counter // self.cfg.decimation
            self.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=env_step_count)




        self.extras["log"] = dict()

        info = self.observation_manager.reset(env_ids)
        self.extras["log"].update(info)

        info = self.action_manager.reset(env_ids)
        self.extras["log"].update(info)

        info = self.reward_manager.reset(env_ids)
        self.extras["log"].update(info)

        info = self.curriculum_manager.reset(env_ids)
        self.extras["log"].update(info)

        info = self.command_manager.reset(env_ids)
        self.extras["log"].update(info)

        info = self.event_manager.reset(env_ids)
        self.extras["log"].update(info)

        info = self.termination_manager.reset(env_ids)
        self.extras["log"].update(info)

        info = self.recorder_manager.reset(env_ids)
        self.extras["log"].update(info)


        self.episode_length_buf[env_ids] = 0
