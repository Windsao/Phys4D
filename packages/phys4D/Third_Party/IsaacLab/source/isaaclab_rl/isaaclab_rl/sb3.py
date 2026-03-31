




"""Wrapper to configure an environment instance to Stable-Baselines3 vectorized environment.

The following example shows how to wrap an environment for Stable-Baselines3:

.. code-block:: python

    from isaaclab_rl.sb3 import Sb3VecEnvWrapper

    env = Sb3VecEnvWrapper(env)

"""


from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import warnings
from typing import Any

from stable_baselines3.common.preprocessing import is_image_space, is_image_space_channels_first
from stable_baselines3.common.utils import constant_fn
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn

from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv


warnings.filterwarnings("ignore", message="You are trying to run PPO on the GPU")

"""
Configuration Parser.
"""


def process_sb3_cfg(cfg: dict, num_envs: int) -> dict:
    """Convert simple YAML types to Stable-Baselines classes/components.

    Args:
        cfg: A configuration dictionary.
        num_envs: the number of parallel environments (used to compute `batch_size` for a desired number of minibatches)

    Returns:
        A dictionary containing the converted configuration.

    Reference:
        https://github.com/DLR-RM/rl-baselines3-zoo/blob/0e5eb145faefa33e7d79c7f8c179788574b20da5/utils/exp_manager.py#L358
    """

    def update_dict(hyperparams: dict[str, Any], depth: int) -> dict[str, Any]:
        for key, value in hyperparams.items():
            if isinstance(value, dict):
                update_dict(value, depth + 1)
            if isinstance(value, str):
                if value.startswith("nn."):
                    hyperparams[key] = getattr(nn, value[3:])
            if depth == 0:
                if key in ["learning_rate", "clip_range", "clip_range_vf"]:
                    if isinstance(value, str):
                        _, initial_value = value.split("_")
                        initial_value = float(initial_value)
                        hyperparams[key] = lambda progress_remaining: progress_remaining * initial_value
                    elif isinstance(value, (float, int)):

                        if value < 0:
                            continue
                        hyperparams[key] = constant_fn(float(value))
                    else:
                        raise ValueError(f"Invalid value for {key}: {hyperparams[key]}")


        if "n_minibatches" in hyperparams:
            hyperparams["batch_size"] = (hyperparams.get("n_steps", 2048) * num_envs) // hyperparams["n_minibatches"]
            del hyperparams["n_minibatches"]

        return hyperparams


    return update_dict(cfg, depth=0)


"""
Vectorized environment wrapper.
"""


class Sb3VecEnvWrapper(VecEnv):
    """Wraps around Isaac Lab environment for Stable Baselines3.

    Isaac Sim internally implements a vectorized environment. However, since it is
    still considered a single environment instance, Stable Baselines tries to wrap
    around it using the :class:`DummyVecEnv`. This is only done if the environment
    is not inheriting from their :class:`VecEnv`. Thus, this class thinly wraps
    over the environment from :class:`ManagerBasedRLEnv` or :class:`DirectRLEnv`.

    Note:
        While Stable-Baselines3 supports Gym 0.26+ API, their vectorized environment
        uses their own API (i.e. it is closer to Gym 0.21). Thus, we implement
        the API for the vectorized environment.

    We also add monitoring functionality that computes the un-discounted episode
    return and length. This information is added to the info dicts under key `episode`.

    In contrast to the Isaac Lab environment, stable-baselines expect the following:

    1. numpy datatype for MDP signals
    2. a list of info dicts for each sub-environment (instead of a dict)
    3. when environment has terminated, the observations from the environment should correspond
       to the one after reset. The "real" final observation is passed using the info dicts
       under the key ``terminal_observation``.

    .. warning::

        By the nature of physics stepping in Isaac Sim, it is not possible to forward the
        simulation buffers without performing a physics step. Thus, reset is performed
        inside the :meth:`step()` function after the actual physics step is taken.
        Thus, the returned observations for terminated environments is the one after the reset.

    .. caution::

        This class must be the last wrapper in the wrapper chain. This is because the wrapper does not follow
        the :class:`gym.Wrapper` interface. Any subsequent wrappers will need to be modified to work with this
        wrapper.

    Reference:

    1. https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html
    2. https://stable-baselines3.readthedocs.io/en/master/common/monitor.html

    """

    def __init__(self, env: ManagerBasedRLEnv | DirectRLEnv, fast_variant: bool = True):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap around.
            fast_variant: Use fast variant for processing info
                (Only episodic reward, lengths and truncation info are included)
        Raises:
            ValueError: When the environment is not an instance of :class:`ManagerBasedRLEnv` or :class:`DirectRLEnv`.
        """

        if not isinstance(env.unwrapped, ManagerBasedRLEnv) and not isinstance(env.unwrapped, DirectRLEnv):
            raise ValueError(
                "The environment must be inherited from ManagerBasedRLEnv or DirectRLEnv. Environment type:"
                f" {type(env)}"
            )

        self.env = env
        self.fast_variant = fast_variant

        self.num_envs = self.unwrapped.num_envs
        self.sim_device = self.unwrapped.device
        self.render_mode = self.unwrapped.render_mode
        self.observation_processors = {}
        self._process_spaces()

        self._ep_rew_buf = np.zeros(self.num_envs)
        self._ep_len_buf = np.zeros(self.num_envs)

    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    """
    Properties -- Gym.Wrapper
    """

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def unwrapped(self) -> ManagerBasedRLEnv | DirectRLEnv:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped

    """
    Properties
    """

    def get_episode_rewards(self) -> list[float]:
        """Returns the rewards of all the episodes."""
        return self._ep_rew_buf.tolist()

    def get_episode_lengths(self) -> list[int]:
        """Returns the number of time-steps of all the episodes."""
        return self._ep_len_buf.tolist()

    """
    Operations - MDP
    """

    def seed(self, seed: int | None = None) -> list[int | None]:
        return [self.unwrapped.seed(seed)] * self.unwrapped.num_envs

    def reset(self) -> VecEnvObs:
        obs_dict, _ = self.env.reset()

        self._ep_rew_buf = np.zeros(self.num_envs)
        self._ep_len_buf = np.zeros(self.num_envs)

        return self._process_obs(obs_dict)

    def step_async(self, actions):

        if not isinstance(actions, torch.Tensor):
            actions = np.asarray(actions)
            actions = torch.from_numpy(actions).to(device=self.sim_device, dtype=torch.float32)
        else:
            actions = actions.to(device=self.sim_device, dtype=torch.float32)

        self._async_actions = actions

    def step_wait(self) -> VecEnvStepReturn:

        obs_dict, rew, terminated, truncated, extras = self.env.step(self._async_actions)

        dones = terminated | truncated



        obs = self._process_obs(obs_dict)
        rewards = rew.detach().cpu().numpy()
        terminated = terminated.detach().cpu().numpy()
        truncated = truncated.detach().cpu().numpy()
        dones = dones.detach().cpu().numpy()

        reset_ids = dones.nonzero()[0]


        self._ep_rew_buf += rewards
        self._ep_len_buf += 1

        infos = self._process_extras(obs, terminated, truncated, extras, reset_ids)


        self._ep_rew_buf[reset_ids] = 0.0
        self._ep_len_buf[reset_ids] = 0

        return obs, rewards, dones, infos

    def close(self):
        self.env.close()

    def get_attr(self, attr_name, indices=None):

        if indices is None:
            indices = slice(None)
            num_indices = self.num_envs
        else:
            num_indices = len(indices)

        attr_val = getattr(self.env, attr_name)

        if not isinstance(attr_val, torch.Tensor):
            return [attr_val] * num_indices
        else:
            return attr_val[indices].detach().cpu().numpy()

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError("Setting attributes is not supported.")

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        if method_name == "render":

            return self.env.render()
        else:


            env_method = getattr(self.env, method_name)
            return env_method(*method_args, indices=indices, **method_kwargs)

    def env_is_wrapped(self, wrapper_class, indices=None):

        return [False]

    def get_images(self):
        raise NotImplementedError("Getting images is not supported.")

    """
    Helper functions.
    """

    def _process_spaces(self):

        observation_space = self.unwrapped.single_observation_space["policy"]
        if isinstance(observation_space, gym.spaces.Dict):
            for obs_key, obs_space in observation_space.spaces.items():
                processors: list[callable[[torch.Tensor], Any]] = []


                if is_image_space(obs_space, check_channels=True, normalized_image=True):
                    actually_normalized = np.all(obs_space.low == -1.0) and np.all(obs_space.high == 1.0)
                    if not actually_normalized:
                        if np.any(obs_space.low != 0) or np.any(obs_space.high != 255):
                            raise ValueError(
                                "Your image observation is not normalized in environment, and will not be"
                                "normalized by sb3 if its min is not 0 and max is not 255."
                            )

                        if obs_space.dtype != np.uint8:
                            processors.append(lambda obs: obs.to(torch.uint8))
                        observation_space.spaces[obs_key] = gym.spaces.Box(0, 255, obs_space.shape, np.uint8)
                    else:



                        if not is_image_space_channels_first(obs_space):

                            def tranp(img: torch.Tensor) -> torch.Tensor:
                                return img.permute(2, 0, 1) if len(img.shape) == 3 else img.permute(0, 3, 1, 2)

                            processors.append(tranp)
                            h, w, c = obs_space.shape
                            observation_space.spaces[obs_key] = gym.spaces.Box(-1.0, 1.0, (c, h, w), obs_space.dtype)

                    def chained_processor(obs: torch.Tensor, procs=processors) -> Any:
                        for proc in procs:
                            obs = proc(obs)
                        return obs


                    if len(processors) > 0:
                        self.observation_processors[obs_key] = chained_processor




        action_space = self.unwrapped.single_action_space
        if isinstance(action_space, gym.spaces.Box) and not action_space.is_bounded("both"):
            action_space = gym.spaces.Box(low=-100, high=100, shape=action_space.shape)


        VecEnv.__init__(self, self.num_envs, observation_space, action_space)

    def _process_obs(self, obs_dict: torch.Tensor | dict[str, torch.Tensor]) -> np.ndarray | dict[str, np.ndarray]:
        """Convert observations into NumPy data type."""

        obs = obs_dict["policy"]

        if isinstance(obs, dict):
            for key, value in obs.items():
                if key in self.observation_processors:
                    obs[key] = self.observation_processors[key](value)
                obs[key] = obs[key].detach().cpu().numpy()
        elif isinstance(obs, torch.Tensor):
            obs = obs.detach().cpu().numpy()
        else:
            raise NotImplementedError(f"Unsupported data type: {type(obs)}")
        return obs

    def _process_extras(
        self, obs: np.ndarray, terminated: np.ndarray, truncated: np.ndarray, extras: dict, reset_ids: np.ndarray
    ) -> list[dict[str, Any]]:
        """Convert miscellaneous information into dictionary for each sub-environment."""

        if self.fast_variant:
            infos = [{} for _ in range(self.num_envs)]

            for idx in reset_ids:

                infos[idx]["episode"] = {
                    "r": self._ep_rew_buf[idx],
                    "l": self._ep_len_buf[idx],
                }


                infos[idx]["TimeLimit.truncated"] = truncated[idx] and not terminated[idx]


                if isinstance(obs, dict):
                    terminal_obs = {key: value[idx] for key, value in obs.items()}
                else:
                    terminal_obs = obs[idx]
                infos[idx]["terminal_observation"] = terminal_obs

            return infos


        infos: list[dict[str, Any]] = [dict.fromkeys(extras.keys()) for _ in range(self.num_envs)]


        for idx in range(self.num_envs):

            if idx in reset_ids:
                infos[idx]["episode"] = dict()
                infos[idx]["episode"]["r"] = float(self._ep_rew_buf[idx])
                infos[idx]["episode"]["l"] = float(self._ep_len_buf[idx])
            else:
                infos[idx]["episode"] = None

            infos[idx]["TimeLimit.truncated"] = truncated[idx] and not terminated[idx]

            for key, value in extras.items():


                if key == "log":

                    if infos[idx]["episode"] is not None:
                        for sub_key, sub_value in value.items():
                            infos[idx]["episode"][sub_key] = sub_value
                else:
                    infos[idx][key] = value[idx]

            if idx in reset_ids:

                if isinstance(obs, dict):
                    terminal_obs = dict.fromkeys(obs.keys())
                    for key, value in obs.items():
                        terminal_obs[key] = value[idx]
                else:
                    terminal_obs = obs[idx]

                infos[idx]["terminal_observation"] = terminal_obs
            else:
                infos[idx]["terminal_observation"] = None

        return infos
