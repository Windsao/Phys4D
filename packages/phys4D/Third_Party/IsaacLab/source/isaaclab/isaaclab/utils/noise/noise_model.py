




from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import noise_cfg






def constant_noise(data: torch.Tensor, cfg: noise_cfg.ConstantNoiseCfg) -> torch.Tensor:
    """Applies a constant noise bias to a given data set.

    Args:
        data: The unmodified data set to apply noise to.
        cfg: The configuration parameters for constant noise.

    Returns:
        The data modified by the noise parameters provided.
    """


    if isinstance(cfg.bias, torch.Tensor):
        cfg.bias = cfg.bias.to(device=data.device)

    if cfg.operation == "add":
        return data + cfg.bias
    elif cfg.operation == "scale":
        return data * cfg.bias
    elif cfg.operation == "abs":
        return torch.zeros_like(data) + cfg.bias
    else:
        raise ValueError(f"Unknown operation in noise: {cfg.operation}")


def uniform_noise(data: torch.Tensor, cfg: noise_cfg.UniformNoiseCfg) -> torch.Tensor:
    """Applies a uniform noise to a given data set.

    Args:
        data: The unmodified data set to apply noise to.
        cfg: The configuration parameters for uniform noise.

    Returns:
        The data modified by the noise parameters provided.
    """


    if isinstance(cfg.n_max, torch.Tensor):
        cfg.n_max = cfg.n_max.to(data.device)

    if isinstance(cfg.n_min, torch.Tensor):
        cfg.n_min = cfg.n_min.to(data.device)

    if cfg.operation == "add":
        return data + torch.rand_like(data) * (cfg.n_max - cfg.n_min) + cfg.n_min
    elif cfg.operation == "scale":
        return data * (torch.rand_like(data) * (cfg.n_max - cfg.n_min) + cfg.n_min)
    elif cfg.operation == "abs":
        return torch.rand_like(data) * (cfg.n_max - cfg.n_min) + cfg.n_min
    else:
        raise ValueError(f"Unknown operation in noise: {cfg.operation}")


def gaussian_noise(data: torch.Tensor, cfg: noise_cfg.GaussianNoiseCfg) -> torch.Tensor:
    """Applies a gaussian noise to a given data set.

    Args:
        data: The unmodified data set to apply noise to.
        cfg: The configuration parameters for gaussian noise.

    Returns:
        The data modified by the noise parameters provided.
    """


    if isinstance(cfg.mean, torch.Tensor):
        cfg.mean = cfg.mean.to(data.device)

    if isinstance(cfg.std, torch.Tensor):
        cfg.std = cfg.std.to(data.device)

    if cfg.operation == "add":
        return data + cfg.mean + cfg.std * torch.randn_like(data)
    elif cfg.operation == "scale":
        return data * (cfg.mean + cfg.std * torch.randn_like(data))
    elif cfg.operation == "abs":
        return cfg.mean + cfg.std * torch.randn_like(data)
    else:
        raise ValueError(f"Unknown operation in noise: {cfg.operation}")







class NoiseModel:
    """Base class for noise models."""

    def __init__(self, noise_model_cfg: noise_cfg.NoiseModelCfg, num_envs: int, device: str):
        """Initialize the noise model.

        Args:
            noise_model_cfg: The noise configuration to use.
            num_envs: The number of environments.
            device: The device to use for the noise model.
        """
        self._noise_model_cfg = noise_model_cfg
        self._num_envs = num_envs
        self._device = device

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset the noise model.

        This method can be implemented by derived classes to reset the noise model.
        This is useful when implementing temporal noise models such as random walk.

        Args:
            env_ids: The environment ids to reset the noise model for. Defaults to None,
                in which case all environments are considered.
        """
        pass

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the noise to the data.

        Args:
            data: The data to apply the noise to. Shape is (num_envs, ...).

        Returns:
            The data with the noise applied. Shape is the same as the input data.
        """
        return self._noise_model_cfg.noise_cfg.func(data, self._noise_model_cfg.noise_cfg)


class NoiseModelWithAdditiveBias(NoiseModel):
    """Noise model with an additive bias.

    The bias term is sampled from a the specified distribution on reset.
    """

    def __init__(self, noise_model_cfg: noise_cfg.NoiseModelWithAdditiveBiasCfg, num_envs: int, device: str):

        super().__init__(noise_model_cfg, num_envs, device)

        self._bias_noise_cfg = noise_model_cfg.bias_noise_cfg
        self._bias = torch.zeros((num_envs, 1), device=self._device)
        self._num_components: int | None = None
        self._sample_bias_per_component = noise_model_cfg.sample_bias_per_component

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset the noise model.

        This method resets the bias term for the specified environments.

        Args:
            env_ids: The environment ids to reset the noise model for. Defaults to None,
                in which case all environments are considered.
        """

        if env_ids is None:
            env_ids = slice(None)

        self._bias[env_ids] = self._bias_noise_cfg.func(self._bias[env_ids], self._bias_noise_cfg)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply bias noise to the data.

        Args:
            data: The data to apply the noise to. Shape is (num_envs, ...).

        Returns:
            The data with the noise applied. Shape is the same as the input data.
        """

        if self._sample_bias_per_component and self._num_components is None:
            *_, self._num_components = data.shape

            self._bias = self._bias.repeat(1, self._num_components)

            self.reset()
        return super().__call__(data) + self._bias
