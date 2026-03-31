




"""Functions to specify the symmetry in the observation and action space for cartpole."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


__all__ = ["compute_symmetric_states"]


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Augments the given observations and actions by applying symmetry transformations.

    This function creates augmented versions of the provided observations and actions by applying
    two symmetrical transformations: original, left-right. The symmetry
    transformations are beneficial for reinforcement learning tasks by providing additional
    diverse data without requiring additional data collection.

    Args:
        env: The environment instance.
        obs: The original observation tensor dictionary. Defaults to None.
        actions: The original actions tensor. Defaults to None.

    Returns:
        Augmented observations and actions tensors, or None if the respective input was None.
    """


    if obs is not None:
        batch_size = obs.batch_size[0]

        obs_aug = obs.repeat(2)

        obs_aug["policy"][:batch_size] = obs["policy"][:]

        obs_aug["policy"][batch_size : 2 * batch_size] = -obs["policy"]
    else:
        obs_aug = None


    if actions is not None:
        batch_size = actions.shape[0]

        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)

        actions_aug[:batch_size] = actions[:]

        actions_aug[batch_size : 2 * batch_size] = -actions
    else:
        actions_aug = None

    return obs_aug, actions_aug
