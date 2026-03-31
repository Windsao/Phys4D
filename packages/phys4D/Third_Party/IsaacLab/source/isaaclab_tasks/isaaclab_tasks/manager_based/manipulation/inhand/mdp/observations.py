




"""Functions specific to the in-hand dexterous manipulation environments."""

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from .commands import InHandReOrientationCommand


def goal_quat_diff(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, make_quat_unique: bool
) -> torch.Tensor:
    """Goal orientation relative to the asset's root frame.

    The quaternion is represented as (w, x, y, z). The real part is always positive.
    """

    asset: RigidObject = env.scene[asset_cfg.name]
    command_term: InHandReOrientationCommand = env.command_manager.get_term(command_name)


    goal_quat_w = command_term.command[:, 3:7]
    asset_quat_w = asset.data.root_quat_w


    quat = math_utils.quat_mul(asset_quat_w, math_utils.quat_conjugate(goal_quat_w))

    return math_utils.quat_unique(quat) if make_quat_unique else quat
