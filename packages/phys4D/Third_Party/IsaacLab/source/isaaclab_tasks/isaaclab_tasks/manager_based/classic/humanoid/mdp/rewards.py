




from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

from . import observations as obs

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def upright_posture_bonus(
    env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for maintaining an upright posture."""
    up_proj = obs.base_up_proj(env, asset_cfg).squeeze(-1)
    return (up_proj > threshold).float()


def move_to_target_bonus(
    env: ManagerBasedRLEnv,
    threshold: float,
    target_pos: tuple[float, float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for moving to the target heading."""
    heading_proj = obs.base_heading_proj(env, target_pos, asset_cfg).squeeze(-1)
    return torch.where(heading_proj > threshold, 1.0, heading_proj / threshold)


class progress_reward(ManagerTermBase):
    """Reward for making progress towards the target."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):

        super().__init__(cfg, env)

        self.potentials = torch.zeros(env.num_envs, device=env.device)
        self.prev_potentials = torch.zeros_like(self.potentials)

    def reset(self, env_ids: torch.Tensor):

        asset: Articulation = self._env.scene["robot"]

        target_pos = torch.tensor(self.cfg.params["target_pos"], device=self.device)
        to_target_pos = target_pos - asset.data.root_pos_w[env_ids, :3]

        self.potentials[env_ids] = -torch.norm(to_target_pos, p=2, dim=-1) / self._env.step_dt
        self.prev_potentials[env_ids] = self.potentials[env_ids]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        target_pos: tuple[float, float, float],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:

        asset: Articulation = env.scene[asset_cfg.name]

        target_pos = torch.tensor(target_pos, device=env.device)
        to_target_pos = target_pos - asset.data.root_pos_w[:, :3]
        to_target_pos[:, 2] = 0.0

        self.prev_potentials[:] = self.potentials[:]
        self.potentials[:] = -torch.norm(to_target_pos, p=2, dim=-1) / env.step_dt

        return self.potentials - self.prev_potentials


class joint_pos_limits_penalty_ratio(ManagerTermBase):
    """Penalty for violating joint position limits weighted by the gear ratio."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):

        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))

        asset: Articulation = env.scene[asset_cfg.name]


        self.gear_ratio = torch.ones(env.num_envs, asset.num_joints, device=env.device)
        index_list, _, value_list = string_utils.resolve_matching_names_values(
            cfg.params["gear_ratio"], asset.joint_names
        )
        self.gear_ratio[:, index_list] = torch.tensor(value_list, device=env.device)
        self.gear_ratio_scaled = self.gear_ratio / torch.max(self.gear_ratio)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        threshold: float,
        gear_ratio: dict[str, float],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:

        asset: Articulation = env.scene[asset_cfg.name]

        joint_pos_scaled = math_utils.scale_transform(
            asset.data.joint_pos, asset.data.soft_joint_pos_limits[..., 0], asset.data.soft_joint_pos_limits[..., 1]
        )

        violation_amount = (torch.abs(joint_pos_scaled) - threshold) / (1 - threshold)
        violation_amount = violation_amount * self.gear_ratio_scaled

        return torch.sum((torch.abs(joint_pos_scaled) > threshold) * violation_amount, dim=-1)


class power_consumption(ManagerTermBase):
    """Penalty for the power consumed by the actions to the environment.

    This is computed as commanded torque times the joint velocity.
    """

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):

        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))

        asset: Articulation = env.scene[asset_cfg.name]


        self.gear_ratio = torch.ones(env.num_envs, asset.num_joints, device=env.device)
        index_list, _, value_list = string_utils.resolve_matching_names_values(
            cfg.params["gear_ratio"], asset.joint_names
        )
        self.gear_ratio[:, index_list] = torch.tensor(value_list, device=env.device)
        self.gear_ratio_scaled = self.gear_ratio / torch.max(self.gear_ratio)

    def __call__(
        self, env: ManagerBasedRLEnv, gear_ratio: dict[str, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:

        asset: Articulation = env.scene[asset_cfg.name]

        return torch.sum(torch.abs(env.action_manager.action * asset.data.joint_vel * self.gear_ratio_scaled), dim=-1)
