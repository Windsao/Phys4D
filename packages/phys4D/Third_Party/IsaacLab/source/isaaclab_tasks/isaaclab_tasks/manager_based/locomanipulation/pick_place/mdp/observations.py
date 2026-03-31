




import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def upper_body_last_action(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Extract the last action of the upper body."""
    asset = env.scene[asset_cfg.name]
    joint_pos_target = asset.data.joint_pos_target


    joint_names = asset_cfg.joint_names if hasattr(asset_cfg, "joint_names") else None
    if joint_names is None:
        raise ValueError("asset_cfg must have 'joint_names' attribute for upper_body_last_action.")


    joint_indices, _ = asset.find_joints(joint_names)
    joint_indices = torch.tensor(joint_indices, dtype=torch.long)


    upper_body_joint_pos_target = joint_pos_target[:, joint_indices]

    return upper_body_joint_pos_target
