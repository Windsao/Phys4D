from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from magicsim.Env.Environment.Isaac import IsaacRLEnv


def ee_frame_pos(
    env: IsaacRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos = ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins[:, 0:3]

    return ee_frame_pos


def ee_frame_quat(
    env, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_quat = ee_frame.data.target_quat_w[:, 0, :]

    return ee_frame_quat


def gripper_pos(
    env, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    finger_joint_1 = robot.data.joint_pos[:, -1].clone().unsqueeze(1)
    finger_joint_2 = -1 * robot.data.joint_pos[:, -2].clone().unsqueeze(1)

    return torch.cat((finger_joint_1, finger_joint_2), dim=1)


def ee_rel_pos(
    env: IsaacRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos = ee_frame.data.target_pos_source[:, 0, :]
    return ee_frame_pos


def ee_rel_pos_arm_base(
    env: IsaacRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    ee_frame_pos = (
        ee_frame.data.target_pos_source[:, 0, :]
        - ee_frame.data.target_pos_source[:, 1, :]
    )

    return ee_frame_pos


def ee_rel_quat(
    env: IsaacRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_quat = ee_frame.data.target_quat_source[:, 0, :]
    return ee_frame_quat


def base_pos(env, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    base_pos = robot.data.root_pos_w - env.scene.env_origins[:, 0:3]
    return base_pos


def base_quat(env, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    base_quat = robot.data.root_quat_w
    return base_quat


def front_steer(
    env, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    joint_ids, _ = robot.find_joints(
        ["Knuckle__Upright__Front_Left", "Knuckle__Upright__Front_Right"]
    )
    steers = robot.data.joint_pos[:, joint_ids]
    return steers
