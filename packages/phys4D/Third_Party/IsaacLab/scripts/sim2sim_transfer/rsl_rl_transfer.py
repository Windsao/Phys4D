




"""Script to play a checkpoint of an RL agent from RSL-RL with policy transfer capabilities."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher


sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from scripts.reinforcement_learning.rsl_rl import cli_args


parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL with policy transfer.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

parser.add_argument(
    "--policy_transfer_file",
    type=str,
    default=None,
    help="Path to YAML file containing joint mapping configuration for policy transfer between physics engines.",
)

cli_args.add_rsl_rl_args(parser)

AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True


sys.argv = [sys.argv[0]] + hydra_args


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import yaml

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config




def get_joint_mappings(args_cli, action_space_dim):
    """Get joint mappings based on command line arguments.

    Args:
            args_cli: Command line arguments
            action_space_dim: Dimension of the action space (number of joints)

    Returns:
            tuple: (source_to_target_list, target_to_source_list, source_to_target_obs_list)
    """
    num_joints = action_space_dim
    if args_cli.policy_transfer_file:

        try:
            with open(args_cli.policy_transfer_file) as file:
                config = yaml.safe_load(file)
        except Exception as e:
            raise RuntimeError(f"Failed to load joint mapping from {args_cli.policy_transfer_file}: {e}")

        source_joint_names = config["source_joint_names"]
        target_joint_names = config["target_joint_names"]

        source_to_target = []
        target_to_source = []


        for joint_name in source_joint_names:
            if joint_name in target_joint_names:
                source_to_target.append(target_joint_names.index(joint_name))
            else:
                raise ValueError(f"Joint '{joint_name}' not found in target joint names")


        for joint_name in target_joint_names:
            if joint_name in source_joint_names:
                target_to_source.append(source_joint_names.index(joint_name))
            else:
                raise ValueError(f"Joint '{joint_name}' not found in source joint names")
        print(f"[INFO] Loaded joint mapping for policy transfer from YAML: {args_cli.policy_transfer_file}")
        assert (
            len(source_to_target) == len(target_to_source) == num_joints
        ), "Number of source and target joints must match"
    else:

        identity_map = list(range(num_joints))
        source_to_target, target_to_source = identity_map, identity_map


    obs_map = (
        [0, 1, 2]
        + [3, 4, 5]
        + [6, 7, 8]
        + [9, 10, 11]
        + [i + 12 + num_joints * 0 for i in source_to_target]
        + [i + 12 + num_joints * 1 for i in source_to_target]
        + [i + 12 + num_joints * 2 for i in source_to_target]
    )

    return source_to_target, target_to_source, obs_map


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent with policy transfer capabilities."""


    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs



    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device


    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)


    env_cfg.log_dir = log_dir


    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)


    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)


    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)


    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)


    policy = runner.get_inference_policy(device=env.unwrapped.device)



    try:

        policy_nn = runner.alg.policy
    except AttributeError:

        policy_nn = runner.alg.actor_critic


    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None


    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt


    obs = env.get_observations()
    timestep = 0


    _, target_to_source, obs_map = get_joint_mappings(args_cli, env.action_space.shape[1])


    device = args_cli.device if args_cli.device else "cuda:0"
    target_to_source_tensor = torch.tensor(target_to_source, device=device) if target_to_source else None
    obs_map_tensor = torch.tensor(obs_map, device=device) if obs_map else None

    def remap_obs(obs):
        """Remap the observation to the target observation space."""
        if obs_map_tensor is not None:
            obs = obs[:, obs_map_tensor]
        return obs

    def remap_actions(actions):
        """Remap the actions to the target action space."""
        if target_to_source_tensor is not None:
            actions = actions[:, target_to_source_tensor]
        return actions


    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():

            actions = policy(remap_obs(obs))

            obs, _, _, _ = env.step(remap_actions(actions))
        if args_cli.video:
            timestep += 1

            if timestep == args_cli.video_length:
                break


        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)


    env.close()


if __name__ == "__main__":

    main()

    simulation_app.close()
