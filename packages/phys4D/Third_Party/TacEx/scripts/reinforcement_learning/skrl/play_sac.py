




"""
Script to play a checkpoint of an RL agent from skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher


parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="SAC",
    choices=["SAC"],
    help="The RL algorithm used for training the skrl agent.",
)


AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

import skrl
from packaging import version


SKRL_VERSION = "1.3.0"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch") or args_cli.ml_framework.startswith("jax"):
    pass

import copy


import torch.nn as nn

import omni.isaac.lab_tasks
from omni.isaac.lab.envs import DirectMARLEnv, multi_agent_to_single_agent
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab_tasks.utils import (
    get_checkpoint_path,
    load_cfg_from_registry,
    parse_env_cfg,
)
from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper


from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model


algorithm = args_cli.algorithm.lower()



class StochasticActor(GaussianMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions=False,
        clip_log_std=True,
        min_log_std=-5,
        max_log_std=2,
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_actions),
            nn.Tanh(),
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}


class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations + self.num_actions, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)), {}


def _process_cfg(cfg: dict) -> dict:
    """Convert simple types to skrl classes/components
    :param cfg: A configuration dictionary
    :return: Updated dictionary
    """
    _direct_eval = [
        "learning_rate_scheduler",
        "shared_state_preprocessor",
        "state_preprocessor",
        "value_preprocessor",
    ]

    def reward_shaper_function(scale):
        def reward_shaper(rewards, *args, **kwargs):
            return rewards * scale

        return reward_shaper

    def update_dict(d):
        for key, value in d.items():
            if isinstance(value, dict):
                update_dict(value)
            else:
                if key in _direct_eval:
                    if type(d[key]) is str:
                        d[key] = eval(value)
                elif key.endswith("_kwargs"):
                    d[key] = value if value is not None else {}
                elif key in ["rewards_shaper_scale"]:
                    d["rewards_shaper"] = reward_shaper_function(value)
        return d

    return update_dict(copy.deepcopy(cfg))


def main():
    """Play with skrl agent."""

    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"


    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    try:
        experiment_cfg = load_cfg_from_registry(args_cli.task, f"skrl_{algorithm}_cfg_entry_point")
    except ValueError:
        experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")


    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path,
            run_dir=f".*_{algorithm}_{args_cli.ml_framework}",
            other_dirs=["checkpoints"],
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))


    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

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


    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)


    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)



    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    device = env.device

    memory = RandomMemory(memory_size=15625, num_envs=env.num_envs, device=device)




    models = {
        "policy": StochasticActor(env.observation_space, env.action_space, device),
        "critic_1": Critic(env.observation_space, env.action_space, device),
        "critic_2": Critic(env.observation_space, env.action_space, device),
        "target_critic_1": Critic(env.observation_space, env.action_space, device),
        "target_critic_2": Critic(env.observation_space, env.action_space, device),
    }
    cfg = SAC_DEFAULT_CONFIG.copy()
    cfg.update(_process_cfg(experiment_cfg["agent"]))

    agent = SAC(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    agent.load(resume_path)

    agent.set_running_mode("eval")


    obs, _ = env.reset()
    timestep = 0

    while simulation_app.is_running():

        with torch.inference_mode():

            actions = agent.act(obs, timestep=0, timesteps=0)[0]

            obs, _, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1

            if timestep == args_cli.video_length:
                break


    env.close()


if __name__ == "__main__":

    main()

    simulation_app.close()
