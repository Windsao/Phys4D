




"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--output_dir", type=str, default=None, help="Path to the output directory.")

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()
args_cli.headless = True


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg




def main():
    """Random actions agent with Isaac Lab environment."""

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=True)

    env = gym.make(args_cli.task, cfg=env_cfg)


    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    env.reset()

    outs = env.unwrapped.get_IO_descriptors
    out_observations = outs["observations"]
    out_actions = outs["actions"]
    out_articulations = outs["articulations"]
    out_scene = outs["scene"]

    import yaml

    name = args_cli.task.lower().replace("-", "_")
    name = name.replace(" ", "_")

    if not os.path.exists(args_cli.output_dir):
        os.makedirs(args_cli.output_dir)

    with open(os.path.join(args_cli.output_dir, f"{name}_IO_descriptors.yaml"), "w") as f:
        print(f"[INFO]: Exporting IO descriptors to {os.path.join(args_cli.output_dir, f'{name}_IO_descriptors.yaml')}")
        yaml.safe_dump(outs, f)

    for k in out_actions:
        print(f"--- Action term: {k['name']} ---")
        k.pop("name")
        for k1, v1 in k.items():
            print(f"{k1}: {v1}")

    for obs_group_name, obs_group in out_observations.items():
        print(f"--- Obs group: {obs_group_name} ---")
        for k in obs_group:
            print(f"--- Obs term: {k['name']} ---")
            k.pop("name")
            for k1, v1 in k.items():
                print(f"{k1}: {v1}")

    for articulation_name, articulation_data in out_articulations.items():
        print(f"--- Articulation: {articulation_name} ---")
        for k1, v1 in articulation_data.items():
            print(f"{k1}: {v1}")

    for k1, v1 in out_scene.items():
        print(f"{k1}: {v1}")

    env.step(torch.zeros(env.action_space.shape, device=env.unwrapped.device))
    env.close()


if __name__ == "__main__":

    main()

    simulation_app.close()
