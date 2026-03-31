




"""
This script demonstrates how to run the RL environment for the cartpole balancing task.

.. code-block:: bash

    ./isaaclab.sh -p scripts/tutorials/03_envs/run_cartpole_rl_env.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")


AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import CartpoleEnvCfg


def main():
    """Main function."""

    env_cfg = CartpoleEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    env = ManagerBasedRLEnv(cfg=env_cfg)


    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():

            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")

            joint_efforts = torch.randn_like(env.action_manager.action)

            obs, rew, terminated, truncated, info = env.step(joint_efforts)

            print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())

            count += 1


    env.close()


if __name__ == "__main__":

    main()

    simulation_app.close()
