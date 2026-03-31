




"""
This scripts run training with different RL libraries over a subset of the environments.

It calls the script ``scripts/reinforcement_learning/${args.lib_name}/train.py`` with the appropriate arguments.
Each training run has the corresponding "commit tag" appended to the run name, which allows comparing different
training logs of the same environments.

Example usage:

.. code-block:: bash
    # for rsl-rl
    python run_train_envs.py --lib-name rsl_rl

"""

import argparse
import subprocess

from test_settings import ISAACLAB_PATH, TEST_RL_ENVS


def parse_args() -> argparse.Namespace:
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lib-name",
        type=str,
        default="rsl_rl",
        choices=["rsl_rl", "skrl", "rl_games", "sb3"],
        help="The name of the library to use for training.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace):
    """The main function."""

    git_commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()


    if args.lib_name == "rsl_rl":
        extra_args = ["--run_name", git_commit_hash]
    else:

        extra_args = []


    for env_name in TEST_RL_ENVS:


        print("\033[91m==============================================\033[0m")
        print("\033[91m==============================================\033[0m")
        print(f"\033[91mTraining on {env_name} with {args.lib_name}...\033[0m")
        print("\033[91m==============================================\033[0m")
        print("\033[91m==============================================\033[0m")


        subprocess.run(
            [
                f"{ISAACLAB_PATH}/isaaclab.sh",
                "-p",
                f"{ISAACLAB_PATH}/scripts/reinforcement_learning/{args.lib_name}/train.py",
                "--task",
                env_name,
                "--headless",
            ]
            + extra_args,
            check=False,
        )


if __name__ == "__main__":
    args_cli = parse_args()
    main(args_cli)
