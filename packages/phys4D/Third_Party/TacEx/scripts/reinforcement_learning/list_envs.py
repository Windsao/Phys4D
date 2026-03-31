"""
Script to print all the available environments in the extension.

The script iterates over all registered environments and stores the details in a table.
It prints the name of the environment, the entry point and the config file.
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher


app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
from prettytable import PrettyTable


import tacex_tasks

TEMPLATE = "TacEx"


def main():
    """Print all TacEx environments registered in `tacex_tasks` extension."""

    table = PrettyTable(["S. No.", "Task Name", "Entry Point", "Config"])
    table.title = "Available Environments in TacEx"

    table.align["Task Name"] = "l"
    table.align["Entry Point"] = "l"
    table.align["Config"] = "l"


    index = 0

    for task_spec in gym.registry.values():
        if f"{TEMPLATE}-" in task_spec.id:

            table.add_row([index + 1, task_spec.id, task_spec.entry_point, task_spec.kwargs["env_cfg_entry_point"]])

            index += 1

    print(table)


if __name__ == "__main__":
    try:

        main()
    except Exception as e:
        raise e
    finally:

        simulation_app.close()
