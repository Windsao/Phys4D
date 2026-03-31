




"""This script sets up the vs-code settings for the Isaac Lab project.

This script merges the python.analysis.extraPaths from the "{ISAACSIM_DIR}/.vscode/settings.json" file into
the ".vscode/settings.json" file.

This is necessary because Isaac Sim 2022.2.1 onwards does not add the necessary python packages to the python path
when the "setup_python_env.sh" is run as part of the vs-code launch configuration.
"""

import argparse
import os
import pathlib
import platform
import re
import sys

PROJECT_DIR = pathlib.Path(__file__).parents[2]
"""Path to the the project directory."""

try:
    import isaacsim

    isaacsim_dir = os.environ.get("ISAAC_PATH", "")
except ModuleNotFoundError or ImportError:

    parser = argparse.ArgumentParser(description="Setup the VSCode settings for the project.")
    parser.add_argument("--isaac_path", type=str, help="The absolute path to the Isaac Sim installation.")
    args = parser.parse_args()


    isaacsim_dir = args.isaac_path

    if not os.path.exists(isaacsim_dir):
        raise FileNotFoundError(
            f"Could not find the isaac-sim directory: {isaacsim_dir}. Please provide the correct path to the Isaac Sim"
            " installation."
        )
except EOFError:
    print("Unable to trigger EULA acceptance. This is likely due to the script being run in a non-interactive shell.")
    print("Please run the script in an interactive shell to accept the EULA.")
    print("Skipping the setup of the VSCode settings...")
    sys.exit(0)


if not os.path.exists(isaacsim_dir):
    raise FileNotFoundError(
        f"Could not find the isaac-sim directory: {isaacsim_dir}. There are two possible reasons for this:"
        "\n\t1. The Isaac Sim directory does not exist as provided CLI path."
        "\n\t2. The script could import the 'isaacsim' package. This could be due to the 'isaacsim' package not being "
        "installed in the Python environment.\n"
        "\nPlease make sure that the Isaac Sim directory exists or that the 'isaacsim' package is installed."
    )

ISAACSIM_DIR = isaacsim_dir
"""Path to the isaac-sim directory."""


def overwrite_python_analysis_extra_paths(isaaclab_settings: str) -> str:
    """Overwrite the python.analysis.extraPaths in the Isaac Lab settings file.

    The extraPaths are replaced with the path names from the isaac-sim settings file that exists in the
    "{ISAACSIM_DIR}/.vscode/settings.json" file.

    If the isaac-sim settings file does not exist, the extraPaths are not overwritten.

    Args:
        isaaclab_settings: The settings string to use as template.

    Returns:
        The settings string with overwritten python analysis extra paths.
    """

    isaacsim_vscode_filename = os.path.join(ISAACSIM_DIR, ".vscode", "settings.json")



    if os.path.exists(isaacsim_vscode_filename):

        with open(isaacsim_vscode_filename) as f:
            vscode_settings = f.read()


        settings = re.search(
            r"\"python.analysis.extraPaths\": \[.*?\]", vscode_settings, flags=re.MULTILINE | re.DOTALL
        )
        settings = settings.group(0)
        settings = settings.split('"python.analysis.extraPaths": [')[-1]
        settings = settings.split("]")[0]


        path_names = settings.split(",")
        path_names = [path_name.strip().strip('"') for path_name in path_names]
        path_names = [path_name for path_name in path_names if len(path_name) > 0]


        rel_path = os.path.relpath(ISAACSIM_DIR, PROJECT_DIR)
        path_names = ['"${workspaceFolder}/' + rel_path + "/" + path_name + '"' for path_name in path_names]
    else:
        path_names = []
        print(
            f"[WARN] Could not find Isaac Sim VSCode settings: {isaacsim_vscode_filename}."
            "\n\tThis will result in missing 'python.analysis.extraPaths' in the VSCode"
            "\n\tsettings, which limits the functionality of the Python language server."
            "\n\tHowever, it does not affect the functionality of the Isaac Lab project."
            "\n\tWe are working on a fix for this issue with the Isaac Sim team."
        )


    isaaclab_extensions = os.listdir(os.path.join(PROJECT_DIR, "source"))
    path_names.extend(['"${workspaceFolder}/source/' + ext + '"' for ext in isaaclab_extensions])


    path_names = ",\n\t\t".expandtabs(4).join(path_names)

    path_names = path_names.replace("\\", "/")


    isaaclab_settings = re.sub(
        r"\"python.analysis.extraPaths\": \[.*?\]",
        '"python.analysis.extraPaths": [\n\t\t'.expandtabs(4) + path_names + "\n\t]".expandtabs(4),
        isaaclab_settings,
        flags=re.DOTALL,
    )

    return isaaclab_settings


def overwrite_default_python_interpreter(isaaclab_settings: str) -> str:
    """Overwrite the default python interpreter in the Isaac Lab settings file.

    The default python interpreter is replaced with the path to the python interpreter used by the
    isaac-sim project. This is necessary because the default python interpreter is the one shipped with
    isaac-sim.

    Args:
        isaaclab_settings: The settings string to use as template.

    Returns:
        The settings string with overwritten default python interpreter.
    """

    python_exe = os.path.normpath(sys.executable)



    if f"kit{os.sep}python{os.sep}bin{os.sep}python" in python_exe:

        if platform.system() == "Windows":
            python_exe = python_exe.replace(f"kit{os.sep}python{os.sep}bin{os.sep}python3", "python.bat")
        else:
            python_exe = python_exe.replace(f"kit{os.sep}python{os.sep}bin{os.sep}python3", "python.sh")



    isaaclab_settings = re.sub(
        r"\"python.defaultInterpreterPath\": \".*?\"",
        f'"python.defaultInterpreterPath": "{python_exe}"',
        isaaclab_settings,
        flags=re.DOTALL,
    )

    return isaaclab_settings


def main():

    isaaclab_vscode_template_filename = os.path.join(PROJECT_DIR, ".vscode", "tools", "settings.template.json")

    if not os.path.exists(isaaclab_vscode_template_filename):
        raise FileNotFoundError(
            f"Could not find the Isaac Lab template settings file: {isaaclab_vscode_template_filename}"
        )

    with open(isaaclab_vscode_template_filename) as f:
        isaaclab_template_settings = f.read()


    isaaclab_settings = overwrite_python_analysis_extra_paths(isaaclab_template_settings)


    isaaclab_settings = overwrite_default_python_interpreter(isaaclab_settings)


    header_message = (
        "// This file is a template and is automatically generated by the setup_vscode.py script.\n"
        "// Do not edit this file directly.\n"
        "// \n"
        f"// Generated from: {isaaclab_vscode_template_filename}\n"
    )
    isaaclab_settings = header_message + isaaclab_settings


    isaaclab_vscode_filename = os.path.join(PROJECT_DIR, ".vscode", "settings.json")
    with open(isaaclab_vscode_filename, "w") as f:
        f.write(isaaclab_settings)


    isaaclab_vscode_launch_filename = os.path.join(PROJECT_DIR, ".vscode", "launch.json")
    isaaclab_vscode_template_launch_filename = os.path.join(PROJECT_DIR, ".vscode", "tools", "launch.template.json")
    if not os.path.exists(isaaclab_vscode_launch_filename):

        with open(isaaclab_vscode_template_launch_filename) as f:
            isaaclab_template_launch_settings = f.read()

        header_message = header_message.replace(
            isaaclab_vscode_template_filename, isaaclab_vscode_template_launch_filename
        )
        isaaclab_launch_settings = header_message + isaaclab_template_launch_settings

        with open(isaaclab_vscode_launch_filename, "w") as f:
            f.write(isaaclab_launch_settings)


if __name__ == "__main__":
    main()
