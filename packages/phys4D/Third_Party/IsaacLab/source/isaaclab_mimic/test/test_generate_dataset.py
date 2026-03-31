




"""Test dataset generation for Isaac Lab Mimic workflow."""

from isaaclab.app import AppLauncher


simulation_app = AppLauncher(headless=True).app

import os
import subprocess
import tempfile

import pytest

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, retrieve_file_path

DATASETS_DOWNLOAD_DIR = tempfile.mkdtemp(suffix="_Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0")
NUCLEUS_DATASET_PATH = os.path.join(ISAACLAB_NUCLEUS_DIR, "Tests", "Mimic", "dataset.hdf5")
EXPECTED_SUCCESSFUL_ANNOTATIONS = 10


@pytest.fixture
def setup_test_environment():
    """Set up the environment for testing."""

    if not os.path.exists(DATASETS_DOWNLOAD_DIR):
        print("Creating directory : ", DATASETS_DOWNLOAD_DIR)
        os.makedirs(DATASETS_DOWNLOAD_DIR)


    try:
        retrieve_file_path(NUCLEUS_DATASET_PATH, DATASETS_DOWNLOAD_DIR)
    except Exception as e:
        print(e)
        print("Could not download dataset from Nucleus")
        pytest.fail(
            "The dataset required for this test is currently unavailable. Dataset path: " + NUCLEUS_DATASET_PATH
        )


    pythonunbuffered_env_var_ = os.environ.get("PYTHONUNBUFFERED")
    os.environ["PYTHONUNBUFFERED"] = "1"


    current_dir = os.path.dirname(os.path.abspath(__file__))
    workflow_root = os.path.abspath(os.path.join(current_dir, "../../.."))


    config_command = [
        workflow_root + "/isaaclab.sh",
        "-p",
        os.path.join(workflow_root, "scripts/imitation_learning/isaaclab_mimic/annotate_demos.py"),
        "--task",
        "Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0",
        "--input_file",
        DATASETS_DOWNLOAD_DIR + "/dataset.hdf5",
        "--output_file",
        DATASETS_DOWNLOAD_DIR + "/annotated_dataset.hdf5",
        "--auto",
        "--headless",
    ]
    print(config_command)


    result = subprocess.run(config_command, capture_output=True, text=True)

    print(f"Annotate demos result: {result.returncode}\n\n\n\n\n\n\n\n\n\n\n\n")


    print("Config generation result:")
    print(result.stdout)
    print(result.stderr)


    assert result.returncode == 0, result.stderr



    success_line = None
    for line in result.stdout.split("\n"):
        if "Successful task completions:" in line:
            success_line = line
            break

    assert success_line is not None, "Could not find 'Successful task completions:' in output"


    try:
        successful_count = int(success_line.split(":")[-1].strip())
        assert (
            successful_count == EXPECTED_SUCCESSFUL_ANNOTATIONS
        ), f"Expected 10 successful annotations but got {successful_count}"
    except (ValueError, IndexError) as e:
        pytest.fail(f"Could not parse successful task count from line: '{success_line}'. Error: {e}")


    yield workflow_root


    if pythonunbuffered_env_var_:
        os.environ["PYTHONUNBUFFERED"] = pythonunbuffered_env_var_
    else:
        del os.environ["PYTHONUNBUFFERED"]


@pytest.mark.isaacsim_ci
def test_generate_dataset(setup_test_environment):
    """Test the dataset generation script."""
    workflow_root = setup_test_environment


    command = [
        workflow_root + "/isaaclab.sh",
        "-p",
        os.path.join(workflow_root, "scripts/imitation_learning/isaaclab_mimic/generate_dataset.py"),
        "--input_file",
        DATASETS_DOWNLOAD_DIR + "/annotated_dataset.hdf5",
        "--output_file",
        DATASETS_DOWNLOAD_DIR + "/generated_dataset.hdf5",
        "--generation_num_trials",
        "1",
        "--headless",
    ]


    result = subprocess.run(command, capture_output=True, text=True)


    print("Dataset generation result:")
    print(result.stdout)
    print(result.stderr)


    assert result.returncode == 0, result.stderr


    expected_output = "successes/attempts. Exiting"
    assert expected_output in result.stdout
