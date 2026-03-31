




"""Test dataset generation with SkillGen for Isaac Lab Mimic workflow."""

from isaaclab.app import AppLauncher


simulation_app = AppLauncher(headless=True).app

import os
import subprocess
import tempfile

import pytest

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, retrieve_file_path

DATASETS_DOWNLOAD_DIR = tempfile.mkdtemp(suffix="_Isaac-Stack-Cube-Franka-IK-Rel-Skillgen-v0")
NUCLEUS_SKILLGEN_ANNOTATED_DATASET_PATH = os.path.join(
    ISAACLAB_NUCLEUS_DIR, "Mimic", "franka_stack_datasets", "annotated_dataset_skillgen.hdf5"
)


@pytest.fixture
def setup_skillgen_test_environment():
    """Prepare environment for SkillGen dataset generation test."""

    if not os.path.exists(DATASETS_DOWNLOAD_DIR):
        print("Creating directory : ", DATASETS_DOWNLOAD_DIR)
        os.makedirs(DATASETS_DOWNLOAD_DIR)


    retrieve_file_path(NUCLEUS_SKILLGEN_ANNOTATED_DATASET_PATH, DATASETS_DOWNLOAD_DIR)


    pythonunbuffered_env_var_ = os.environ.get("PYTHONUNBUFFERED")
    os.environ["PYTHONUNBUFFERED"] = "1"


    current_dir = os.path.dirname(os.path.abspath(__file__))
    workflow_root = os.path.abspath(os.path.join(current_dir, "../../.."))


    yield workflow_root


    if pythonunbuffered_env_var_:
        os.environ["PYTHONUNBUFFERED"] = pythonunbuffered_env_var_
    else:
        del os.environ["PYTHONUNBUFFERED"]


def test_generate_dataset_skillgen(setup_skillgen_test_environment):
    """Test dataset generation with SkillGen enabled."""
    workflow_root = setup_skillgen_test_environment

    input_file = os.path.join(DATASETS_DOWNLOAD_DIR, "annotated_dataset_skillgen.hdf5")
    output_file = os.path.join(DATASETS_DOWNLOAD_DIR, "generated_dataset_skillgen.hdf5")

    command = [
        workflow_root + "/isaaclab.sh",
        "-p",
        os.path.join(workflow_root, "scripts/imitation_learning/isaaclab_mimic/generate_dataset.py"),
        "--device",
        "cpu",
        "--input_file",
        input_file,
        "--output_file",
        output_file,
        "--num_envs",
        "1",
        "--generation_num_trials",
        "1",
        "--use_skillgen",
        "--headless",
        "--task",
        "Isaac-Stack-Cube-Franka-IK-Rel-Skillgen-v0",
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    print("SkillGen dataset generation result:")
    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0, result.stderr
    expected_output = "successes/attempts. Exiting"
    assert expected_output in result.stdout
