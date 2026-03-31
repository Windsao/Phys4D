




from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .state_file import StateFile


class ContainerInterface:
    """A helper class for managing Isaac Lab containers."""

    def __init__(
        self,
        context_dir: Path,
        profile: str = "base",
        yamls: list[str] | None = None,
        envs: list[str] | None = None,
        statefile: StateFile | None = None,
        suffix: str | None = None,
    ):
        """Initialize the container interface with the given parameters.

        Args:
            context_dir: The context directory for Docker operations.
            profile: The profile name for the container. Defaults to "base".
            yamls: A list of yaml files to extend ``docker-compose.yaml`` settings. These are extended in the order
                they are provided.
            envs: A list of environment variable files to extend the ``.env.base`` file. These are extended in the order
                they are provided.
            statefile: An instance of the :class:`Statefile` class to manage state variables. Defaults to None, in
                which case a new configuration object is created by reading the configuration file at the path
                ``context_dir/.container.cfg``.
            suffix: Optional docker image and container name suffix.  Defaults to None, in which case, the docker name
                suffix is set to the empty string. A hyphen is inserted in between the profile and the suffix if
                the suffix is a nonempty string.  For example, if "base" is passed to profile, and "custom" is
                passed to suffix, then the produced docker image and container will be named ``isaac-lab-base-custom``.
        """

        self.context_dir = context_dir



        if statefile is None:
            self.statefile = StateFile(path=self.context_dir / ".container.cfg")
        else:
            self.statefile = statefile


        self.profile = profile
        if self.profile == "isaaclab":


            self.profile = "base"


        if suffix is None or suffix == "":

            self.suffix = ""
        else:

            self.suffix = f"-{suffix}"

        self.container_name = f"isaac-lab-{self.profile}{self.suffix}"
        self.image_name = f"isaac-lab-{self.profile}{self.suffix}:latest"



        self.environ = os.environ.copy()
        self.environ["DOCKER_NAME_SUFFIX"] = self.suffix


        self._resolve_image_extension(yamls, envs)

        self._parse_dot_vars()

    """
    Operations.
    """

    def is_container_running(self) -> bool:
        """Check if the container is running.

        Returns:
            True if the container is running, otherwise False.
        """
        status = subprocess.run(
            ["docker", "container", "inspect", "-f", "{{.State.Status}}", self.container_name],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        return status == "running"

    def does_image_exist(self) -> bool:
        """Check if the Docker image exists.

        Returns:
            True if the image exists, otherwise False.
        """
        result = subprocess.run(["docker", "image", "inspect", self.image_name], capture_output=True, text=True)
        return result.returncode == 0

    def start(self):
        """Build and start the Docker container using the Docker compose command."""
        print(
            f"[INFO] Building the docker image and starting the container '{self.container_name}' in the"
            " background...\n"
        )

        container_history_file = self.context_dir / ".isaac-lab-docker-history"
        if not container_history_file.exists():

            container_history_file.touch(mode=0o2644, exist_ok=True)


        if self.profile != "base":
            subprocess.run(
                [
                    "docker",
                    "compose",
                    "--file",
                    "docker-compose.yaml",
                    "--env-file",
                    ".env.base",
                    "build",
                    "isaac-lab-base",
                ],
                check=False,
                cwd=self.context_dir,
                env=self.environ,
            )


        subprocess.run(
            ["docker", "compose"]
            + self.add_yamls
            + self.add_profiles
            + self.add_env_files
            + ["up", "--detach", "--build", "--remove-orphans"],
            check=False,
            cwd=self.context_dir,
            env=self.environ,
        )

    def enter(self):
        """Enter the running container by executing a bash shell.

        Raises:
            RuntimeError: If the container is not running.
        """
        if self.is_container_running():
            print(f"[INFO] Entering the existing '{self.container_name}' container in a bash session...\n")
            subprocess.run([
                "docker",
                "exec",
                "--interactive",
                "--tty",
                *(["-e", f"DISPLAY={os.environ['DISPLAY']}"] if "DISPLAY" in os.environ else []),
                f"{self.container_name}",
                "bash",
            ])
        else:
            raise RuntimeError(f"The container '{self.container_name}' is not running.")

    def stop(self):
        """Stop the running container using the Docker compose command.

        Raises:
            RuntimeError: If the container is not running.
        """
        if self.is_container_running():
            print(f"[INFO] Stopping the launched docker container '{self.container_name}'...\n")
            subprocess.run(
                ["docker", "compose"] + self.add_yamls + self.add_profiles + self.add_env_files + ["down", "--volumes"],
                check=False,
                cwd=self.context_dir,
                env=self.environ,
            )
        else:
            raise RuntimeError(f"Can't stop container '{self.container_name}' as it is not running.")

    def copy(self, output_dir: Path | None = None):
        """Copy artifacts from the running container to the host machine.

        Args:
            output_dir: The directory to copy the artifacts to. Defaults to None, in which case
                the context directory is used.

        Raises:
            RuntimeError: If the container is not running.
        """
        if self.is_container_running():
            print(f"[INFO] Copying artifacts from the '{self.container_name}' container...\n")
            if output_dir is None:
                output_dir = self.context_dir


            output_dir = output_dir.joinpath("artifacts")
            if not output_dir.is_dir():
                output_dir.mkdir()


            docker_isaac_lab_path = Path(self.dot_vars["DOCKER_ISAACLAB_PATH"])
            artifacts = {
                docker_isaac_lab_path.joinpath("logs"): output_dir.joinpath("logs"),
                docker_isaac_lab_path.joinpath("docs/_build"): output_dir.joinpath("docs"),
                docker_isaac_lab_path.joinpath("data_storage"): output_dir.joinpath("data_storage"),
            }

            for container_path, host_path in artifacts.items():
                print(f"\t -{container_path} -> {host_path}")

            for path in artifacts.values():
                shutil.rmtree(path, ignore_errors=True)


            for container_path, host_path in artifacts.items():
                subprocess.run(
                    [
                        "docker",
                        "cp",
                        f"isaac-lab-{self.profile}{self.suffix}:{container_path}/",
                        f"{host_path}",
                    ],
                    check=False,
                )
            print("\n[INFO] Finished copying the artifacts from the container.")
        else:
            raise RuntimeError(f"The container '{self.container_name}' is not running.")

    def config(self, output_yaml: Path | None = None):
        """Process the Docker compose configuration based on the passed yamls and environment files.

        If the :attr:`output_yaml` is not None, the configuration is written to the file. Otherwise, it is printed to
        the terminal.

        Args:
            output_yaml: The path to the yaml file where the configuration is written to. Defaults
                to None, in which case the configuration is printed to the terminal.
        """
        print("[INFO] Configuring the passed options into a yaml...\n")


        if output_yaml is not None:
            output = ["--output", output_yaml]
        else:
            output = []


        subprocess.run(
            ["docker", "compose"] + self.add_yamls + self.add_profiles + self.add_env_files + ["config"] + output,
            check=False,
            cwd=self.context_dir,
            env=self.environ,
        )

    """
    Helper functions.
    """

    def _resolve_image_extension(self, yamls: list[str] | None = None, envs: list[str] | None = None):
        """
        Resolve the image extension by setting up YAML files, profiles, and environment files for the Docker compose command.

        Args:
            yamls: A list of yaml files to extend ``docker-compose.yaml`` settings. These are extended in the order
                they are provided.
            envs: A list of environment variable files to extend the ``.env.base`` file. These are extended in the order
                they are provided.
        """
        self.add_yamls = ["--file", "docker-compose.yaml"]
        self.add_profiles = ["--profile", f"{self.profile}"]
        self.add_env_files = ["--env-file", ".env.base"]


        if self.profile != "base":
            self.add_env_files += ["--env-file", f".env.{self.profile}"]


        if envs is not None:
            for env in envs:
                self.add_env_files += ["--env-file", env]


        if yamls is not None:
            for yaml in yamls:
                self.add_yamls += ["--file", yaml]

    def _parse_dot_vars(self):
        """Parse the environment variables from the .env files.

        Based on the passed ".env" files, this function reads the environment variables and stores them in a dictionary.
        The environment variables are read in order and overwritten if there are name conflicts, mimicking the behavior
        of Docker compose.
        """
        self.dot_vars: dict[str, Any] = {}


        if len(self.add_env_files) % 2 != 0:
            raise RuntimeError(
                "The parameters for env files are configured incorrectly. There should be an even number of arguments."
                f" Received: {self.add_env_files}."
            )


        for i in range(1, len(self.add_env_files), 2):
            with open(self.context_dir / self.add_env_files[i]) as f:
                self.dot_vars.update(dict(line.strip().split("=", 1) for line in f if "=" in line))
