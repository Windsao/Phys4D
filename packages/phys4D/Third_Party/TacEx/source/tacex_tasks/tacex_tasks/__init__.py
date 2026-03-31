




"""Package containing task implementations for various robotic environments."""

import os
import toml


ISAACLAB_TASKS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""

ISAACLAB_TASKS_METADATA = toml.load(os.path.join(ISAACLAB_TASKS_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""


__version__ = ISAACLAB_TASKS_METADATA["package"]["version"]





from isaaclab_tasks.utils import import_packages


_BLACKLIST_PKGS = ["utils", ".mdp"]

import_packages(__name__, _BLACKLIST_PKGS)
