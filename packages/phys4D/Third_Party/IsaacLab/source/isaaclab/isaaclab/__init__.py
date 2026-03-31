




"""Package containing the core framework."""

import os
import toml


ISAACLAB_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""

ISAACLAB_METADATA = toml.load(os.path.join(ISAACLAB_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""


__version__ = ISAACLAB_METADATA["package"]["version"]
