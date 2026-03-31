



"""Package containing asset and sensor configurations."""

import os
import toml


TACEX_ASSETS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""


TACEX_ASSETS_DATA_DIR = os.path.join(TACEX_ASSETS_EXT_DIR, "tacex_assets/data")
"""Path to the extension data directory."""

TACEX_ASSETS_METADATA = toml.load(os.path.join(TACEX_ASSETS_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""


__version__ = TACEX_ASSETS_METADATA["package"]["version"]

from .robots import *
from .sensors import *
