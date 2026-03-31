




"""Utilities for file I/O with yaml."""

import os
import yaml

from isaaclab.utils import class_to_dict


def load_yaml(filename: str) -> dict:
    """Loads an input PKL file safely.

    Args:
        filename: The path to pickled file.

    Raises:
        FileNotFoundError: When the specified file does not exist.

    Returns:
        The data read from the input file.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    with open(filename) as f:
        data = yaml.full_load(f)
    return data


def dump_yaml(filename: str, data: dict | object, sort_keys: bool = False):
    """Saves data into a YAML file safely.

    Note:
        The function creates any missing directory along the file's path.

    Args:
        filename: The path to save the file at.
        data: The data to save either a dictionary or class object.
        sort_keys: Whether to sort the keys in the output file. Defaults to False.
    """

    if not filename.endswith("yaml"):
        filename += ".yaml"

    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    if not isinstance(data, dict):
        data = class_to_dict(data)

    with open(filename, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=sort_keys)
