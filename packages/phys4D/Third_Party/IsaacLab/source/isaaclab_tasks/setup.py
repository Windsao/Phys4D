




"""Installation script for the 'isaaclab_tasks' python package."""

import os
import toml

from setuptools import setup


EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))

EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))


INSTALL_REQUIRES = [

    "numpy<2",
    "torch>=2.7",
    "torchvision>=0.14.1",
    "protobuf>=4.25.8,!=5.26.0",

    "tensorboard",
    "numba",
]

PYTORCH_INDEX_URL = ["https://download.pytorch.org/whl/cu128"]


setup(
    name="isaaclab_tasks",
    author="Isaac Lab Project Developers",
    maintainer="Isaac Lab Project Developers",
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    dependency_links=PYTORCH_INDEX_URL,
    packages=["isaaclab_tasks"],
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Isaac Sim :: 4.5.0",
        "Isaac Sim :: 5.0.0",
        "Isaac Sim :: 5.1.0",
    ],
    zip_safe=False,
)
