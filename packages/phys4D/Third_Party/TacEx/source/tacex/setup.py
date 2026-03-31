"""Installation script for the 'tacex' python package."""

import os
import toml

from setuptools import setup


EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))

EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))


INSTALL_REQUIRES = [

    "debugpy",










    (
        "torch_scatter @"
        "https://data.pyg.org/whl/torch-2.8.0%2Bcu128/torch_scatter-2.1.2%2Bpt28cu128-cp311-cp311-linux_x86_64.whl"
    ),
    "psutil",
    "nvidia-ml-py",
    "pre-commit",
]




setup(
    name="tacex",
    packages=["tacex"],
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,

    license="MIT",
    include_package_data=True,
    python_requires=">=3.10",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Isaac Sim :: 2023.1.1",
        "Isaac Sim :: 4.5.0",
    ],
    zip_safe=False,
)
