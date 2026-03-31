




"""Installation script for the 'isaaclab' python package."""

import os
import toml

from setuptools import setup


EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))

EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))


INSTALL_REQUIRES = [

    "numpy<2",
    "torch>=2.7",
    "onnx>=1.18.0",
    "prettytable==3.3.0",
    "toml",

    "hidapi==0.14.0.post2",

    "gymnasium==1.2.1",

    "trimesh",
    "pyglet<2",

    "transformers",
    "einops",
    "warp-lang",

    "pillow==11.3.0",

    "starlette==0.45.3",

    "pytest",
    "pytest-mock",
    "junitparser",
    "flatdict==4.0.1",
    "flaky",
]


SUPPORTED_ARCHS_ARM = "platform_machine in 'x86_64,AMD64,aarch64,arm64'"
SUPPORTED_ARCHS = "platform_machine in 'x86_64,AMD64'"
INSTALL_REQUIRES += [

    f"pin-pink==3.1.0 ; platform_system == 'Linux' and ({SUPPORTED_ARCHS_ARM})",

    f"dex-retargeting==0.4.6 ; platform_system == 'Linux' and ({SUPPORTED_ARCHS})",
]

PYTORCH_INDEX_URL = ["https://download.pytorch.org/whl/cu128"]


setup(
    name="isaaclab",
    author="Isaac Lab Project Developers",
    maintainer="Isaac Lab Project Developers",
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    license="BSD-3-Clause",
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    dependency_links=PYTORCH_INDEX_URL,
    packages=["isaaclab"],
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
