




"""Installation script for the 'isaaclab_rl' python package."""

import itertools
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

    "hydra-core",

    "h5py",

    "tensorboard",

    "moviepy",

    "pillow==11.3.0",
    "packaging<24",
]

PYTORCH_INDEX_URL = ["https://download.pytorch.org/whl/cu128"]


EXTRAS_REQUIRE = {
    "sb3": ["stable-baselines3>=2.6", "tqdm", "rich"],
    "skrl": ["skrl>=1.4.3"],
    "rl-games": [
        "rl-games @ git+https://github.com/isaac-sim/rl_games.git@python3.11",
        "gym",
    ],
    "rsl-rl": ["rsl-rl-lib==3.0.1", "onnxscript>=0.5"],
}

EXTRAS_REQUIRE["rl_games"] = EXTRAS_REQUIRE["rl-games"]
EXTRAS_REQUIRE["rsl_rl"] = EXTRAS_REQUIRE["rsl-rl"]


EXTRAS_REQUIRE["all"] = list(itertools.chain.from_iterable(EXTRAS_REQUIRE.values()))

EXTRAS_REQUIRE["all"] = list(set(EXTRAS_REQUIRE["all"]))


setup(
    name="isaaclab_rl",
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
    extras_require=EXTRAS_REQUIRE,
    packages=["isaaclab_rl"],
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
