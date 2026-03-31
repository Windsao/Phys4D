




"""Installation script for the 'isaaclab_mimic' python package."""

import itertools
import os
import platform
import toml

from setuptools import setup


EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))

EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))


INSTALL_REQUIRES = [
    "tomli",

    "ipywidgets==8.1.5",
]


EXTRAS_REQUIRE = {"robomimic": []}


if platform.system() == "Linux":
    EXTRAS_REQUIRE["robomimic"].append("robomimic@git+https://github.com/ARISE-Initiative/robomimic.git@v0.4.0")


EXTRAS_REQUIRE["all"] = list(itertools.chain.from_iterable(EXTRAS_REQUIRE.values()))

EXTRAS_REQUIRE["all"] = list(set(EXTRAS_REQUIRE["all"]))


setup(
    name="isaaclab_mimic",
    packages=["isaaclab_mimic"],
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    license="Apache-2.0",
    include_package_data=True,
    python_requires=">=3.10",
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
