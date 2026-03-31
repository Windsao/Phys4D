




















import os
import sys

sys.path.insert(0, os.path.abspath("../source/tacex"))
sys.path.insert(0, os.path.abspath("../source/tacex/tacex"))
sys.path.insert(0, os.path.abspath("../source/tacex_assets/tacex_assets"))
sys.path.insert(0, os.path.abspath("../source/tacex_tasks/tacex_tasks"))
sys.path.insert(0, os.path.abspath("../source/tacex_uipc/tacex_uipc"))



project = "TacEx"
copyright = "2025, TacEx Developers"
author = "TacEx Developers"


with open(os.path.join(os.path.dirname(__file__), "../..", "VERSION")) as f:
    full_version = f.read().strip()
    version = ".".join(full_version.split(".")[:3])






extensions = [
    "autodocsumm",
    "myst_parser",
    "sphinx.ext.napoleon",
    "sphinxemoji.sphinxemoji",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.icon",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_tabs.tabs",
    "sphinx_multiversion",
]


mathjax3_config = {
    "tex": {
        "inlineMath": [["\\(", "\\)"]],
        "displayMath": [["\\[", "\\]"]],
    },
}


panels_add_bootstrap_css = False
panels_add_fontawesome_css = True


source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}






autodoc_typehints = "signature"


autoclass_content = "class"

autodoc_class_signature = "separated"

autodoc_member_order = "bysource"

autodoc_inherit_docstrings = True

bibtex_bibfiles = ["./_static/refs.bib"]

autosummary_generate = True
autosummary_generate_overwrite = False

autodoc_default_options = {
    "autosummary": True,
}


intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "isaac": ("https://docs.omniverse.nvidia.com/py/isaacsim", None),
    "gymnasium": ("https://gymnasium.farama.org/", None),
    "warp": ("https://nvidia.github.io/warp/", None),
    "dev-guide": ("https://docs.omniverse.nvidia.com/dev-guide/latest", None),
}


templates_path = []




exclude_patterns = ["_build", "_redirect", "_templates", "Thumbs.db", ".DS_Store", "README.md", "licenses/*"]


autodoc_mock_imports = [
    "torch",
    "torchvision",
    "numpy",
    "matplotlib",
    "scipy",
    "carb",
    "warp",
    "pxr",
    "isaacsim",
    "omni",
    "omni.kit",
    "omni.log",
    "omni.usd",
    "omni.client",
    "omni.physx",
    "omni.physics",
    "pxr.PhysxSchema",
    "pxr.PhysicsSchemaTools",
    "omni.replicator",
    "omni.isaac.core",
    "omni.isaac.kit",
    "omni.isaac.cloner",
    "omni.isaac.urdf",
    "omni.isaac.version",
    "omni.isaac.motion_generation",
    "omni.isaac.ui",
    "isaacsim",
    "isaacsim.core.api",
    "isaacsim.core.cloner",
    "isaacsim.core.version",
    "isaacsim.robot_motion.motion_generation",
    "isaacsim.gui.components",
    "isaacsim.asset.importer.urdf",
    "isaacsim.asset.importer.mjcf",
    "omni.syntheticdata",
    "omni.timeline",
    "omni.ui",
    "gym",
    "skrl",
    "stable_baselines3",
    "rsl_rl",
    "rl_games",
    "ray",
    "h5py",
    "hid",
    "prettytable",
    "tqdm",
    "tensordict",
    "trimesh",
    "toml",
    "pink",
    "pinocchio",
    "nvidia.srl",
    "flatdict",
    "IPython",
    "ipywidgets",
    "mpl_toolkits",
]



suppress_warnings = [















    "ref.python",
]




language = "en"



import sphinx_book_theme

html_title = "TacEx Documentation"
html_theme_path = [sphinx_book_theme.get_html_theme_path()]
html_theme = "sphinx_book_theme"

html_show_copyright = True
html_show_sphinx = False
html_last_updated_fmt = ""







html_theme_options = {
    "path_to_docs": "docs/",
    "collapse_navigation": True,
    "repository_url": "https://github.com/DH-Ng/TacEx",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "show_toc_level": 1,
    "use_sidenotes": True,





    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/DH-Ng/TacEx",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },












    ],
    "icon_links_label": "Quick Links",
}

templates_path = [
    "_templates",
]


smv_remote_whitelist = r"^.*$"

smv_branch_whitelist = os.getenv("SMV_BRANCH_WHITELIST", r"^(main|devel)$")

smv_tag_whitelist = os.getenv("SMV_TAG_WHITELIST", r"^v[1-9]\d*\.\d+\.\d+$")
html_sidebars = {"**": ["navbar-logo.html", "icon-links.html", "search-field.html", "sbt-sidebar-nav.html"]}





def skip_member(app, what, name, obj, skip, options):

    exclusions = ["from_dict", "to_dict", "replace", "copy", "validate", "__post_init__"]
    if name in exclusions:
        return True
    return None


def setup(app):
    app.connect("autodoc-skip-member", skip_member)
