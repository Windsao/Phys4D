from magicsim import MAGICSIM_ASSETS, MAGICSIM_HOME
from magicsim import MAGICSIM_CONF
from magicsim.Env.Environment.BaseEnv import NVIDIA_ASSETS
from isaacsim.storage.native import get_assets_root_path
from omegaconf import DictConfig, OmegaConf
import os
import torch
from typing import List
from magicsim.Env.Scene.Object.Primitives import PRIMITIVE_MAP


def resolve_path(config_path: str) -> str:
    """Resolve paths from configuration by replacing placeholders with actual directory paths.

    Replaces special path placeholders with their corresponding actual directory paths
    and ensures the result is an absolute path.

    Args:
        config_path: Path string from configuration that may contain placeholders

    Returns:
        Resolved absolute path with placeholders replaced by actual directory paths

    Notes:
        Resolution rules:
        - If path contains `$MAGICSIM_ASSETS`, replace with value of MAGICSIM_ASSETS variable
        - If path contains `$NVIDIA_ASSETS`, replace with value of NVIDIA_ASSETS variable
        - If path is absolute (starts with `/`), return as-is without replacement
    """

    path_mapping = {
        "$MAGICSIM_ASSETS": MAGICSIM_ASSETS,
        "$MAGICSIM_CONF": MAGICSIM_CONF,
        "$NVIDIA_ASSETS": NVIDIA_ASSETS,
        "$MAGICSIM_HOME": MAGICSIM_HOME,
    }

    if os.path.isabs(config_path):
        return config_path

    resolved_path = config_path
    for placeholder, actual_path in path_mapping.items():
        if placeholder in resolved_path:
            resolved_path = resolved_path.replace(placeholder, actual_path)

    return resolved_path


def get_usd_paths_from_folder(
    folder_path: str,
    recursive: bool = True,
    usd_paths: list[str] = None,
    skip_keywords: list[str] = None,
) -> list[str]:
    """Retrieve USD file paths from a folder, optionally searching recursively and filtering by keywords."""
    if usd_paths is None:
        usd_paths = []
    skip_keywords = skip_keywords or []

    import omni.kit.app

    ext_manager = omni.kit.app.get_app().get_extension_manager()
    if not ext_manager.is_extension_enabled("omni.client"):
        ext_manager.set_extension_enabled_immediate("omni.client", True)
    import omni.client

    result, entries = omni.client.list(folder_path)
    if result != omni.client.Result.OK:
        print(f"Could list assets in path: {folder_path}")
        return usd_paths

    for entry in entries:
        if any(
            keyword.lower() in entry.relative_path.lower() for keyword in skip_keywords
        ):
            continue
        _, ext = os.path.splitext(entry.relative_path)
        if ext in [".usd", ".usda", ".usdc"]:
            path_posix = os.path.join(folder_path, entry.relative_path).replace(
                "\\", "/"
            )
            usd_paths.append(path_posix)
        elif recursive and entry.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN:
            sub_folder = os.path.join(folder_path, entry.relative_path).replace(
                "\\", "/"
            )
            get_usd_paths_from_folder(
                sub_folder,
                recursive=recursive,
                usd_paths=usd_paths,
                skip_keywords=skip_keywords,
            )

    return usd_paths


def get_mdl_paths_from_folder(
    folder_path: str,
    recursive: bool = True,
    mdl_paths: list[str] = None,
    skip_keywords: list[str] = None,
) -> list[str]:
    """Retrieve MDL file paths from a folder, optionally searching recursively and filtering by keywords."""
    if mdl_paths is None:
        mdl_paths = []
    skip_keywords = skip_keywords or []

    import omni.kit.app

    ext_manager = omni.kit.app.get_app().get_extension_manager()
    if not ext_manager.is_extension_enabled("omni.client"):
        ext_manager.set_extension_enabled_immediate("omni.client", True)
    import omni.client

    result, entries = omni.client.list(folder_path)
    if result != omni.client.Result.OK:
        print(f"Could list assets in path: {folder_path}")
        return mdl_paths

    for entry in entries:
        if any(
            keyword.lower() in entry.relative_path.lower() for keyword in skip_keywords
        ):
            continue
        _, ext = os.path.splitext(entry.relative_path)
        if ext == ".mdl":
            path_posix = os.path.join(folder_path, entry.relative_path).replace(
                "\\", "/"
            )
            mdl_paths.append(path_posix)
        elif recursive and entry.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN:
            sub_folder = os.path.join(folder_path, entry.relative_path).replace(
                "\\", "/"
            )
            get_mdl_paths_from_folder(
                sub_folder,
                recursive=recursive,
                mdl_paths=mdl_paths,
                skip_keywords=skip_keywords,
            )

    return mdl_paths


def resolve_mdl_paths(folder_path: str) -> list[str]:
    """Resolve MDL file paths from a folder path.

    This function resolves path placeholders (e.g., $MAGICSIM_ASSETS) and then
    retrieves all MDL files from the resolved folder path.

    Args:
        folder_path: Folder path that may contain placeholders (e.g., $MAGICSIM_ASSETS/...)

    Returns:
        List of resolved MDL file paths
    """
    resolved_folder_path = resolve_path(folder_path)
    return get_mdl_paths_from_folder(resolved_folder_path, recursive=True)


def get_usd_paths(
    files: list[str] = None,
    folders: list[str] = None,
    skip_folder_keywords: list[str] = None,
) -> list[str]:
    """Retrieve USD paths from specified files and folders, optionally filtering out specific folder keywords."""
    files = files or []
    folders = folders or []
    skip_folder_keywords = skip_folder_keywords or []

    assets_root_path = get_assets_root_path()
    env_paths = []

    for file_path in files:
        file_path = (
            file_path
            if file_path.startswith(("omniverse://", "http://", "https://", "file://"))
            else assets_root_path + file_path
        )
        env_paths.append(file_path)

    for folder_path in folders:
        folder_path = (
            folder_path
            if folder_path.startswith(
                ("omniverse://", "http://", "https://", "file://")
            )
            else assets_root_path + folder_path
        )
        env_paths.extend(
            get_usd_paths_from_folder(
                folder_path, recursive=True, skip_keywords=skip_folder_keywords
            )
        )

    return env_paths


def deep_resolve_paths(cfg: DictConfig):
    """
    Deeply resolve path placeholders (with "$") in an OmegaConf DictConfig to real accessible paths.

    This function recursively traverses the configuration dictionary and resolves any string values
    containing "$" placeholders to their actual file system paths.

    Args:
        cfg: OmegaConf DictConfig to resolve (e.g., room/object configs like room_top_cfg, cat_spec)
            The configuration is modified in-place.
    """
    for key, value in cfg.items():
        if isinstance(value, DictConfig):
            deep_resolve_paths(value)
        elif isinstance(value, str) and "$" in value:
            OmegaConf.set_struct(cfg, False)
            cfg[key] = resolve_path(value)
            OmegaConf.set_struct(cfg, True)


def _resolve_asset_paths(asset_list: List[str]) -> List[str]:
    """Resolve mixed asset list containing USD paths and primitive names.

    Args:
        asset_list: List of asset sources (USD paths, folders, or primitive names)

    Returns:
        List of resolved asset sources (USD file paths and primitive names)
    """
    all_asset_sources = []

    for item in asset_list:
        if item in PRIMITIVE_MAP:
            all_asset_sources.append(item)
        else:
            resolved_path = resolve_path(item)

            if os.path.isdir(resolved_path):
                all_asset_sources.extend(
                    get_usd_paths_from_folder(
                        resolved_path, skip_keywords=[".thumbs"], recursive=True
                    )
                )
            elif os.path.isfile(resolved_path) and resolved_path.endswith(
                (".usd", ".usda", ".usdc")
            ):
                all_asset_sources.append(os.path.abspath(resolved_path))
            else:
                all_asset_sources.extend(
                    get_usd_paths_from_folder(
                        item, skip_keywords=[".thumbs"], recursive=False
                    )
                )

    return all_asset_sources


def _select_assets(
    all_assets: List[str], num_per_env: int, random_flag: bool, device
) -> List[str]:
    """Select assets from the available list based on selection criteria.

    Args:
        all_assets: List of all available assets
        num_per_env: Number of assets to select per environment
        random_flag: Whether to select randomly or sequentially

    Returns:
        List of selected assets
    """
    if random_flag:
        indices = torch.randint(
            0, len(all_assets), (num_per_env,), device=device
        ).tolist()
        return [all_assets[i] for i in indices]
    else:
        return [all_assets[i % len(all_assets)] for i in range(num_per_env)]
