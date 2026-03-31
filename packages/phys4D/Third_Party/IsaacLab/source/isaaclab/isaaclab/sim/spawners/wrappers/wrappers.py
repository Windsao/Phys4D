




from __future__ import annotations

import random
import re
from typing import TYPE_CHECKING

import carb
import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
from isaacsim.core.utils.stage import get_current_stage
from pxr import Sdf, Usd

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files import UsdFileCfg

if TYPE_CHECKING:
    from . import wrappers_cfg


def spawn_multi_asset(
    prim_path: str,
    cfg: wrappers_cfg.MultiAssetSpawnerCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    clone_in_fabric: bool = False,
    replicate_physics: bool = False,
) -> Usd.Prim:
    """Spawn multiple assets based on the provided configurations.

    This function spawns multiple assets based on the provided configurations. The assets are spawned
    in the order they are provided in the list. If the :attr:`~MultiAssetSpawnerCfg.random_choice` parameter is
    set to True, a random asset configuration is selected for each spawn.

    Args:
        prim_path: The prim path to spawn the assets.
        cfg: The configuration for spawning the assets.
        translation: The translation of the spawned assets. Default is None.
        orientation: The orientation of the spawned assets in (w, x, y, z) order. Default is None.
        clone_in_fabric: Whether to clone in fabric. Default is False.
        replicate_physics: Whether to replicate physics. Default is False.

    Returns:
        The created prim at the first prim path.
    """

    stage = get_current_stage()



    root_path, asset_path = prim_path.rsplit("/", 1)


    is_regex_expression = re.match(r"^[a-zA-Z0-9/_]+$", root_path) is None


    if is_regex_expression and root_path != "":
        source_prim_paths = sim_utils.find_matching_prim_paths(root_path)

        if len(source_prim_paths) == 0:
            raise RuntimeError(
                f"Unable to find source prim path: '{root_path}'. Please create the prim before spawning."
            )
    else:
        source_prim_paths = [root_path]


    template_prim_path = stage_utils.get_next_free_path("/World/Template")
    prim_utils.create_prim(template_prim_path, "Scope")


    proto_prim_paths = list()
    for index, asset_cfg in enumerate(cfg.assets_cfg):

        if cfg.semantic_tags is not None:
            if asset_cfg.semantic_tags is None:
                asset_cfg.semantic_tags = cfg.semantic_tags
            else:
                asset_cfg.semantic_tags += cfg.semantic_tags

        attr_names = ["mass_props", "rigid_props", "collision_props", "activate_contact_sensors", "deformable_props"]
        for attr_name in attr_names:
            attr_value = getattr(cfg, attr_name)
            if hasattr(asset_cfg, attr_name) and attr_value is not None:
                setattr(asset_cfg, attr_name, attr_value)

        proto_prim_path = f"{template_prim_path}/Asset_{index:04d}"
        asset_cfg.func(
            proto_prim_path,
            asset_cfg,
            translation=translation,
            orientation=orientation,
            clone_in_fabric=clone_in_fabric,
            replicate_physics=replicate_physics,
        )

        proto_prim_paths.append(proto_prim_path)


    prim_paths = [f"{source_prim_path}/{asset_path}" for source_prim_path in source_prim_paths]




    with Sdf.ChangeBlock():
        for index, prim_path in enumerate(prim_paths):

            env_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)

            if cfg.random_choice:
                proto_path = random.choice(proto_prim_paths)
            else:
                proto_path = proto_prim_paths[index % len(proto_prim_paths)]

            Sdf.CopySpec(env_spec.layer, Sdf.Path(proto_path), env_spec.layer, Sdf.Path(prim_path))


    prim_utils.delete_prim(template_prim_path)




    carb_settings_iface = carb.settings.get_settings()
    carb_settings_iface.set_bool("/isaaclab/spawn/multi_assets", True)


    return prim_utils.get_prim_at_path(prim_paths[0])


def spawn_multi_usd_file(
    prim_path: str,
    cfg: wrappers_cfg.MultiUsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    clone_in_fabric: bool = False,
    replicate_physics: bool = False,
) -> Usd.Prim:
    """Spawn multiple USD files based on the provided configurations.

    This function creates configuration instances corresponding the individual USD files and
    calls the :meth:`spawn_multi_asset` method to spawn them into the scene.

    Args:
        prim_path: The prim path to spawn the assets.
        cfg: The configuration for spawning the assets.
        translation: The translation of the spawned assets. Default is None.
        orientation: The orientation of the spawned assets in (w, x, y, z) order. Default is None.
        clone_in_fabric: Whether to clone in fabric. Default is False.
        replicate_physics: Whether to replicate physics. Default is False.

    Returns:
        The created prim at the first prim path.
    """

    from .wrappers_cfg import MultiAssetSpawnerCfg


    if isinstance(cfg.usd_path, str):
        usd_paths = [cfg.usd_path]
    else:
        usd_paths = cfg.usd_path


    usd_template_cfg = UsdFileCfg()
    for attr_name, attr_value in cfg.__dict__.items():

        if attr_name in ["func", "usd_path", "random_choice"]:
            continue

        setattr(usd_template_cfg, attr_name, attr_value)


    multi_asset_cfg = MultiAssetSpawnerCfg(assets_cfg=[])
    for usd_path in usd_paths:
        usd_cfg = usd_template_cfg.replace(usd_path=usd_path)
        multi_asset_cfg.assets_cfg.append(usd_cfg)

    multi_asset_cfg.random_choice = cfg.random_choice






    if hasattr(cfg, "activate_contact_sensors"):
        multi_asset_cfg.activate_contact_sensors = cfg.activate_contact_sensors


    return spawn_multi_asset(prim_path, multi_asset_cfg, translation, orientation, clone_in_fabric, replicate_physics)
