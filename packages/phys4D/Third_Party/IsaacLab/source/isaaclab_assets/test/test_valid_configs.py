







"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher


app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


"""Rest everything follows."""


import pytest

import isaaclab_assets as lab_assets

from isaaclab.assets import AssetBase, AssetBaseCfg
from isaaclab.sim import build_simulation_context


@pytest.fixture(scope="module")
def registered_entities():

    registered_entities: dict[str, AssetBaseCfg] = {}

    for obj_name in dir(lab_assets):
        obj = getattr(lab_assets, obj_name)

        if isinstance(obj, AssetBaseCfg):
            registered_entities[obj_name] = obj

    print(">>> All registered entities:", list(registered_entities.keys()))
    return registered_entities



@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_asset_configs(registered_entities, device):
    """Check all registered asset configurations."""

    for asset_name, entity_cfg in registered_entities.items():

        with build_simulation_context(device=device, auto_add_lighting=True) as sim:
            sim._app_control_on_stop_handle = None

            print(f">>> Testing entity {asset_name} on device {device}")

            entity_cfg.prim_path = "/World/asset"

            entity: AssetBase = entity_cfg.class_type(entity_cfg)


            sim.reset()


            assert entity.is_initialized
