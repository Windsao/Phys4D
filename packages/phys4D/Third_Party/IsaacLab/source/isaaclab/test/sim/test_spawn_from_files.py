




from isaaclab.app import AppLauncher

"""Launch Isaac Sim Simulator first."""


simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
import pytest
from isaacsim.core.api.simulation_context import SimulationContext
from isaacsim.core.utils.extensions import enable_extension, get_extension_path_from_name

import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


@pytest.fixture
def sim():
    """Create a blank new stage for each test."""

    stage_utils.create_new_stage()

    dt = 0.1

    sim = SimulationContext(physics_dt=dt, rendering_dt=dt, backend="numpy")

    stage_utils.update_stage()

    yield sim


    sim.stop()
    sim.clear()
    sim.clear_all_callbacks()
    sim.clear_instance()


@pytest.mark.isaacsim_ci
def test_spawn_usd(sim):
    """Test loading prim from Usd file."""

    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd")
    prim = cfg.func("/World/Franka", cfg)

    assert prim.IsValid()
    assert prim_utils.is_prim_path_valid("/World/Franka")
    assert prim.GetPrimTypeInfo().GetTypeName() == "Xform"


@pytest.mark.isaacsim_ci
def test_spawn_usd_fails(sim):
    """Test loading prim from Usd file fails when asset usd path is invalid."""

    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda2_instanceable.usd")

    with pytest.raises(FileNotFoundError):
        cfg.func("/World/Franka", cfg)


@pytest.mark.isaacsim_ci
def test_spawn_urdf(sim):
    """Test loading prim from URDF file."""

    enable_extension("isaacsim.asset.importer.urdf")
    extension_path = get_extension_path_from_name("isaacsim.asset.importer.urdf")

    cfg = sim_utils.UrdfFileCfg(
        asset_path=f"{extension_path}/data/urdf/robots/franka_description/robots/panda_arm_hand.urdf",
        fix_base=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
        ),
    )
    prim = cfg.func("/World/Franka", cfg)

    assert prim.IsValid()
    assert prim_utils.is_prim_path_valid("/World/Franka")
    assert prim.GetPrimTypeInfo().GetTypeName() == "Xform"


@pytest.mark.isaacsim_ci
def test_spawn_ground_plane(sim):
    """Test loading prim for the ground plane from grid world USD."""

    cfg = sim_utils.GroundPlaneCfg(color=(0.1, 0.1, 0.1), size=(10.0, 10.0))
    prim = cfg.func("/World/ground_plane", cfg)

    assert prim.IsValid()
    assert prim_utils.is_prim_path_valid("/World/ground_plane")
    assert prim.GetPrimTypeInfo().GetTypeName() == "Xform"
