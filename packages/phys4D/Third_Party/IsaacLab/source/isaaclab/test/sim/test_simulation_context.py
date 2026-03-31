




"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher


simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import numpy as np

import isaacsim.core.utils.prims as prim_utils
import pytest
from isaacsim.core.api.simulation_context import SimulationContext as IsaacSimulationContext

from isaaclab.sim import SimulationCfg, SimulationContext


@pytest.fixture(autouse=True)
def test_setup_teardown():
    """Setup and teardown for each test."""

    SimulationContext.clear_instance()


    yield


    SimulationContext.clear_instance()


@pytest.mark.isaacsim_ci
def test_singleton():
    """Tests that the singleton is working."""
    sim1 = SimulationContext()
    sim2 = SimulationContext()
    sim3 = IsaacSimulationContext()
    assert sim1 is sim2
    assert sim1 is sim3


    sim2.clear_instance()
    assert sim1.instance() is None

    sim4 = SimulationContext()
    assert sim1 is not sim4
    assert sim3 is not sim4
    assert sim1.instance() is sim4.instance()
    assert sim3.instance() is sim4.instance()

    sim3.clear_instance()


@pytest.mark.isaacsim_ci
def test_initialization():
    """Test the simulation config."""
    cfg = SimulationCfg(physics_prim_path="/Physics/PhysX", render_interval=5, gravity=(0.0, -0.5, -0.5))
    sim = SimulationContext(cfg)





    assert sim.get_physics_dt() == cfg.dt
    assert sim.get_rendering_dt() == cfg.dt * cfg.render_interval
    assert not sim.has_rtx_sensors()

    assert prim_utils.is_prim_path_valid("/Physics/PhysX")
    assert prim_utils.is_prim_path_valid("/Physics/PhysX/defaultMaterial")

    gravity_dir, gravity_mag = sim.get_physics_context().get_gravity()
    gravity = np.array(gravity_dir) * gravity_mag
    np.testing.assert_almost_equal(gravity, cfg.gravity)


@pytest.mark.isaacsim_ci
def test_sim_version():
    """Test obtaining the version."""
    sim = SimulationContext()
    version = sim.get_version()
    assert len(version) > 0
    assert version[0] >= 4


@pytest.mark.isaacsim_ci
def test_carb_setting():
    """Test setting carb settings."""
    sim = SimulationContext()

    sim.set_setting("/physics/physxDispatcher", False)
    assert sim.get_setting("/physics/physxDispatcher") is False

    sim.set_setting("/myExt/using_omniverse_version", sim.get_version())
    assert tuple(sim.get_setting("/myExt/using_omniverse_version")) == tuple(sim.get_version())


@pytest.mark.isaacsim_ci
def test_headless_mode():
    """Test that render mode is headless since we are running in headless mode."""
    sim = SimulationContext()

    assert sim.render_mode == sim.RenderMode.NO_GUI_OR_RENDERING































@pytest.mark.isaacsim_ci
def test_zero_gravity():
    """Test that gravity can be properly disabled."""
    cfg = SimulationCfg(gravity=(0.0, 0.0, 0.0))

    sim = SimulationContext(cfg)

    gravity_dir, gravity_mag = sim.get_physics_context().get_gravity()
    gravity = np.array(gravity_dir) * gravity_mag
    np.testing.assert_almost_equal(gravity, cfg.gravity)
