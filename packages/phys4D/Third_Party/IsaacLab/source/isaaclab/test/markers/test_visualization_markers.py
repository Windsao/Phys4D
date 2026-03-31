




"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher


simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import torch

import isaacsim.core.utils.stage as stage_utils
import pytest
from isaacsim.core.api.simulation_context import SimulationContext

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG, POSITION_GOAL_MARKER_CFG
from isaaclab.utils.math import random_orientation
from isaaclab.utils.timer import Timer


@pytest.fixture
def sim():
    """Create a blank new stage for each test."""

    dt = 0.01

    stage_utils.create_new_stage()

    sim_context = SimulationContext(physics_dt=dt, rendering_dt=dt, backend="torch", device="cuda:0")
    yield sim_context

    sim_context.stop()
    sim_context.clear_instance()
    stage_utils.close_stage()


def test_instantiation(sim):
    """Test that the class can be initialized properly."""
    config = VisualizationMarkersCfg(
        prim_path="/World/Visuals/test",
        markers={
            "test": sim_utils.SphereCfg(radius=1.0),
        },
    )
    test_marker = VisualizationMarkers(config)
    print(test_marker)

    assert test_marker.num_prototypes == 1


def test_usd_marker(sim):
    """Test with marker from a USD."""

    config = FRAME_MARKER_CFG.copy()
    config.prim_path = "/World/Visuals/test_frames"
    test_marker = VisualizationMarkers(config)


    sim.reset()

    num_frames = 0

    for count in range(1000):

        if count % 50 == 0:
            num_frames = torch.randint(10, 1000, (1,)).item()
            frame_translations = torch.randn(num_frames, 3, device=sim.device)
            frame_rotations = random_orientation(num_frames, device=sim.device)

            test_marker.visualize(translations=frame_translations, orientations=frame_rotations)

        sim.step()

        assert test_marker.count == num_frames


def test_usd_marker_color(sim):
    """Test with marker from a USD with its color modified."""

    config = FRAME_MARKER_CFG.copy()
    config.prim_path = "/World/Visuals/test_frames"
    config.markers["frame"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))
    test_marker = VisualizationMarkers(config)


    sim.reset()

    for count in range(1000):

        if count % 50 == 0:
            num_frames = torch.randint(10, 1000, (1,)).item()
            frame_translations = torch.randn(num_frames, 3, device=sim.device)
            frame_rotations = random_orientation(num_frames, device=sim.device)

            test_marker.visualize(translations=frame_translations, orientations=frame_rotations)

        sim.step()


def test_multiple_prototypes_marker(sim):
    """Test with multiple prototypes of spheres."""

    config = POSITION_GOAL_MARKER_CFG.copy()
    config.prim_path = "/World/Visuals/test_protos"
    test_marker = VisualizationMarkers(config)


    sim.reset()

    for count in range(1000):

        if count % 50 == 0:
            num_frames = torch.randint(100, 1000, (1,)).item()
            frame_translations = torch.randn(num_frames, 3, device=sim.device)

            marker_indices = torch.randint(0, test_marker.num_prototypes, (num_frames,), device=sim.device)

            test_marker.visualize(translations=frame_translations, marker_indices=marker_indices)

        sim.step()


def test_visualization_time_based_on_prototypes(sim):
    """Test with time taken when number of prototypes is increased."""

    config = POSITION_GOAL_MARKER_CFG.copy()
    config.prim_path = "/World/Visuals/test_protos"
    test_marker = VisualizationMarkers(config)


    sim.reset()

    num_frames = 4096


    assert test_marker.is_visible()

    frame_translations = torch.randn(num_frames, 3, device=sim.device)
    marker_indices = torch.randint(0, test_marker.num_prototypes, (num_frames,), device=sim.device)

    with Timer("Marker visualization with explicit indices") as timer:
        test_marker.visualize(translations=frame_translations, marker_indices=marker_indices)

        time_with_marker_indices = timer.time_elapsed

    with Timer("Marker visualization with no indices") as timer:
        test_marker.visualize(translations=frame_translations)

        time_with_no_marker_indices = timer.time_elapsed


    sim.step()

    assert time_with_no_marker_indices < time_with_marker_indices


def test_visualization_time_based_on_visibility(sim):
    """Test with visibility of markers. When invisible, the visualize call should return."""

    config = POSITION_GOAL_MARKER_CFG.copy()
    config.prim_path = "/World/Visuals/test_protos"
    test_marker = VisualizationMarkers(config)


    sim.reset()

    num_frames = 4096


    assert test_marker.is_visible()

    frame_translations = torch.randn(num_frames, 3, device=sim.device)
    marker_indices = torch.randint(0, test_marker.num_prototypes, (num_frames,), device=sim.device)

    with Timer("Marker visualization") as timer:
        test_marker.visualize(translations=frame_translations, marker_indices=marker_indices)

        time_with_visualization = timer.time_elapsed


    sim.step()

    test_marker.set_visibility(False)


    assert not test_marker.is_visible()

    frame_translations = torch.randn(num_frames, 3, device=sim.device)
    marker_indices = torch.randint(0, test_marker.num_prototypes, (num_frames,), device=sim.device)

    with Timer("Marker no visualization") as timer:
        test_marker.visualize(translations=frame_translations, marker_indices=marker_indices)

        time_with_no_visualization = timer.time_elapsed


    assert time_with_no_visualization < time_with_visualization
