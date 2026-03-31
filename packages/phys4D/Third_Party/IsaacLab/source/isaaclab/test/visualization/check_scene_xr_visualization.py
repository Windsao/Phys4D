




"""
This script checks if the XR visualization widgets are visible from the camera.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/isaaclab/test/visualization/check_scene_visualization.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Check XR visualization widgets in Isaac Lab.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()


args_cli.xr = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import time
from typing import Any

from pxr import Gf

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.ui.xr_widgets import DataCollector, TriggerType, VisualizationManager, XRVisualization, update_instruction
from isaaclab.utils import configclass






@configclass
class SimpleSceneCfg(InteractiveSceneCfg):
    """Design the scene with sensors on the robot."""


    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())


    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )


def get_camera_position():
    """Get the current camera position from the USD stage.

    Returns:
        tuple: (x, y, z) camera position or None if not available
    """
    try:
        import isaacsim.core.utils.stage as stage_utils
        from pxr import UsdGeom

        stage = stage_utils.get_current_stage()
        if stage is not None:

            camera_prim_path = "/OmniverseKit_Persp"
            camera_prim = stage.GetPrimAtPath(camera_prim_path)

            if camera_prim and camera_prim.IsValid():

                camera_xform = UsdGeom.Xformable(camera_prim)
                world_transform = camera_xform.ComputeLocalToWorldTransform(0)


                camera_pos = world_transform.ExtractTranslation()
                return (camera_pos[0], camera_pos[1], camera_pos[2])
        return None
    except Exception as e:
        print(f"[ERROR]: Failed to get camera position: {e}")
        return None


def _sample_handle_ik_error(mgr: VisualizationManager, data_collector: DataCollector, params: Any = None) -> None:
    error_text_color = getattr(mgr, "_error_text_color", 0xFF0000FF)
    mgr.display_widget(
        "IK Error Detected",
        "/ik_error",
        VisualizationManager.message_widget_preset()
        | {
            "text_color": error_text_color,
            "prim_path_source": "/World/defaultGroundPlane/GroundPlane",
            "translation": Gf.Vec3f(0, 0, 1),
        },
    )


def _sample_update_error_text_color(mgr: VisualizationManager, data_collector: DataCollector) -> None:
    current_color = getattr(mgr, "_error_text_color", 0xFF0000FF)
    new_color = current_color + 0x100
    if new_color >= 0xFFFFFFFF:
        new_color = 0xFF0000FF
    mgr.set_attr("_error_text_color", new_color)


def _sample_update_left_panel(mgr: VisualizationManager, data_collector: DataCollector) -> None:
    left_panel_id = getattr(mgr, "left_panel_id", None)

    if left_panel_id is None:
        return

    left_panel_created = getattr(mgr, "_left_panel_created", False)
    if left_panel_created is False:

        mgr.display_widget(
            "Left Panel",
            left_panel_id,
            VisualizationManager.panel_widget_preset()
            | {
                "text_color": 0xFFFFFFFF,
                "prim_path_source": "/World/defaultGroundPlane/GroundPlane",
                "translation": Gf.Vec3f(0, -3, 1),
            },
        )
        mgr.set_attr("_left_panel_created", True)

    updated_times = getattr(mgr, "_left_panel_updated_times", 0)

    content = f"Left Panel\nUpdated #{updated_times} times"
    update_instruction(left_panel_id, content)
    mgr.set_attr("_left_panel_updated_times", updated_times + 1)


def _sample_update_right_panel(mgr: VisualizationManager, data_collector: DataCollector) -> None:
    right_panel_id = getattr(mgr, "right_panel_id", None)

    if right_panel_id is None:
        return

    updated_times = getattr(mgr, "_right_panel_updated_times", 0)

    right_panel_data = data_collector.get_data("right_panel_data")
    if right_panel_data is not None:
        assert isinstance(right_panel_data, (tuple, list)), "Right panel data must be a tuple or list"

        formatted_data = tuple(f"{x:.3f}" for x in right_panel_data)
        content = f"Right Panel\nUpdated #{updated_times} times\nData: {formatted_data}"
    else:
        content = f"Right Panel\nUpdated #{updated_times} times\nData: None"

    right_panel_created = getattr(mgr, "_right_panel_created", False)
    if right_panel_created is False:

        mgr.display_widget(
            content,
            right_panel_id,
            VisualizationManager.panel_widget_preset()
            | {
                "text_color": 0xFFFFFFFF,
                "prim_path_source": "/World/defaultGroundPlane/GroundPlane",
                "translation": Gf.Vec3f(0, 3, 1),
            },
        )
        mgr.set_attr("_right_panel_created", True)

    update_instruction(right_panel_id, content)
    mgr.set_attr("_right_panel_updated_times", updated_times + 1)


def apply_sample_visualization():

    XRVisualization.register_callback(TriggerType.TRIGGER_ON_EVENT, {"event_name": "ik_error"}, _sample_handle_ik_error)



    XRVisualization.set_attrs({
        "left_panel_id": "/left_panel",
        "left_panel_translation": Gf.Vec3f(-2, 2.6, 2),
        "left_panel_updated_times": 0,
        "right_panel_updated_times": 0,
    })
    XRVisualization.register_callback(TriggerType.TRIGGER_ON_PERIOD, {"period": 1.0}, _sample_update_left_panel)



    XRVisualization.set_attrs({
        "right_panel_id": "/right_panel",
        "right_panel_translation": Gf.Vec3f(1.5, 2, 2),
    })
    XRVisualization.register_callback(
        TriggerType.TRIGGER_ON_CHANGE, {"variable_name": "right_panel_data"}, _sample_update_right_panel
    )


    XRVisualization.set_attrs({
        "error_text_color": 0xFF0000FF,
    })
    XRVisualization.register_callback(TriggerType.TRIGGER_ON_UPDATE, {}, _sample_update_error_text_color)


def run_simulator(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
):
    """Run the simulator."""


    sim_dt = sim.get_physics_dt()

    apply_sample_visualization()


    while simulation_app.is_running():
        if int(time.time()) % 10 < 1:
            XRVisualization.push_event("ik_error")

        XRVisualization.push_data({"right_panel_data": get_camera_position()})

        sim.step()
        scene.update(sim_dt)


def main():
    """Main function."""


    sim_cfg = sim_utils.SimulationCfg(dt=0.005)
    sim = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view(eye=(8, 0, 4), target=(0.0, 0.0, 0.0))

    scene = InteractiveScene(SimpleSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0))

    sim.reset()

    print("[INFO]: Setup complete...")

    run_simulator(sim, scene)


if __name__ == "__main__":

    main()

    simulation_app.close()
