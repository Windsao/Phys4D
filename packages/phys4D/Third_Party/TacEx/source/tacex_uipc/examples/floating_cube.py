"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Test scene for GIPC.")

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app






"""Rest everything follows."""
import pathlib

import isaacsim.core.utils.prims as prims_utils
from isaacsim.util.debug_draw import _debug_draw

draw = _debug_draw.acquire_debug_draw_interface()

import numpy as np

from uipc import Animation, Vector3, builtin, view
from uipc.constitution import SoftPositionConstraint
from uipc.geometry import GeometrySlot, SimplicialComplex

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils.timer import Timer

from tacex_uipc import UipcObject, UipcObjectCfg, UipcSim, UipcSimCfg


def main():
    """Main function."""


    sim_cfg = sim_utils.SimulationCfg(dt=1 / 60)
    sim = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])


    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    cfg_light_dome = sim_utils.DomeLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light_dome.func("/World/lightDome", cfg_light_dome, translation=(1, 0, 10))

    prims_utils.define_prim("/World/Objects", "Xform")


    uipc_cfg = UipcSimCfg(

        contact=UipcSimCfg.Contact(

            d_hat=0.01,
        )
    )
    uipc_sim = UipcSim(uipc_cfg)









    tet_cube_asset_path = pathlib.Path(__file__).parent.resolve() / "assets" / "cube.usd"
    cube_cfg = UipcObjectCfg(
        prim_path="/World/Objects/Cube0",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 2.25]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(tet_cube_asset_path),

        ),

        constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg(),
    )
    cube = UipcObject(cube_cfg, uipc_sim)


    spc = SoftPositionConstraint()


    spc.apply_to(cube.uipc_meshes[0], 100)















    sim.reset()




    animator = uipc_sim.scene.animator()

    def animate_tet(info: Animation.UpdateInfo):
        geo_slots: list[GeometrySlot] = info.geo_slots()
        geo: SimplicialComplex = geo_slots[0].geometry()
        rest_geo_slots: list[GeometrySlot] = info.rest_geo_slots()
        rest_geo: SimplicialComplex = rest_geo_slots[0].geometry()

        is_constrained = geo.vertices().find(builtin.is_constrained)
        is_constrained_view = view(is_constrained)
        aim_position = geo.vertices().find(builtin.aim_position)
        aim_position_view = view(aim_position)
        rest_position_view = rest_geo.positions().view()

        is_constrained_view[0] = 1
        is_constrained_view[1] = 1

        t = info.dt() * info.frame()
        theta = np.pi * t
        z = -np.sin(theta)

        aim_position_view[0] = rest_position_view[0] + Vector3.UnitZ() * z

    animator.insert(cube.uipc_scene_objects[0], animate_tet)




    uipc_sim.setup_sim()


    print("[INFO]: Setup complete...")

    step = 1
    start_uipc_test = True

    total_uipc_sim_time = 0.0
    total_uipc_render_time = 0.0

    while simulation_app.is_running():
        sim.render()

        if start_uipc_test:
            print("")
            print("====================================================================================")
            print("====================================================================================")
            print("Step number ", step)
            with Timer("[INFO]: Time taken for uipc sim step", name="uipc_step"):
                sim.step()

            with Timer("[INFO]: Time taken for updating the render meshes", name="render_update"):

                uipc_sim.update_render_meshes()





            total_uipc_sim_time += Timer.get_timer_info("uipc_step")
            total_uipc_render_time += Timer.get_timer_info("render_update")

            step += 1


        if sim.is_playing() is False:
            start_uipc_test = True
            print("Start uipc simulation by pressing Play")


if __name__ == "__main__":

    main()

    simulation_app.close()
