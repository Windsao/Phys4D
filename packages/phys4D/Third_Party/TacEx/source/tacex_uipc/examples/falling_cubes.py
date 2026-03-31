"""This script demonstrates how to spawn prims into the scene.

.. code-block:: bash

    # Usage
    isaaclab -p ./examples/falling_cubes.py
"""

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

import random

from pxr import Sdf

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils.timer import Timer

from tacex_uipc import UipcObject, UipcObjectCfg, UipcSim, UipcSimCfg
from tacex_uipc.utils import TetMeshCfg




def design_scene():
    """Designs the scene by spawning ground plane, light, objects and meshes from usd files."""

    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)


    cfg_light_dome = sim_utils.DomeLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light_dome.func("/World/lightDome", cfg_light_dome, translation=(1, 0, 10))


    prims_utils.define_prim("/World/Objects", "Xform")


def change_mat_color(stage, shader_prim_path, color):

    shader_prim = stage.GetPrimAtPath(shader_prim_path)
    if not shader_prim.GetAttribute("inputs:diffuse_color_constant").IsValid():
        shader_prim.CreateAttribute("inputs:diffuse_color_constant", Sdf.ValueTypeNames.Color3f, custom=True).Set(
            (0.0, 0.0, 0.0)
        )

    if not shader_prim.GetAttribute("inputs:diffuse_tint").IsValid():
        shader_prim.CreateAttribute("inputs:diffuse_tint", Sdf.ValueTypeNames.Color3f, custom=True).Set((0.0, 0.0, 0.0))


    shader_prim.GetAttribute("inputs:diffuse_color_constant").Set(color)
    shader_prim.GetAttribute("inputs:diffuse_tint").Set(color)


def main():
    """Main function."""



    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])


    design_scene()


    uipc_cfg = UipcSimCfg(

        contact=UipcSimCfg.Contact(

            d_hat=0.01,
        )
    )
    uipc_sim = UipcSim(uipc_cfg)

    mesh_cfg = TetMeshCfg(
        stop_quality=8,
        max_its=100,
        edge_length_r=0.1,

    )
    print("Mesh cfg ", mesh_cfg)



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














    tet_cube_asset_path = pathlib.Path(__file__).parent.resolve() / "assets" / "cube.usd"

    num_cubes = 5
    cubes = []
    for i in range(num_cubes):
        if i % 2 == 1:
            constitution_type = UipcObjectCfg.AffineBodyConstitutionCfg(kinematic=True)
        else:
            constitution_type = UipcObjectCfg.StableNeoHookeanCfg()


        pos = (0, 0, 3.0 + 0.3 * i)
        rot = (
            random.uniform(-1.0, 1.0),
            random.uniform(-1.0, 1.0),
            random.uniform(-1.0, 1.0),
            random.uniform(-1.0, 1.0),
        )
        cube_cfg = UipcObjectCfg(
            prim_path=f"/World/Objects/Cube{i + 1}",
            init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=rot),
            spawn=sim_utils.UsdFileCfg(usd_path=str(tet_cube_asset_path), scale=(0.15, 0.15, 0.15)),
            constitution_cfg=constitution_type,
        )
        cubeX = UipcObject(cube_cfg, uipc_sim)
        cubes.append(cubeX)

    rot = (random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0))
    cube_cfg = UipcObjectCfg(
        prim_path="/World/Objects/CubeTop",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 3.65 + 0.3 * num_cubes], rot=rot),
        spawn=sim_utils.UsdFileCfg(usd_path=str(tet_cube_asset_path), scale=(1.0, 1.0, 1.0)),

        constitution_cfg=UipcObjectCfg.AffineBodyConstitutionCfg(),
    )
    cube_top = UipcObject(cube_cfg, uipc_sim)


    sim.reset()




    uipc_sim.setup_sim()


    print("[INFO]: Setup complete...")

    step = 1
    start_uipc_test = False

    total_uipc_sim_time = 0.0
    total_uipc_render_time = 0.0

    num_resets = 0

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

        if step % 250 == 0:
            print("")
            print("====================================================================================")
            print("====================================================================================")
            print("Reset simulation")
            if start_uipc_test:
                print("systems offsets ", uipc_sim._system_vertex_offsets)
                cube.write_vertex_positions_to_sim(vertex_positions=cube.init_vertex_pos)
                cube_top.write_vertex_positions_to_sim(vertex_positions=cube_top.init_vertex_pos)

                small_cube_id = random.randint(0, num_cubes - 1)
                cubes[small_cube_id].write_vertex_positions_to_sim(
                    vertex_positions=cubes[small_cube_id].init_vertex_pos
                )

                uipc_sim.reset()
                sim.render()

            avg_uipc_step_time = total_uipc_sim_time / step
            print(f"Sim step for uipc took in avg {avg_uipc_step_time} per frame.")

            avg_uipc_render_time = total_uipc_render_time / step
            print(f"Render update for uipc took in avg {avg_uipc_render_time} per frame.")
            print("====================================================================================")

            step = 1
            num_resets += 1

        if num_resets == 5:
            print("Stopping [Falling Cubes] example.")
            break


if __name__ == "__main__":

    main()

    simulation_app.close()
