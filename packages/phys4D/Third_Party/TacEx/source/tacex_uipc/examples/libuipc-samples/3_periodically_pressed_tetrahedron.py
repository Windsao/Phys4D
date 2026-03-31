"""Showcase on how to use libuipc with Isaac Sim/Lab.

This example corresponds to
https://github.com/spiriMirror/libuipc-samples/blob/main/python/1_hello_libuipc/main.py


"""

"""Launch Isaac Sim Simulator first."""
import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Showcase on how to use libuipc with Isaac Sim/Lab.")

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np

import omni.usd
from pxr import UsdGeom
from uipc import Animation, Vector3, builtin, view
from uipc.constitution import ElasticModuli, SoftPositionConstraint, StableNeoHookean
from uipc.geometry import (
    GeometrySlot,
    SimplicialComplex,
    flip_inward_triangles,
    label_surface,
    label_triangle_orient,
    tetmesh,
)
from uipc.unit import MPa

import isaaclab.sim as sim_utils
from isaaclab.utils.timer import Timer

from tacex_uipc import UipcSim, UipcSimCfg


def setup_base_scene(sim: sim_utils.SimulationContext):
    """To make the scene pretty."""

    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)


    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func(
        prim_path="/World/defaultGroundPlane",
        cfg=cfg_ground,
        translation=[0, -0.5, 0],
        orientation=[0.7071068, -0.7071068, 0, 0],
    )


    cfg_light_dome = sim_utils.DomeLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light_dome.func("/World/lightDome", cfg_light_dome, translation=(1, 10, 0))


def setup_libuipc_scene(scene):
    snh = StableNeoHookean()
    spc = SoftPositionConstraint()
    tet_object = scene.objects().create("tet_object")
    Vs = np.array([[0, 1, 0], [0, 0, 1], [-np.sqrt(3) / 2, 0, -0.5], [np.sqrt(3) / 2, 0, -0.5]])
    Ts = np.array([[0, 1, 2, 3]])
    tet = tetmesh(Vs, Ts)
    label_surface(tet)
    label_triangle_orient(tet)
    tet = flip_inward_triangles(tet)
    moduli = ElasticModuli.youngs_poisson(0.1 * MPa, 0.49)
    snh.apply_to(tet, moduli)
    spc.apply_to(tet, 100)
    tet_object.geometries().create(tet)

    animator = scene.animator()

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

        t = info.dt() * info.frame()
        theta = np.pi * t
        y = -np.sin(theta)

        aim_position_view[0] = rest_position_view[0] + Vector3.UnitY() * y

    animator.insert(tet_object, animate_tet)


def main():
    """Main function."""

    sim_cfg = sim_utils.SimulationCfg(
        dt=1 / 60,
        gravity=[0.0, -9.8, 0.0],
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    setup_base_scene(sim)


    uipc_cfg = UipcSimCfg(
        dt=0.02,
        gravity=[0.0, -9.8, 0.0],
        ground_normal=[0, 1, 0],
        ground_height=-0.5,

        contact=UipcSimCfg.Contact(
            default_friction_ratio=0.1,
            default_contact_resistance=1.0,
        ),
    )
    uipc_sim = UipcSim(uipc_cfg)

    setup_libuipc_scene(uipc_sim.scene)


    uipc_sim.setup_sim()
    uipc_sim.init_libuipc_scene_rendering()


    print("[INFO]: Setup complete...")

    step = 0

    total_uipc_sim_time = 0.0
    total_uipc_render_time = 0.0


    while simulation_app.is_running():

        sim.render()

        if sim.is_playing():
            print("")
            print("====================================================================================")
            print("====================================================================================")
            print("Step number ", step)
            with Timer("[INFO]: Time taken for uipc sim step", name="uipc_step"):
                uipc_sim.step()

            with Timer("[INFO]: Time taken for rendering", name="render_update"):
                uipc_sim.update_render_meshes()
                sim.render()


            uipc_sim.get_sim_time_report()
            total_uipc_sim_time += Timer.get_timer_info("uipc_step")
            total_uipc_render_time += Timer.get_timer_info("render_update")

            step += 1


if __name__ == "__main__":

    main()

    simulation_app.close()
