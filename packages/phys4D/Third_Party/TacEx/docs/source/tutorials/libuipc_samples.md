
**libuipc** provides plenty of [samples](https:

- we need to start Isaac Sim first and create a base scene
- we render in Isaac Sim, instead of polyscope

Let's take a look at one example.



[hello_libuipc](https:

- configure and create the uipc simulation
- set default contact properties (friction ratio and contact resistance)
- create meshes and objects for the UIPC simulation
- step through the sim
- render the results with polyscope

Let's take a look how `hello_libuipc` looks like in TacEx.



<details>
<summary>Code for `hello_libuipc.py` in TacEx</summary>

![1_hello_libuipc.py](https:
```python
"""Showcase on how to use libuipc with Isaac Sim/Lab.

This example corresponds to
https:


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

import isaaclab.sim as sim_utils
from isaaclab.utils.timer import Timer

from pxr import Gf, Sdf, Usd, UsdGeom
import omni.usd
import usdrt

import uipc
from uipc.core import Engine, World, Scene
from uipc.geometry import tetmesh, label_surface, label_triangle_orient, flip_inward_triangles, extract_surface
from uipc.constitution import AffineBodyConstitution
from uipc.unit import MPa, GPa

from tacex_uipc import UipcSim, UipcSimCfg

def setup_base_scene(sim: sim_utils.SimulationContext):
    """To make the scene pretty.

    """

    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)


    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func(
        prim_path="/World/defaultGroundPlane",
        cfg=cfg_ground,
        translation=[0, -1, 0],
        orientation=[0.7071068, -0.7071068, 0, 0]
    )


    cfg_light_dome = sim_utils.DomeLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light_dome.func("/World/lightDome", cfg_light_dome, translation=(1, 10, 0))

def setup_libuipc_scene(uipc_sim: UipcSim):
    scene = uipc_sim.scene


    abd = AffineBodyConstitution()


    scene.contact_tabular().default_model(0.5, 1.0 * GPa)
    default_element = scene.contact_tabular().default_element()


    Vs = np.array([[0,1,0],
                   [0,0,1],
                   [-np.sqrt(3)/2, 0, -0.5],
                   [np.sqrt(3)/2, 0, -0.5]])
    Ts = np.array([[0,1,2,3]])


    base_mesh = tetmesh(Vs, Ts)

    abd.apply_to(base_mesh, 100 * MPa)

    default_element.apply_to(base_mesh)


    label_surface(base_mesh)

    label_triangle_orient(base_mesh)

    base_mesh = flip_inward_triangles(base_mesh)

    mesh1 = base_mesh.copy()
    pos_view = uipc.view(mesh1.positions())

    pos_view += uipc.Vector3.UnitY() * 1.5

    mesh2 = base_mesh.copy()
    is_fixed = mesh2.instances().find(uipc.builtin.is_fixed)
    is_fixed_view = uipc.view(is_fixed)
    is_fixed_view[:] = 1


    object1 = scene.objects().create("upper_tet")
    object1.geometries().create(mesh1)

    object2 = scene.objects().create("lower_tet")
    object2.geometries().create(mesh2)

def main():
    """Main function."""

    sim_cfg = sim_utils.SimulationCfg(
        dt=1/60,
        gravity=[0.0, -9.8, 0.0],
    )
    sim = sim_utils.SimulationContext(sim_cfg)

    setup_base_scene(sim)


    uipc_cfg = UipcSimCfg(
        dt=0.02,
        gravity=[0.0, -9.8, 0.0],
        ground_normal=[0, 1, 0],
        ground_height=-1.0,

        contact=UipcSimCfg.Contact(
            default_friction_ratio=0.5,
            default_contact_resistance=1.0,
        )
    )
    uipc_sim = UipcSim(uipc_cfg)

    setup_libuipc_scene(uipc_sim)

    uipc_sim.init_libuipc_scene_rendering()

    uipc_sim.setup_sim()


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
```
</details>



Before we import python modules, we first need to launch the Isaac Sim application

```python
import argparse
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Showcase on how to use libuipc with Isaac Sim/Lab.")

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
```

This is the first crucial step for every `standalone` script in Isaac Sim.

Now, let's take a look at our `main` method, which contains the main logic of the script, i.e.

- setting up simulation
- setting up scenes (plural, cause we need to set up a Isaac scene and a libuipc scene)
- stepping through simulation
- rendering



Besides launching Isaac Sim, we also need to start the Isaac simulation itself (otherwise, we cannot render anything).\
To start the Isaac Simulation, we create a `SimulationCfg` object and initialize the `SimulationContext`

```python
    sim_cfg = sim_utils.SimulationCfg(
        dt=1/60,
        gravity=[0.0, -9.8, 0.0],
    )
    sim = sim_utils.SimulationContext(sim_cfg)
```

Then we create the base scene via

```python
setup_base_scene(sim)
```

We will look at the scene setup later. First we look at how we initialize the UIPC simulation.

In TacEx the configuration happens through `configclasses`, just as the Isaac simulation had to be initialized.\
While the original libuipc sample sets its scene config first and then the default contact properties, we can do it already with our `UipcSimCfg` class:

```python


    uipc_cfg = UipcSimCfg(
        dt=0.02,
        gravity=[0.0, -9.8, 0.0],
        ground_normal=[0, 1, 0],
        ground_height=-1.0,

        contact=UipcSimCfg.Contact(
            default_friction_ratio=0.5,
            default_contact_resistance=1.0,
        )
    )
    uipc_sim = UipcSim(uipc_cfg)
```

After that, we setup the libuipc scene via

```python
setup_libuipc_scene(uipc_sim.scene)
```



We need a basic Isaac scene, so that the rendered results is somewhat pleasant to look at.

The default coordinate system in Isaac has the z-axis showing upwards. Every libuipc samples has y-axix showing upwards, so we match this in our examples.

```python

    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
```

For our base scene we create a ground

```python

    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func(
        prim_path="/World/defaultGroundPlane",
        cfg=cfg_ground,
        translation=[0, -1, 0],
        orientation=[0.7071068, -0.7071068, 0, 0]
    )
```

with translation set to match the ground height set in the `UipcSimCfg` and orientation set to match that the y-axis shows upwards.

Then we spawn a light

```python
    cfg_light_dome = sim_utils.DomeLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light_dome.func("/World/lightDome", cfg_light_dome, translation=(1, 10, 0))
```

We put this inside the method `setup_base_scene`, which can be used like this for every libuipc-sample.

Next, we take a look on how we can setup the libuipc scene.



This is actually identically to the scene creation of the [libuipc sample](https:

- create constitution and contact model
- create default contact settings (this is redundant, but we wanted to show that you can just copy-paste the sample)
- create meshes and objects

```python
    cfg_light_dome = sim_utils.DomeLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light_dome.func("/World/lightDome", cfg_light_dome, translation=(1, 10, 0))
```

This is actually the same for the other libuipc samples. You can basically copy paste the libuipc sample code from "after scene and world configuration" to "the world initizalisation happens" into the method `setup_libuipc_scene`.



Before we can run the simulation, we just need to initialize the `world` of the UIPC simulation and setup the libuipc rendering.

In the original it looks like this:

```python
world.init(scene)
sgui = SceneGUI(scene)

ps.init()
tri_surf, line_surf, point_surf = sgui.register()
tri_surf.set_edge_width(1)
```

In TacEx like this:

```python

    uipc_sim.setup_sim()
    uipc_sim.init_libuipc_scene_rendering()
```



In the original it looks like this:

```python
    if(run):
        world.advance()
        world.retrieve()
        sgui.update()
```

In TacEx like this:

```python
    uipc_sim.step()
    uipc_sim.update_render_meshes()

    sim.render()
```

The `uipc_sim.step()` method advances and retrieves the state of the UIPC simulation world. `uipc_sim.update_render_meshes()` updates the USD meshes in Isaac with the newest data from the UIPC simulation and `sim.render()` renders the Isaac scene.



You can use libuipc in TacEx very similar to how you would use it originally. Just copy the example file and adjust the `setup_libuipc_scene` method and you should basically be good to go. Feel free to take a look at our other libuipc-samples and compare them to the originals.

TacEx also provides an API which corresponds to the Isaac Lab API: We use configclasses for creating UIPC objects.\
What is showcased in the libuipc-samples happens "under the hood" now. Another big difference is that we usually use USD assets and not `.msh` files.

Further information can be found here TacEx UIPC Tutorial.
