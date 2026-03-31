from __future__ import annotations

import pathlib

import omni.usd
import usdrt
import usdrt.Usd
from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

try:
    from isaacsim.util.debug_draw import _debug_draw

    draw = _debug_draw.acquire_debug_draw_interface()
except ImportError:
    import warnings

    warnings.warn("_debug_draw failed to import", ImportWarning)
    draw = None

from uipc import Logger, Timer, builtin
from uipc.core import Engine, Scene, SceneIO, World
from uipc.geometry import extract_surface, ground
from uipc.unit import GPa

from tacex_uipc.objects import UipcObject
from tacex_uipc.utils import MeshGenerator


@configclass
class UipcSimCfg:
    device: str | None = "cuda"

    dt: float = 0.01

    sanity_check_enable: bool = True
    sanity_check_mode: str = "quiet"
    """
    Options: "quiet", "normal"
    "quiet" = do not export mesh
    """

    workspace: str = str(pathlib.Path().resolve())
    """Path to directory where the libuipc files (e.g. systems.json) should be saved.

    Defaults to current working directory.
    """

    logger_level: str = "Error"

    gravity: tuple = (0.0, 0.0, -9.8)
    ground_height: float = 0.0
    ground_normal: tuple = (0.0, 0.0, 1.0)

    @configclass
    class Newton:
        max_iter: int = 1024

        use_adaptive_tol: bool = False


        velocity_tol: float = 0.05
        """Convergence tolerance 1)
        max(dx*dt) <= velocity_tol
        """

        ccd_tol: float = 1.0
        """Convergence tolerance 2)
        ccd_toi >= ccd_tol
        """

        transrate_tol: float = 0.1 / 1.0
        """Convergence tolerance 3)
        max dF <=  dF <= transform_tol * dt

        e.g. 0.1/1.0 = 10%/s change in transform
        """

    newton: Newton = Newton()

    @configclass
    class LinearSystem:
        solver: str = "linear_pcg"
        """
        Options: linear_pcg,
        """

        tol_rate: float = 1e-3

    linear_system: LinearSystem = LinearSystem()

    @configclass
    class LineSearch:
        max_iter: int = 8

        report_energy: bool = False

    line_search: LineSearch = LineSearch()

    @configclass
    class Contact:
        enable: bool = True

        enable_friction: bool = True

        default_friction_ratio: float = 0.5

        default_contact_resistance: float = 10.0
        """
        in [GPa]
        """

        constitution: str = "ipc"

        d_hat: float = 0.001

        eps_velocity: float = 0.01
        """
        in [m/s]
        """

    contact: Contact = Contact()

    collision_detection_method: str = "linear_bvh"
    """
    Options: linear_bvh, informative_linear_bvh
    """

    diff_sim: bool = False


class UipcSim:
    cfg: UipcSimCfg

    def __init__(self, cfg: UipcSimCfg = None):
        """Initialize the uipc simulation."""

        self.isaac_sim = None

        if cfg is None:
            cfg = UipcSimCfg()
        self.cfg = cfg

        Timer.enable_all()
        if self.cfg.logger_level == "Error":
            Logger.set_level(Logger.Error)
        elif self.cfg.logger_level == "Info":
            Logger.set_level(Logger.Info)

        pathlib.Path(self.cfg.workspace).mkdir(parents=True, exist_ok=True)

        self.engine: Engine = Engine(backend_name=self.cfg.device, workspace=self.cfg.workspace)
        self.world: World = World(self.engine)


        uipc_config = Scene.default_config()
        uipc_config["dt"] = self.cfg.dt
        uipc_config["sanity_check"] = {"enable": self.cfg.sanity_check_enable, "mode": self.cfg.sanity_check_mode}
        uipc_config["gravity"] = [[self.cfg.gravity[0]], [self.cfg.gravity[1]], [self.cfg.gravity[2]]]
        uipc_config["collision_detection"]["method"] = self.cfg.collision_detection_method
        uipc_config["contact"] = {
            "enable": self.cfg.contact.enable,
            "constitution": self.cfg.contact.constitution,
            "d_hat": self.cfg.contact.d_hat,
            "eps_velocity": self.cfg.contact.eps_velocity,
            "friction": {"enable": self.cfg.contact.enable_friction},
        }
        uipc_config["diff_sim"] = {
            "enable": self.cfg.diff_sim,
        }
        uipc_config["line_search"] = {
            "max_iter": self.cfg.line_search.max_iter,
            "report_energy": self.cfg.line_search.report_energy,
        }
        uipc_config["linear_system"] = {
            "solver": self.cfg.linear_system.solver,
            "tol_rate": self.cfg.linear_system.tol_rate,
        }
        uipc_config["newton"] = {
            "ccd_tol": self.cfg.newton.ccd_tol,
            "max_iter": self.cfg.newton.max_iter,
            "use_adaptive_tol": self.cfg.newton.use_adaptive_tol,
            "velocity_tol": self.cfg.newton.velocity_tol,
            "transrate_tol": self.cfg.newton.transrate_tol,
        }

        self.scene = Scene(uipc_config)


        ground_obj = self.scene.objects().create("ground")
        g = ground(self.cfg.ground_height, self.cfg.ground_normal)
        ground_obj.geometries().create(g)


        self.scene.contact_tabular().default_model(
            friction_rate=self.cfg.contact.default_friction_ratio,
            resistance=self.cfg.contact.default_contact_resistance * GPa,
            enable=self.cfg.contact.enable,
        )


        self._system_vertex_offsets = {
            "uipc::backend::cuda::GlobalVertexManager": [0],
            "uipc::backend::cuda::FiniteElementMethod": [0],
            "uipc::backend::cuda::AffineBodyDynamics": [0],
        }


        self._surf_vertex_offsets = [0]

        self._fabric_meshes = []
        self.uipc_objects: list[UipcObject] = []

        self.stage = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())


        self.sio = SceneIO(self.scene)

    def __del__(self):
        """Unsubscribe from the callbacks."""

        if self._debug_vis_handle:
            self._debug_vis_handle.unsubscribe()
            self._debug_vis_handle = None

    def setup_sim(self):
        self.world.init(self.scene)
        self.world.retrieve()





        for uipc_obj in self.uipc_objects:
            geo_slot = uipc_obj.geo_slot_list[0]
            geo = geo_slot.geometry()
            gvo = geo.meta().find(builtin.global_vertex_offset)
            global_vertex_offset = int(gvo.view()[0])
            self._system_vertex_offsets["uipc::backend::cuda::GlobalVertexManager"].append(global_vertex_offset)

            uipc_obj.global_system_id = len(self._system_vertex_offsets["uipc::backend::cuda::GlobalVertexManager"]) - 1
            print(f"{uipc_obj.cfg.prim_path} has global id {uipc_obj.global_system_id}")


        self.isaac_sim: sim_utils.SimulationContext = sim_utils.SimulationContext.instance()
        self.isaac_sim.add_physics_callback("uicp_step", self.step)

    def step(self, dt=0):
        self.world.advance()
        self.world.retrieve()

    def reset(self):










        self.world.retrieve()
        self.update_render_meshes()

    def update_render_meshes(self, dt=0):
        all_trimesh_points = self.sio.simplicial_surface(2).positions().view().reshape(-1, 3)

        for i, fabric_prim in enumerate(self._fabric_meshes):
            trimesh_points = all_trimesh_points[self._surf_vertex_offsets[i] : self._surf_vertex_offsets[i + 1]]

            fabric_mesh_points = fabric_prim.GetAttribute("points")
            fabric_mesh_points.Set(usdrt.Vt.Vec3fArray(trimesh_points))






        if self.isaac_sim is not None:

            self.isaac_sim.render()

    @staticmethod
    def get_sim_time_report(as_json: bool = False):
        report = None
        if as_json:
            report = Timer.report_as_json()
        else:
            Timer.report()
        return report

    def save_frame(self):
        """Saves the current frame into multiple files, which can be retrieved to replay the animation later.

        For replaying use the method `replay_frames`
        """
        self.world.dump()


    def replay_frame(self, frame_num):
        """

        Won't work if the loaded tet meshes are different!
        """

        if self.world.recover(frame_num):
            self.world.retrieve()
        else:
            print(f"No data for frame {frame_num}.")

    def init_libuipc_scene_rendering(self):
        num_objs = self.scene.objects().size()
        for obj_id in range(1, num_objs):

            obj = self.scene.objects().find(obj_id)
            obj_geometry_ids = obj.geometries().ids()



            for id in obj_geometry_ids:
                scene_geom = self.scene.geometries()
                geo_slot, geo_slot_rest = scene_geom.find(id)

                geom = geo_slot.geometry()


                num_geom = geom.instances().size()
                for instance_num in range(num_geom):
                    id += instance_num

                    stage = omni.usd.get_context().get_stage()
                    prim_path = f"/World/uipc_mesh_{id}"
                    prim = UsdGeom.Mesh.Define(stage, prim_path)

                    dim = geom.dim()
                    if dim == 2:

                        surf = geom
                    else:

                        surf = extract_surface(geom)

                    surf_tri = surf.triangles().topo().view().reshape(-1).tolist()
                    surf_points_world = surf.positions().view().reshape(-1, 3)

                    MeshGenerator.update_usd_mesh(prim=prim, surf_points=surf_points_world, triangles=surf_tri)


                    fabric_stage = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())
                    fabric_prim = fabric_stage.GetPrimAtPath(usdrt.Sdf.Path(prim_path))


                    if not fabric_prim.HasAttribute("Deformable"):
                        fabric_prim.CreateAttribute("Deformable", usdrt.Sdf.ValueTypeNames.PrimTypeTag, True)


                    rtxformable = usdrt.Rt.Xformable(fabric_prim)
                    rtxformable.CreateFabricHierarchyWorldMatrixAttr()

                    rtxformable.GetFabricHierarchyWorldMatrixAttr().Set(usdrt.Gf.Matrix4d())


                    fabric_mesh_points_attr = fabric_prim.GetAttribute("points")
                    fabric_mesh_points_attr.Set(usdrt.Vt.Vec3fArray(surf_points_world))


                    self._fabric_meshes.append(fabric_prim)


                    num_surf_points = surf_points_world.shape[0]
                    self._surf_vertex_offsets.append(self._surf_vertex_offsets[-1] + num_surf_points)
