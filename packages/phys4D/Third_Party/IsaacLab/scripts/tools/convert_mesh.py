




"""
Utility to convert a OBJ/STL/FBX into USD format.

The OBJ file format is a simple data-format that represents 3D geometry alone — namely, the position
of each vertex, the UV position of each texture coordinate vertex, vertex normals, and the faces that
make each polygon defined as a list of vertices, and texture vertices.

An STL file describes a raw, unstructured triangulated surface by the unit normal and vertices (ordered
by the right-hand rule) of the triangles using a three-dimensional Cartesian coordinate system.

FBX files are a type of 3D model file created using the Autodesk FBX software. They can be designed and
modified in various modeling applications, such as Maya, 3ds Max, and Blender. Moreover, FBX files typically
contain mesh, material, texture, and skeletal animation data.
Link: https://www.autodesk.com/products/fbx/overview


This script uses the asset converter extension from Isaac Sim (``omni.kit.asset_converter``) to convert a
OBJ/STL/FBX asset into USD format. It is designed as a convenience script for command-line use.


positional arguments:
  input               The path to the input mesh (.OBJ/.STL/.FBX) file.
  output              The path to store the USD file.

optional arguments:
  -h, --help                    Show this help message and exit
  --make-instanceable,          Make the asset instanceable for efficient cloning. (default: False)
  --collision-approximation     The method used for approximating collision mesh. Defaults to convexDecomposition.
                                Set to \"none\" to not add a collision mesh to the converted mesh. (default: convexDecomposition)
  --mass                        The mass (in kg) to assign to the converted asset. (default: None)

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher


_valid_collision_approx = [
    "convexDecomposition",
    "convexHull",
    "triangleMesh",
    "meshSimplification",
    "sdf",
    "boundingCube",
    "boundingSphere",
    "none",
]


parser = argparse.ArgumentParser(description="Utility to convert a mesh file into USD format.")
parser.add_argument("input", type=str, help="The path to the input mesh file.")
parser.add_argument("output", type=str, help="The path to store the USD file.")
parser.add_argument(
    "--make-instanceable",
    action="store_true",
    default=False,
    help="Make the asset instanceable for efficient cloning.",
)
parser.add_argument(
    "--collision-approximation",
    type=str,
    default="convexDecomposition",
    choices=_valid_collision_approx,
    help="The method used for approximating the collision mesh. Set to 'none' to disable collision mesh generation.",
)
parser.add_argument(
    "--mass",
    type=float,
    default=None,
    help="The mass (in kg) to assign to the converted asset. If not provided, then no mass is added.",
)

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import os

import carb
import isaacsim.core.utils.stage as stage_utils
import omni.kit.app

from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.schemas import schemas_cfg
from isaaclab.utils.assets import check_file_path
from isaaclab.utils.dict import print_dict

collision_approximation_map = {
    "convexDecomposition": schemas_cfg.ConvexDecompositionPropertiesCfg,
    "convexHull": schemas_cfg.ConvexHullPropertiesCfg,
    "triangleMesh": schemas_cfg.TriangleMeshPropertiesCfg,
    "meshSimplification": schemas_cfg.TriangleMeshSimplificationPropertiesCfg,
    "sdf": schemas_cfg.SDFMeshPropertiesCfg,
    "boundingCube": schemas_cfg.BoundingCubePropertiesCfg,
    "boundingSphere": schemas_cfg.BoundingSpherePropertiesCfg,
    "none": None,
}


def main():

    mesh_path = args_cli.input
    if not os.path.isabs(mesh_path):
        mesh_path = os.path.abspath(mesh_path)
    if not check_file_path(mesh_path):
        raise ValueError(f"Invalid mesh file path: {mesh_path}")


    dest_path = args_cli.output
    if not os.path.isabs(dest_path):
        dest_path = os.path.abspath(dest_path)


    if args_cli.mass is not None:
        mass_props = schemas_cfg.MassPropertiesCfg(mass=args_cli.mass)
        rigid_props = schemas_cfg.RigidBodyPropertiesCfg()
    else:
        mass_props = None
        rigid_props = None


    collision_props = schemas_cfg.CollisionPropertiesCfg(collision_enabled=args_cli.collision_approximation != "none")


    cfg_class = collision_approximation_map.get(args_cli.collision_approximation)
    if cfg_class is None and args_cli.collision_approximation != "none":
        valid_keys = ", ".join(sorted(collision_approximation_map.keys()))
        raise ValueError(
            f"Invalid collision approximation type '{args_cli.collision_approximation}'. "
            f"Valid options are: {valid_keys}."
        )
    collision_cfg = cfg_class() if cfg_class is not None else None

    mesh_converter_cfg = MeshConverterCfg(
        mass_props=mass_props,
        rigid_props=rigid_props,
        collision_props=collision_props,
        asset_path=mesh_path,
        force_usd_conversion=True,
        usd_dir=os.path.dirname(dest_path),
        usd_file_name=os.path.basename(dest_path),
        make_instanceable=args_cli.make_instanceable,
        mesh_collision_props=collision_cfg,
    )


    print("-" * 80)
    print("-" * 80)
    print(f"Input Mesh file: {mesh_path}")
    print("Mesh importer config:")
    print_dict(mesh_converter_cfg.to_dict(), nesting=0)
    print("-" * 80)
    print("-" * 80)


    mesh_converter = MeshConverter(mesh_converter_cfg)

    print("Mesh importer output:")
    print(f"Generated USD file: {mesh_converter.usd_path}")
    print("-" * 80)
    print("-" * 80)



    carb_settings_iface = carb.settings.get_settings()

    local_gui = carb_settings_iface.get("/app/window/enabled")

    livestream_gui = carb_settings_iface.get("/app/livestream/enabled")


    if local_gui or livestream_gui:

        stage_utils.open_stage(mesh_converter.usd_path)

        app = omni.kit.app.get_app_interface()

        with contextlib.suppress(KeyboardInterrupt):
            while app.is_running():

                app.update()


if __name__ == "__main__":

    main()

    simulation_app.close()
