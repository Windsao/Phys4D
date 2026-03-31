




"""
Convert a mesh file to `.obj` using blender.

This file processes a given dae mesh file and saves the resulting mesh file in obj format.

It needs to be called using the python packaged with blender, i.e.:

    blender --background --python blender_obj.py -- -in_file FILE -out_file FILE

For more information: https://docs.blender.org/api/current/index.html

The script was tested on Blender 3.2 on Ubuntu 20.04LTS.
"""

import bpy
import os
import sys


def parse_cli_args():
    """Parse the input command line arguments."""
    import argparse



    argv = sys.argv

    if "--" not in argv:
        argv = []
    else:
        argv = argv[argv.index("--") + 1 :]


    usage_text = (
        f"Run blender in background mode with this script:\n\tblender --background --python {__file__} -- [options]"
    )
    parser = argparse.ArgumentParser(description=usage_text)

    parser.add_argument("-i", "--in_file", metavar="FILE", type=str, required=True, help="Path to input OBJ file.")
    parser.add_argument("-o", "--out_file", metavar="FILE", type=str, required=True, help="Path to output OBJ file.")
    args = parser.parse_args(argv)

    if not argv or not args.in_file or not args.out_file:
        parser.print_help()
        return None

    return args


def convert_to_obj(in_file: str, out_file: str, save_usd: bool = False):
    """Convert a mesh file to `.obj` using blender.

    Args:
        in_file: Input mesh file to process.
        out_file: Path to store output obj file.
    """

    if not os.path.exists(in_file):
        raise FileNotFoundError(in_file)

    if not out_file.endswith(".obj"):
        out_file += ".obj"

    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file), exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)

    if in_file.endswith(".dae"):
        bpy.ops.wm.collada_import(filepath=in_file)
    elif in_file.endswith(".stl") or in_file.endswith(".STL"):
        bpy.ops.import_mesh.stl(filepath=in_file)
    else:
        raise ValueError(f"Input file not in dae/stl format: {in_file}")



    bpy.ops.export_scene.obj(
        filepath=out_file, check_existing=False, axis_forward="Y", axis_up="Z", global_scale=1, path_mode="RELATIVE"
    )

    if save_usd:
        out_file = out_file.replace("obj", "usd")
        bpy.ops.wm.usd_export(filepath=out_file, check_existing=False)


if __name__ == "__main__":

    cli_args = parse_cli_args()

    if cli_args is None:
        sys.exit()

    convert_to_obj(cli_args.in_file, cli_args.out_file)
