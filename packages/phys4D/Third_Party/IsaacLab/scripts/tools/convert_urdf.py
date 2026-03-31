




"""
Utility to convert a URDF into USD format.

Unified Robot Description Format (URDF) is an XML file format used in ROS to describe all elements of
a robot. For more information, see: http://wiki.ros.org/urdf

This script uses the URDF importer extension from Isaac Sim (``isaacsim.asset.importer.urdf``) to convert a
URDF asset into USD format. It is designed as a convenience script for command-line use. For more
information on the URDF importer, see the documentation for the extension:
https://docs.isaacsim.omniverse.nvidia.com/latest/robot_setup/ext_isaacsim_asset_importer_urdf.html


positional arguments:
  input               The path to the input URDF file.
  output              The path to store the USD file.

optional arguments:
  -h, --help                Show this help message and exit
  --merge-joints            Consolidate links that are connected by fixed joints. (default: False)
  --fix-base                Fix the base to where it is imported. (default: False)
  --joint-stiffness         The stiffness of the joint drive. (default: 100.0)
  --joint-damping           The damping of the joint drive. (default: 1.0)
  --joint-target-type       The type of control to use for the joint drive. (default: "position")

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Utility to convert a URDF into USD format.")
parser.add_argument("input", type=str, help="The path to the input URDF file.")
parser.add_argument("output", type=str, help="The path to store the USD file.")
parser.add_argument(
    "--merge-joints",
    action="store_true",
    default=False,
    help="Consolidate links that are connected by fixed joints.",
)
parser.add_argument("--fix-base", action="store_true", default=False, help="Fix the base to where it is imported.")
parser.add_argument(
    "--joint-stiffness",
    type=float,
    default=100.0,
    help="The stiffness of the joint drive.",
)
parser.add_argument(
    "--joint-damping",
    type=float,
    default=1.0,
    help="The damping of the joint drive.",
)
parser.add_argument(
    "--joint-target-type",
    type=str,
    default="position",
    choices=["position", "velocity", "none"],
    help="The type of control to use for the joint drive.",
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

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.utils.assets import check_file_path
from isaaclab.utils.dict import print_dict


def main():

    urdf_path = args_cli.input
    if not os.path.isabs(urdf_path):
        urdf_path = os.path.abspath(urdf_path)
    if not check_file_path(urdf_path):
        raise ValueError(f"Invalid file path: {urdf_path}")

    dest_path = args_cli.output
    if not os.path.isabs(dest_path):
        dest_path = os.path.abspath(dest_path)


    urdf_converter_cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=os.path.dirname(dest_path),
        usd_file_name=os.path.basename(dest_path),
        fix_base=args_cli.fix_base,
        merge_fixed_joints=args_cli.merge_joints,
        force_usd_conversion=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=args_cli.joint_stiffness,
                damping=args_cli.joint_damping,
            ),
            target_type=args_cli.joint_target_type,
        ),
    )


    print("-" * 80)
    print("-" * 80)
    print(f"Input URDF file: {urdf_path}")
    print("URDF importer config:")
    print_dict(urdf_converter_cfg.to_dict(), nesting=0)
    print("-" * 80)
    print("-" * 80)


    urdf_converter = UrdfConverter(urdf_converter_cfg)

    print("URDF importer output:")
    print(f"Generated USD file: {urdf_converter.usd_path}")
    print("-" * 80)
    print("-" * 80)



    carb_settings_iface = carb.settings.get_settings()

    local_gui = carb_settings_iface.get("/app/window/enabled")

    livestream_gui = carb_settings_iface.get("/app/livestream/enabled")


    if local_gui or livestream_gui:

        stage_utils.open_stage(urdf_converter.usd_path)

        app = omni.kit.app.get_app_interface()

        with contextlib.suppress(KeyboardInterrupt):
            while app.is_running():

                app.update()


if __name__ == "__main__":

    main()

    simulation_app.close()
