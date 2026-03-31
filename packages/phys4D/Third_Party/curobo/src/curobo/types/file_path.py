










"""Contains a class for storing file paths."""

from dataclasses import dataclass
from typing import Optional


from curobo.util.logger import log_error, log_info
from curobo.util_file import (
    get_assets_path,
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
)


@dataclass(frozen=True)
class ContentPath:
    """Dataclass to store root path of configuration and assets."""


    robot_config_root_path: str = get_robot_configs_path()


    robot_xrdf_root_path: str = get_robot_configs_path()


    robot_urdf_root_path: str = get_assets_path()


    robot_usd_root_path: str = get_assets_path()


    robot_asset_root_path: str = get_assets_path()


    world_config_root_path: str = get_world_configs_path()


    world_asset_root_path: str = get_assets_path()



    robot_config_absolute_path: Optional[str] = None



    robot_xrdf_absolute_path: Optional[str] = None



    robot_urdf_absolute_path: Optional[str] = None



    robot_usd_absolute_path: Optional[str] = None



    robot_asset_absolute_path: Optional[str] = None



    world_config_absolute_path: Optional[str] = None




    robot_config_file: Optional[str] = None




    robot_xrdf_file: Optional[str] = None




    robot_urdf_file: Optional[str] = None




    robot_usd_file: Optional[str] = None


    robot_asset_subroot_path: Optional[str] = None




    world_config_file: Optional[str] = None

    def __post_init__(self):
        if self.robot_config_file is not None:
            if self.robot_config_absolute_path is not None:
                log_error(
                    "robot_config_file and robot_config_absolute_path cannot be provided together."
                )
            object.__setattr__(
                self,
                "robot_config_absolute_path",
                join_path(self.robot_config_root_path, self.robot_config_file),
            )
        if self.robot_xrdf_file is not None:
            if self.robot_xrdf_absolute_path is not None:
                log_error(
                    "robot_xrdf_file and robot_xrdf_absolute_path cannot be provided together."
                )
            object.__setattr__(
                self,
                "robot_xrdf_absolute_path",
                join_path(self.robot_xrdf_root_path, self.robot_xrdf_file),
            )
        if self.robot_urdf_file is not None:
            if self.robot_urdf_absolute_path is not None:
                log_error(
                    "robot_urdf_file and robot_urdf_absolute_path cannot be provided together."
                )
            object.__setattr__(
                self,
                "robot_urdf_absolute_path",
                join_path(self.robot_urdf_root_path, self.robot_urdf_file),
            )
        if self.robot_usd_file is not None:
            if self.robot_usd_absolute_path is not None:
                log_error("robot_usd_file and robot_usd_absolute_path cannot be provided together.")
            object.__setattr__(
                self,
                "robot_usd_absolute_path",
                join_path(self.robot_usd_root_path, self.robot_usd_file),
            )
        if self.robot_asset_subroot_path is not None:
            if self.robot_asset_absolute_path is not None:
                log_error(
                    "robot_asset_subroot_path and robot_asset_absolute_path cannot be provided together."
                )
            object.__setattr__(
                self,
                "robot_asset_absolute_path",
                join_path(self.robot_asset_root_path, self.robot_asset_subroot_path),
            )

        if self.world_config_file is not None:
            if self.world_config_absolute_path is not None:
                log_error(
                    "world_config_file and world_config_absolute_path cannot be provided together."
                )
            object.__setattr__(
                self,
                "world_config_absolute_path",
                join_path(self.world_config_root_path, self.world_config_file),
            )

    def get_robot_configuration_path(self):
        """Get the robot configuration path."""
        if self.robot_config_absolute_path is None:
            log_info("cuRobo configuration file not found, trying XRDF")
            if self.robot_xrdf_absolute_path is None:
                log_error("No Robot configuration file found")
            else:
                return self.robot_xrdf_absolute_path
        return self.robot_config_absolute_path
