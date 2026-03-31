




"""Spacemouse controller for SE(2) control."""

import hid
import numpy as np
import threading
import time
import torch
from collections.abc import Callable
from dataclasses import dataclass

from isaaclab.utils.array import convert_to_torch

from ..device_base import DeviceBase, DeviceCfg
from .utils import convert_buffer


@dataclass
class Se2SpaceMouseCfg(DeviceCfg):
    """Configuration for SE2 space mouse devices."""

    v_x_sensitivity: float = 0.8
    v_y_sensitivity: float = 0.4
    omega_z_sensitivity: float = 1.0
    sim_device: str = "cpu"


class Se2SpaceMouse(DeviceBase):
    r"""A space-mouse controller for sending SE(2) commands as delta poses.

    This class implements a space-mouse controller to provide commands to mobile base.
    It uses the `HID-API`_ which interfaces with USD and Bluetooth HID-class devices across multiple platforms.

    The command comprises of the base linear and angular velocity: :math:`(v_x, v_y, \omega_z)`.

    Note:
        The interface finds and uses the first supported device connected to the computer.

    Currently tested for following devices:

    - SpaceMouse Compact: https://3dconnexion.com/de/product/spacemouse-compact/

    .. _HID-API: https://github.com/libusb/hidapi

    """

    def __init__(self, cfg: Se2SpaceMouseCfg):
        """Initialize the spacemouse layer.

        Args:
            cfg: Configuration for the spacemouse device.
        """

        self.v_x_sensitivity = cfg.v_x_sensitivity
        self.v_y_sensitivity = cfg.v_y_sensitivity
        self.omega_z_sensitivity = cfg.omega_z_sensitivity
        self._sim_device = cfg.sim_device

        self._device = hid.device()
        self._find_device()

        self._base_command = np.zeros(3)

        self._additional_callbacks = dict()

        self._thread = threading.Thread(target=self._run_device)
        self._thread.daemon = True
        self._thread.start()

    def __del__(self):
        """Destructor for the class."""
        self._thread.join()

    def __str__(self) -> str:
        """Returns: A string containing the information of joystick."""
        msg = f"Spacemouse Controller for SE(2): {self.__class__.__name__}\n"
        msg += f"\tManufacturer: {self._device.get_manufacturer_string()}\n"
        msg += f"\tProduct: {self._device.get_product_string()}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tRight button: reset command\n"
        msg += "\tMove mouse laterally: move base horizontally in x-y plane\n"
        msg += "\tTwist mouse about z-axis: yaw base about a corresponding axis"
        return msg

    """
    Operations
    """

    def reset(self):

        self._base_command.fill(0.0)

    def add_callback(self, key: str, func: Callable):
        """Add additional functions to bind spacemouse.

        Args:
            key: The keyboard button to check against.
            func: The function to call when key is pressed. The callback function should not
                take any arguments.
        """
        self._additional_callbacks[key] = func

    def advance(self) -> torch.Tensor:
        """Provides the result from spacemouse event state.

        Returns:
            A 3D tensor containing the linear (x,y) and angular velocity (z).
        """
        return convert_to_torch(self._base_command, device=self._sim_device)

    """
    Internal helpers.
    """

    def _find_device(self):
        """Find the device connected to computer."""
        found = False

        for _ in range(5):
            for device in hid.enumerate():
                if device["product_string"] == "SpaceMouse Compact":

                    found = True
                    vendor_id = device["vendor_id"]
                    product_id = device["product_id"]

                    self._device.open(vendor_id, product_id)

            if not found:
                time.sleep(1.0)
            else:
                break

        if not found:
            raise OSError("No device found by SpaceMouse. Is the device connected?")

    def _run_device(self):
        """Listener thread that keeps pulling new messages."""

        while True:

            data = self._device.read(13)
            if data is not None:

                if data[0] == 1:

                    self._base_command[1] = self.v_y_sensitivity * convert_buffer(data[1], data[2])

                    self._base_command[0] = self.v_x_sensitivity * convert_buffer(data[3], data[4])
                elif data[0] == 2:

                    self._base_command[2] = self.omega_z_sensitivity * convert_buffer(data[3], data[4])

                elif data[0] == 3:

                    if data[1] == 1:

                        if "L" in self._additional_callbacks:
                            self._additional_callbacks["L"]

                    if data[1] == 2:

                        self.reset()

                        if "R" in self._additional_callbacks:
                            self._additional_callbacks["R"]
