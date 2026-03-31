




"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher


simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import importlib
import torch

import pytest


from isaaclab.devices import (
    OpenXRDevice,
    OpenXRDeviceCfg,
    Se2Gamepad,
    Se2GamepadCfg,
    Se2Keyboard,
    Se2KeyboardCfg,
    Se2SpaceMouse,
    Se2SpaceMouseCfg,
    Se3Gamepad,
    Se3GamepadCfg,
    Se3Keyboard,
    Se3KeyboardCfg,
    Se3SpaceMouse,
    Se3SpaceMouseCfg,
)
from isaaclab.devices.openxr import XrCfg
from isaaclab.devices.openxr.retargeters import GripperRetargeterCfg, Se3AbsRetargeterCfg


from isaaclab.devices.teleop_device_factory import create_teleop_device


@pytest.fixture
def mock_environment(mocker):
    """Set up common mock objects for tests."""

    carb_mock = mocker.MagicMock()
    omni_mock = mocker.MagicMock()
    appwindow_mock = mocker.MagicMock()
    keyboard_mock = mocker.MagicMock()
    gamepad_mock = mocker.MagicMock()
    input_mock = mocker.MagicMock()
    settings_mock = mocker.MagicMock()
    hid_mock = mocker.MagicMock()
    device_mock = mocker.MagicMock()


    omni_mock.appwindow.get_default_app_window.return_value = appwindow_mock
    appwindow_mock.get_keyboard.return_value = keyboard_mock
    appwindow_mock.get_gamepad.return_value = gamepad_mock
    carb_mock.input.acquire_input_interface.return_value = input_mock
    carb_mock.settings.get_settings.return_value = settings_mock


    carb_mock.input.KeyboardEventType.KEY_PRESS = 1
    carb_mock.input.KeyboardEventType.KEY_RELEASE = 2


    hid_mock.enumerate.return_value = [{"product_string": "SpaceMouse Compact", "vendor_id": 123, "product_id": 456}]
    hid_mock.device.return_value = device_mock



    message_bus_mock = mocker.MagicMock()
    singleton_mock = mocker.MagicMock()
    omni_mock.kit.xr.core.XRCore.get_singleton.return_value = singleton_mock
    singleton_mock.get_message_bus.return_value = message_bus_mock
    omni_mock.kit.xr.core.XRPoseValidityFlags.POSITION_VALID = 1
    omni_mock.kit.xr.core.XRPoseValidityFlags.ORIENTATION_VALID = 2

    return {
        "carb": carb_mock,
        "omni": omni_mock,
        "appwindow": appwindow_mock,
        "keyboard": keyboard_mock,
        "gamepad": gamepad_mock,
        "input": input_mock,
        "settings": settings_mock,
        "hid": hid_mock,
        "device": device_mock,
    }


"""
Test keyboard devices.
"""


def test_se2keyboard_constructors(mock_environment, mocker):
    """Test constructor for Se2Keyboard."""

    config = Se2KeyboardCfg(
        v_x_sensitivity=0.9,
        v_y_sensitivity=0.5,
        omega_z_sensitivity=1.2,
    )
    device_mod = importlib.import_module("isaaclab.devices.keyboard.se2_keyboard")
    mocker.patch.dict("sys.modules", {"carb": mock_environment["carb"], "omni": mock_environment["omni"]})
    mocker.patch.object(device_mod, "carb", mock_environment["carb"])
    mocker.patch.object(device_mod, "omni", mock_environment["omni"])

    keyboard = Se2Keyboard(config)


    assert keyboard.v_x_sensitivity == 0.9
    assert keyboard.v_y_sensitivity == 0.5
    assert keyboard.omega_z_sensitivity == 1.2


    result = keyboard.advance()
    assert isinstance(result, torch.Tensor)
    assert result.shape == (3,)


def test_se3keyboard_constructors(mock_environment, mocker):
    """Test constructor for Se3Keyboard."""

    config = Se3KeyboardCfg(
        pos_sensitivity=0.5,
        rot_sensitivity=0.9,
    )
    device_mod = importlib.import_module("isaaclab.devices.keyboard.se3_keyboard")
    mocker.patch.dict("sys.modules", {"carb": mock_environment["carb"], "omni": mock_environment["omni"]})
    mocker.patch.object(device_mod, "carb", mock_environment["carb"])
    mocker.patch.object(device_mod, "omni", mock_environment["omni"])

    keyboard = Se3Keyboard(config)


    assert keyboard.pos_sensitivity == 0.5
    assert keyboard.rot_sensitivity == 0.9


    result = keyboard.advance()
    assert isinstance(result, torch.Tensor)
    assert result.shape == (7,)


"""
Test gamepad devices.
"""


def test_se2gamepad_constructors(mock_environment, mocker):
    """Test constructor for Se2Gamepad."""

    config = Se2GamepadCfg(
        v_x_sensitivity=1.1,
        v_y_sensitivity=0.6,
        omega_z_sensitivity=1.2,
        dead_zone=0.02,
    )
    device_mod = importlib.import_module("isaaclab.devices.gamepad.se2_gamepad")
    mocker.patch.dict("sys.modules", {"carb": mock_environment["carb"], "omni": mock_environment["omni"]})
    mocker.patch.object(device_mod, "carb", mock_environment["carb"])
    mocker.patch.object(device_mod, "omni", mock_environment["omni"])

    gamepad = Se2Gamepad(config)


    assert gamepad.v_x_sensitivity == 1.1
    assert gamepad.v_y_sensitivity == 0.6
    assert gamepad.omega_z_sensitivity == 1.2
    assert gamepad.dead_zone == 0.02


    result = gamepad.advance()
    assert isinstance(result, torch.Tensor)
    assert result.shape == (3,)


def test_se3gamepad_constructors(mock_environment, mocker):
    """Test constructor for Se3Gamepad."""

    config = Se3GamepadCfg(
        pos_sensitivity=1.1,
        rot_sensitivity=1.7,
        dead_zone=0.02,
    )
    device_mod = importlib.import_module("isaaclab.devices.gamepad.se3_gamepad")
    mocker.patch.dict("sys.modules", {"carb": mock_environment["carb"], "omni": mock_environment["omni"]})
    mocker.patch.object(device_mod, "carb", mock_environment["carb"])
    mocker.patch.object(device_mod, "omni", mock_environment["omni"])

    gamepad = Se3Gamepad(config)


    assert gamepad.pos_sensitivity == 1.1
    assert gamepad.rot_sensitivity == 1.7
    assert gamepad.dead_zone == 0.02


    result = gamepad.advance()
    assert isinstance(result, torch.Tensor)
    assert result.shape == (7,)


"""
Test spacemouse devices.
"""


def test_se2spacemouse_constructors(mock_environment, mocker):
    """Test constructor for Se2SpaceMouse."""

    config = Se2SpaceMouseCfg(
        v_x_sensitivity=0.9,
        v_y_sensitivity=0.5,
        omega_z_sensitivity=1.2,
    )
    device_mod = importlib.import_module("isaaclab.devices.spacemouse.se2_spacemouse")
    mocker.patch.dict("sys.modules", {"hid": mock_environment["hid"]})
    mocker.patch.object(device_mod, "hid", mock_environment["hid"])

    spacemouse = Se2SpaceMouse(config)


    assert spacemouse.v_x_sensitivity == 0.9
    assert spacemouse.v_y_sensitivity == 0.5
    assert spacemouse.omega_z_sensitivity == 1.2


    mock_environment["device"].read.return_value = [1, 0, 0, 0, 0]
    result = spacemouse.advance()
    assert isinstance(result, torch.Tensor)
    assert result.shape == (3,)


def test_se3spacemouse_constructors(mock_environment, mocker):
    """Test constructor for Se3SpaceMouse."""

    config = Se3SpaceMouseCfg(
        pos_sensitivity=0.5,
        rot_sensitivity=0.9,
    )
    device_mod = importlib.import_module("isaaclab.devices.spacemouse.se3_spacemouse")
    mocker.patch.dict("sys.modules", {"hid": mock_environment["hid"]})
    mocker.patch.object(device_mod, "hid", mock_environment["hid"])

    spacemouse = Se3SpaceMouse(config)


    assert spacemouse.pos_sensitivity == 0.5
    assert spacemouse.rot_sensitivity == 0.9


    mock_environment["device"].read.return_value = [1, 0, 0, 0, 0, 0, 0]
    result = spacemouse.advance()
    assert isinstance(result, torch.Tensor)
    assert result.shape == (7,)


"""
Test OpenXR devices.
"""


def test_openxr_constructors(mock_environment, mocker):
    """Test constructor for OpenXRDevice."""

    xr_cfg = XrCfg(
        anchor_pos=(1.0, 2.0, 3.0),
        anchor_rot=(0.0, 0.1, 0.2, 0.3),
        near_plane=0.2,
    )
    config = OpenXRDeviceCfg(xr_cfg=xr_cfg)


    mock_controller_retargeter = mocker.MagicMock()
    mock_head_retargeter = mocker.MagicMock()
    retargeters = [mock_controller_retargeter, mock_head_retargeter]

    device_mod = importlib.import_module("isaaclab.devices.openxr.openxr_device")
    mocker.patch.dict(
        "sys.modules",
        {
            "carb": mock_environment["carb"],
            "omni.kit.xr.core": mock_environment["omni"].kit.xr.core,
            "isaacsim.core.prims": mocker.MagicMock(),
        },
    )
    mocker.patch.object(device_mod, "XRCore", mock_environment["omni"].kit.xr.core.XRCore)
    mocker.patch.object(device_mod, "XRPoseValidityFlags", mock_environment["omni"].kit.xr.core.XRPoseValidityFlags)
    mock_single_xform = mocker.patch.object(device_mod, "SingleXFormPrim")


    mock_instance = mock_single_xform.return_value
    mock_instance.prim_path = "/XRAnchor"


    device = OpenXRDevice(config)


    assert device._xr_cfg == xr_cfg


    device = OpenXRDevice(cfg=config, retargeters=retargeters)


    assert device._retargeters == retargeters


    device = OpenXRDevice(cfg=config, retargeters=retargeters)


    assert device._xr_cfg == xr_cfg
    assert device._retargeters == retargeters


    device.reset()


"""
Test teleop device factory.
"""


def test_create_teleop_device_basic(mock_environment, mocker):
    """Test creating devices using the teleop device factory."""

    keyboard_cfg = Se3KeyboardCfg(pos_sensitivity=0.8, rot_sensitivity=1.2)


    devices_cfg = {"test_keyboard": keyboard_cfg}


    device_mod = importlib.import_module("isaaclab.devices.keyboard.se3_keyboard")
    mocker.patch.dict("sys.modules", {"carb": mock_environment["carb"], "omni": mock_environment["omni"]})
    mocker.patch.object(device_mod, "carb", mock_environment["carb"])
    mocker.patch.object(device_mod, "omni", mock_environment["omni"])


    device = create_teleop_device("test_keyboard", devices_cfg)


    assert isinstance(device, Se3Keyboard)
    assert device.pos_sensitivity == 0.8
    assert device.rot_sensitivity == 1.2


def test_create_teleop_device_with_callbacks(mock_environment, mocker):
    """Test creating device with callbacks."""

    xr_cfg = XrCfg(anchor_pos=(0.0, 0.0, 0.0), anchor_rot=(1.0, 0.0, 0.0, 0.0), near_plane=0.15)
    openxr_cfg = OpenXRDeviceCfg(xr_cfg=xr_cfg)


    devices_cfg = {"test_xr": openxr_cfg}


    button_a_callback = mocker.MagicMock()
    button_b_callback = mocker.MagicMock()
    callbacks = {"button_a": button_a_callback, "button_b": button_b_callback}


    device_mod = importlib.import_module("isaaclab.devices.openxr.openxr_device")
    mocker.patch.dict(
        "sys.modules",
        {
            "carb": mock_environment["carb"],
            "omni.kit.xr.core": mock_environment["omni"].kit.xr.core,
            "isaacsim.core.prims": mocker.MagicMock(),
        },
    )
    mocker.patch.object(device_mod, "XRCore", mock_environment["omni"].kit.xr.core.XRCore)
    mocker.patch.object(device_mod, "XRPoseValidityFlags", mock_environment["omni"].kit.xr.core.XRPoseValidityFlags)
    mock_single_xform = mocker.patch.object(device_mod, "SingleXFormPrim")


    mock_instance = mock_single_xform.return_value
    mock_instance.prim_path = "/XRAnchor"


    device = create_teleop_device("test_xr", devices_cfg, callbacks)


    assert isinstance(device, OpenXRDevice)


    device.add_callback("button_a", button_a_callback)
    device.add_callback("button_b", button_b_callback)
    assert len(device._additional_callbacks) == 2


def test_create_teleop_device_with_retargeters(mock_environment, mocker):
    """Test creating device with retargeters."""

    retargeter_cfg1 = Se3AbsRetargeterCfg()
    retargeter_cfg2 = GripperRetargeterCfg()


    xr_cfg = XrCfg()
    device_cfg = OpenXRDeviceCfg(xr_cfg=xr_cfg, retargeters=[retargeter_cfg1, retargeter_cfg2])


    devices_cfg = {"test_xr": device_cfg}


    device_mod = importlib.import_module("isaaclab.devices.openxr.openxr_device")
    mocker.patch.dict(
        "sys.modules",
        {
            "carb": mock_environment["carb"],
            "omni.kit.xr.core": mock_environment["omni"].kit.xr.core,
            "isaacsim.core.prims": mocker.MagicMock(),
        },
    )
    mocker.patch.object(device_mod, "XRCore", mock_environment["omni"].kit.xr.core.XRCore)
    mocker.patch.object(device_mod, "XRPoseValidityFlags", mock_environment["omni"].kit.xr.core.XRPoseValidityFlags)
    mock_single_xform = mocker.patch.object(device_mod, "SingleXFormPrim")


    mock_instance = mock_single_xform.return_value
    mock_instance.prim_path = "/XRAnchor"


    retargeter_mod = importlib.import_module("isaaclab.devices.openxr.retargeters")
    mocker.patch.object(retargeter_mod, "Se3AbsRetargeter")
    mocker.patch.object(retargeter_mod, "GripperRetargeter")


    device = create_teleop_device("test_xr", devices_cfg)


    assert len(device._retargeters) == 2


def test_create_teleop_device_device_not_found():
    """Test error when device name is not found in configuration."""

    devices_cfg = {"keyboard": Se3KeyboardCfg()}


    with pytest.raises(ValueError, match="Device 'gamepad' not found"):
        create_teleop_device("gamepad", devices_cfg)


def test_create_teleop_device_unsupported_config():
    """Test error when device configuration type is not supported."""


    class UnsupportedCfg:
        pass


    devices_cfg = {"unsupported": UnsupportedCfg()}


    with pytest.raises(ValueError, match="Unsupported device configuration type"):
        create_teleop_device("unsupported", devices_cfg)
