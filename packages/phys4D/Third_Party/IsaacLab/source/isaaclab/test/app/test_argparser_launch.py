




import argparse

import pytest

from isaaclab.app import AppLauncher


@pytest.mark.usefixtures("mocker")
def test_livestream_launch_with_argparser(mocker):
    """Test launching with argparser arguments."""

    mocker.patch("argparse.ArgumentParser.parse_args", return_value=argparse.Namespace(livestream=1, headless=True))

    parser = argparse.ArgumentParser()

    AppLauncher.add_app_launcher_args(parser)

    for name in AppLauncher._APPLAUNCHER_CFG_INFO:
        assert parser._option_string_actions[f"--{name}"]

    mock_args = parser.parse_args()

    app = AppLauncher(mock_args).app


    import carb


    carb_settings_iface = carb.settings.get_settings()


    assert carb_settings_iface.get("/app/window/enabled") is False

    assert carb_settings_iface.get("/app/livestream/enabled") is True


    app.close()
