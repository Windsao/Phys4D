




import pytest

from isaaclab.app import AppLauncher


@pytest.mark.usefixtures("mocker")
def test_livestream_launch_with_kwargs(mocker):
    """Test launching with keyword arguments."""

    app = AppLauncher(headless=True, livestream=1).app


    import carb


    carb_settings_iface = carb.settings.get_settings()


    assert carb_settings_iface.get("/app/window/enabled") is False

    assert carb_settings_iface.get("/app/livestream/enabled") is True


    app.close()
