




import os

import pytest

from isaaclab.app import AppLauncher


@pytest.mark.usefixtures("mocker")
def test_livestream_launch_with_env_vars(mocker):
    """Test launching with environment variables."""

    mocker.patch.dict(os.environ, {"LIVESTREAM": "1", "HEADLESS": "1"})

    app = AppLauncher().app


    import carb


    carb_settings_iface = carb.settings.get_settings()


    assert carb_settings_iface.get("/app/window/enabled") is False

    assert carb_settings_iface.get("/app/livestream/enabled") is True


    app.close()
