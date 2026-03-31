







from __future__ import annotations

import time

from isaaclab.app import AppLauncher


def test_kit_start_up_time():
    """Test kit start-up time."""
    start_time = time.time()
    app_launcher = AppLauncher(headless=True).app
    end_time = time.time()
    elapsed_time = end_time - start_time

    assert elapsed_time <= 12.0
