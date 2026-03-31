






"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher


simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import time

from isaaclab.utils.timer import Timer


PRECISION_PLACES = 2


def test_timer_as_object():
    """Test using a `Timer` as a regular object."""
    timer = Timer()
    timer.start()
    assert abs(0 - timer.time_elapsed) < 10 ** (-PRECISION_PLACES)
    time.sleep(1)
    assert abs(1 - timer.time_elapsed) < 10 ** (-PRECISION_PLACES)
    timer.stop()
    assert abs(1 - timer.total_run_time) < 10 ** (-PRECISION_PLACES)


def test_timer_as_context_manager():
    """Test using a `Timer` as a context manager."""
    with Timer() as timer:
        assert abs(0 - timer.time_elapsed) < 10 ** (-PRECISION_PLACES)
        time.sleep(1)
        assert abs(1 - timer.time_elapsed) < 10 ** (-PRECISION_PLACES)
