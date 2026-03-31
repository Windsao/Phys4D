




"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher


app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import torch
from collections.abc import Sequence
from dataclasses import dataclass

import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
import pytest

import isaaclab.sim as sim_utils
from isaaclab.sensors import SensorBase, SensorBaseCfg
from isaaclab.utils import configclass


@dataclass
class DummyData:
    count: torch.Tensor = None


class DummySensor(SensorBase):

    def __init__(self, cfg):
        super().__init__(cfg)
        self._data = DummyData()

    def _initialize_impl(self):
        super()._initialize_impl()
        self._data.count = torch.zeros((self._num_envs), dtype=torch.int, device=self.device)

    @property
    def data(self):

        self._update_outdated_buffers()

        return self._data

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        self._data.count[env_ids] += 1

    def reset(self, env_ids: Sequence[int] | None = None):
        super().reset(env_ids=env_ids)

        if env_ids is None:
            env_ids = slice(None)
        self._data.count[env_ids] = 0


@configclass
class DummySensorCfg(SensorBaseCfg):
    class_type = DummySensor

    prim_path = "/World/envs/env_.*/Cube/dummy_sensor"


def _populate_scene():
    """"""


    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    cfg = sim_utils.SphereLightCfg()
    cfg.func("/World/Light/GreySphere", cfg, translation=(4.5, 3.5, 10.0))
    cfg.func("/World/Light/WhiteSphere", cfg, translation=(-4.5, 3.5, 10.0))


    for i in range(5):
        _ = prim_utils.create_prim(
            f"/World/envs/env_{i:02d}/Cube",
            "Cube",
            translation=(i * 1.0, 0.0, 0.0),
            scale=(0.25, 0.25, 0.25),
        )


@pytest.fixture
def create_dummy_sensor(request, device):


    stage_utils.create_new_stage()


    dt = 0.01

    sim_cfg = sim_utils.SimulationCfg(dt=dt, device=device)
    sim = sim_utils.SimulationContext(sim_cfg)


    _populate_scene()

    sensor_cfg = DummySensorCfg()

    stage_utils.update_stage()

    yield sensor_cfg, sim, dt



    sim._timeline.stop()

    sim.clear_all_callbacks()
    sim.clear_instance()


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_sensor_init(create_dummy_sensor, device):
    """Test that the sensor initializes, steps without update, and forces update."""

    sensor_cfg, sim, dt = create_dummy_sensor
    sensor = DummySensor(cfg=sensor_cfg)


    sim.step()

    sim.reset()

    assert sensor.is_initialized
    assert int(sensor.num_instances) == 5


    for i in range(10):
        sim.step()
        sensor.update(dt=dt, force_recompute=True)
        expected_value = i + 1
        torch.testing.assert_close(
            sensor.data.count,
            torch.tensor(expected_value, device=device, dtype=torch.int32).repeat(sensor.num_instances),
        )
    assert sensor.data.count.shape[0] == 5


    for _ in range(5):
        sim.step()
        sensor.update(dt=dt, force_recompute=False)
        torch.testing.assert_close(
            sensor._data.count,
            torch.tensor(expected_value, device=device, dtype=torch.int32).repeat(sensor.num_instances),
        )


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_sensor_update_rate(create_dummy_sensor, device):
    """Test that the update_rate configuration parameter works by checking the value of the data is old for an update
    period of 2.
    """
    sensor_cfg, sim, dt = create_dummy_sensor
    sensor_cfg.update_period = 2 * dt
    sensor = DummySensor(cfg=sensor_cfg)


    sim.step()

    sim.reset()

    assert sensor.is_initialized
    assert int(sensor.num_instances) == 5
    expected_value = 1
    for i in range(10):
        sim.step()
        sensor.update(dt=dt, force_recompute=True)

        torch.testing.assert_close(
            sensor.data.count,
            torch.tensor(expected_value, device=device, dtype=torch.int32).repeat(sensor.num_instances),
        )
        expected_value += i % 2


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_sensor_reset(create_dummy_sensor, device):
    """Test that sensor can be reset for all or partial env ids."""
    sensor_cfg, sim, dt = create_dummy_sensor
    sensor = DummySensor(cfg=sensor_cfg)


    sim.step()
    sim.reset()

    assert sensor.is_initialized
    assert int(sensor.num_instances) == 5
    for i in range(5):
        sim.step()
        sensor.update(dt=dt)

        torch.testing.assert_close(
            sensor.data.count,
            torch.tensor(i + 1, device=device, dtype=torch.int32).repeat(sensor.num_instances),
        )

    sensor.reset()

    for j in range(5):
        sim.step()
        sensor.update(dt=dt)

        torch.testing.assert_close(
            sensor.data.count,
            torch.tensor(j + 1, device=device, dtype=torch.int32).repeat(sensor.num_instances),
        )

    reset_ids = [2, 4]
    cont_ids = [0, 1, 3]
    sensor.reset(env_ids=reset_ids)

    for k in range(5):
        sim.step()
        sensor.update(dt=dt)

        torch.testing.assert_close(
            sensor.data.count[reset_ids],
            torch.tensor(k + 1, device=device, dtype=torch.int32).repeat(len(reset_ids)),
        )
        torch.testing.assert_close(
            sensor.data.count[cont_ids],
            torch.tensor(k + 6, device=device, dtype=torch.int32).repeat(len(cont_ids)),
        )
