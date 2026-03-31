









"""This module has differentiable layers built from CuRobo's core features for use in Pytorch."""


from dataclasses import dataclass


from curobo.util.logger import log_warn
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig


@dataclass
class CuroboRobotWorldConfig(RobotWorldConfig):
    def __post_init__(self):
        log_warn("CuroboRobotWorldConfig is deprecated, use RobotWorldConfig instead")
        return super().__post_init__()


class CuroboRobotWorld(RobotWorld):
    def __init__(self, config: CuroboRobotWorldConfig):
        log_warn("CuroboRobotWorld is deprecated, use RobotWorld instead")
        return super().__init__(config)
