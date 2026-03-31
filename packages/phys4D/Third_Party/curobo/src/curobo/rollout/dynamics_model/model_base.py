










from abc import ABC, abstractmethod


class DynamicsModelBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, start_state, act_seq, *args):
        pass

    @abstractmethod
    def get_next_state(self, currend_state, act, dt):
        pass

    @abstractmethod
    def filter_robot_state(self, current_state):
        pass

    @abstractmethod
    def get_robot_command(self, current_state, act_seq):
        pass
