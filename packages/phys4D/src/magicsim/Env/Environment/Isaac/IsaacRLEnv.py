"""
This Environment is DirectRLEnv in IsaacLab. In magicsim, it serves as the simulation backbone, it will provide function such as sim_step /sim_reset
The Environment Server as the the backbone of sim just like the world class in isaacsim.
Though this file, we will set physics and render setting for the simulation backbone by setting isaacRLCfg. This file will launched by BaseEnv.py
All the action of this env should be atomic action, meaning that it will be called everytime before sim_step.
"""

from typing import Sequence

import torch
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg
from isaaclab.utils import configclass
from magicsim.Env.Environment.Isaac.DirectRLEnv import CustomDirectRLEnv


@configclass
class IsaacRLEnvCfg(DirectRLEnvCfg):
    decimation: int = 3
    episode_length_s: float = 2000 * (1 / 60) * decimation
    observation_space: int = 0
    action_space: int = 0
    state_space: int = 0

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=4.0,
    )
    sim: SimulationCfg = SimulationCfg(
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(),
    )
    ground: GroundPlaneCfg = GroundPlaneCfg(semantic_tags=[("class", "ground")])

    GROUND_PRIM_PATH = "/World/ground"


class IsaacRLEnv(CustomDirectRLEnv):
    """An RL environment defined with the direct workflow for Isaac Sim."""

    cfg: IsaacRLEnvCfg

    def __init__(self, cfg: IsaacRLEnvCfg, render_mode: str | None = None, **kwargs):
        self.func = kwargs.get("func", None)
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):
        """This is the entry point for spawn all isaaclab object into the scene.
        Here we use isaaclab to spawn ground plane. In Robotbase env we spawn a robot arm and a desk
        You should add all object you want isaaclab to spawn in this function.
        This function will be called by the DirectRLEnv.__init__() function.
        """
        self.func["_setup_scene"](self)
        self.scene.clone_environments(copy_from_source=True)
        self.func["_post_setup_scene"](self)

    def _get_observations(self):
        """
        This function will never be called because we do not execute it in direct_rl_env.step. We just use step function as sim backbone
        """
        pass

    def _get_dones(self):
        """
        This function will never be called because we do not execute it in direct_rl_env.step. We just use step function as sim backbone
        """
        pass

    def _get_states(self):
        """
        This function will never be called because we do not execute it in direct_rl_env.step. We just use step function as sim backbone
        """
        pass

    def _get_rewards(self):
        """
        This function will never be called because we do not execute it in direct_rl_env.step. We just use step function as sim backbone
        """
        pass

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        """
        This is soft reset for robotics arm interface and other isaaclab related things. This function wil not call simulation backend reset. It is only a soft reset entry point for robotics arm.
        This function is called by our own reset in robot base env.
        """
        super()._reset_idx(env_ids)
        self.func["_reset_idx"](self, env_ids)

    def _apply_action(self):
        """
        This is apply actions function for robotics arm and is called by directrlenv.step(action). This serve as the simulation backbone function to control the robot arm.

        ! Important Function !: This function will be called by the step function in the robot base env.
        ! Important Function !: SIMD Entry Point
        """
        self.func["_apply_action"](self)

    def _pre_physics_step(self, actions: torch.Tensor, env_ids: torch.Tensor):
        """
        This is pre physics step function for robotics arm interface and is called by directrlenv.step(action). This serve as the simulation backbone function to control the robot arm.
        ! Important Function !: This function will be called by the step function in the robot base env.
        ! Important Function !: SIMD Entry Point
        """
        self.func["_pre_physics_step"](self, actions, env_ids)
