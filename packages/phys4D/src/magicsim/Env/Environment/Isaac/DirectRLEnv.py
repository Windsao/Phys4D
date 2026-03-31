import torch

from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.envs.direct_rl_env import DirectRLEnv


class CustomDirectRLEnv(DirectRLEnv):
    def sim_step(self):
        """
        This function is used to step the simulation backend
        ! Important Function !: simulation backend step function.
        """

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1

            self.scene.write_data_to_sim()

            self.sim.app.update()

            self.scene.update(dt=self.physics_dt)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        return self.extras

    def step(self, action: torch.Tensor, env_ids: torch.Tensor) -> VecEnvStepReturn:
        """
        Compared to DirectRLEnv.step(), this function won't automatically reset environments that is successful or timed-out.
        In this fucntion, we won't call _get_dones, _get_rewards and _get_observations.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            The extras info of original isaaclab directenv
        """
        action = action.to(self.device)
        env_ids = env_ids.to(self.device)

        self._pre_physics_step(action, env_ids)

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1

            self._apply_action()

            self.scene.write_data_to_sim()

            self.sim.app.update()

            self.scene.update(dt=self.physics_dt)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        return self.extras

    def get_observations(self):
        return self._get_observations()

    def get_dones(self):
        return self._get_dones()
