




from __future__ import annotations

import gymnasium as gym
import torch

from isaaclab_tasks.direct.cartpole.cartpole_camera_env import CartpoleCameraEnv, CartpoleRGBCameraEnvCfg


class CartpoleCameraShowcaseEnv(CartpoleCameraEnv):
    cfg: CartpoleRGBCameraEnvCfg

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:


        if isinstance(self.single_action_space, gym.spaces.Box):
            target = self.cfg.action_scale * self.actions

        elif isinstance(self.single_action_space, gym.spaces.Discrete):
            target = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
            target = torch.where(self.actions == 1, -self.cfg.action_scale, target)
            target = torch.where(self.actions == 2, self.cfg.action_scale, target)

        elif isinstance(self.single_action_space, gym.spaces.MultiDiscrete):

            target = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
            target = torch.where(self.actions[:, [0]] == 1, self.cfg.action_scale / 2.0, target)
            target = torch.where(self.actions[:, [0]] == 2, self.cfg.action_scale, target)

            target = torch.where(self.actions[:, [1]] == 0, -target, target)
        else:
            raise NotImplementedError(f"Action space {type(self.single_action_space)} not implemented")


        self._cartpole.set_joint_effort_target(target, joint_ids=self._cart_dof_idx)

    def _get_observations(self) -> dict:

        data_type = "rgb" if "rgb" in self.cfg.tiled_camera.data_types else "depth"
        if "rgb" in self.cfg.tiled_camera.data_types:
            camera_data = self._tiled_camera.data.output[data_type] / 255.0

            mean_tensor = torch.mean(camera_data, dim=(1, 2), keepdim=True)
            camera_data -= mean_tensor
        elif "depth" in self.cfg.tiled_camera.data_types:
            camera_data = self._tiled_camera.data.output[data_type]
            camera_data[camera_data == float("inf")] = 0



        if isinstance(self.single_observation_space["policy"], gym.spaces.Box):
            obs = camera_data


        elif isinstance(self.single_observation_space["policy"], gym.spaces.Tuple):
            obs = (camera_data, self.joint_vel)

        elif isinstance(self.single_observation_space["policy"], gym.spaces.Dict):
            obs = {"joint-velocities": self.joint_vel, "camera": camera_data}
        else:
            raise NotImplementedError(
                f"Observation space {type(self.single_observation_space['policy'])} not implemented"
            )

        observations = {"policy": obs}
        return observations
