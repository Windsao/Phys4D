from magicsim.Env.Robot.RobotManager import RobotManager
from magicsim.Env.Sensor.OccupancyManager import OccupancyManager
from magicsim.Env.Planner.Planner import Planner
import gymnasium as gym

import torch
import torch.nn.functional as F
import math
import numpy as np


class Dwb(Planner):
    def __init__(
        self,
        max_speed: float = 2.0,
        min_speed: float = -2.0,
        max_steer: float = 0.6,
        min_steer: float = -0.6,
        num_speed_samples: int = 50,
        num_steer_samples: int = 50,
        max_accel: float = 2.0,
        max_steer_rate: float = 2.0,
        dt: float = 0.1,
        robot_manager: RobotManager = None,
        robot_type: str = "mobile",
        robot_name: str = "mobile",
        device: torch.device = torch.device("cpu"),
        occupancy_manager: OccupancyManager = None,
    ):
        self.robot_manager = robot_manager
        self.occupancy_manager = occupancy_manager
        self.robot_name = robot_name
        self.robot_type = robot_type

        self.max_speed = max_speed
        self.min_speed = min_speed
        self.max_steer = max_steer
        self.min_steer = min_steer

        self.num_speed_samples = num_speed_samples
        self.num_steer_samples = num_steer_samples

        self.max_accel = max_accel
        self.max_steer_rate = max_steer_rate
        self.dt = dt

        self.device = torch.device(device)

        self.last_cmd = torch.tensor(
            [0.0, 0.0], dtype=torch.float32, device=self.device
        )

    def generate_occupancy(self, env_ids: torch.Tensor) -> torch.Tensor:
        num_envs = self.occupancy_manager.num_envs

        room_size = 10.0
        half_size = room_size / 2.0

        boundary = [
            -half_size,
            half_size,
            -half_size,
            half_size,
            -2,
            1,
        ]

        boundaries = [boundary] * num_envs

        scan_origin = [0.0, 0.0, 2.5]
        scan_origins = [scan_origin] * num_envs

        print("Generating occupancy maps...")
        grids = self.occupancy_manager.generate(
            origin=scan_origins, boundary=boundaries, type="2d", env_ids=None
        )

        vis_grid = (1 - grids[0]) * 255
        vis_grid = vis_grid.astype(np.uint8)

        return grids[0]

    def dilate_occupancy(
        self,
        occ: torch.Tensor,
        robot_radius: float,
        resolution: float,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        “” inflated costmap 0/1


            occ: [H, W]  [1, H, W]0=>0=
            robot_radius:
            resolution: /


            dilated_occ: [H, W] dtype
        """
        if isinstance(occ, np.ndarray):
            occ = torch.from_numpy(occ)

        if occ.dim() == 2:
            occ2d = occ.unsqueeze(0).unsqueeze(0)
        elif occ.dim() == 3:
            occ2d = occ.unsqueeze(0)
        else:
            raise ValueError(f"occ dim must be 2 or 3, got {occ.shape}")

        device = occ2d.device
        H, W = occ2d.shape[-2], occ2d.shape[-1]

        radius_cells = math.ceil(robot_radius / resolution)
        if radius_cells <= 0:
            return occ.clone()

        ksize = 2 * radius_cells + 1
        ys, xs = torch.meshgrid(
            torch.arange(ksize, device=device),
            torch.arange(ksize, device=device),
            indexing="ij",
        )
        cy, cx = radius_cells, radius_cells
        dist2 = (xs - cx) ** 2 + (ys - cy) ** 2

        kernel = (dist2 <= radius_cells**2).float()
        kernel = kernel.unsqueeze(0).unsqueeze(0)

        occ_bin = (occ2d > 0).float()
        conv = F.conv2d(
            occ_bin,
            kernel,
            padding=radius_cells,
        )
        dilated = (conv > 0).to(occ.dtype)

        dilated = dilated.squeeze(0).squeeze(0)

        return dilated

    @torch.no_grad()
    def sample_controls(
        self,
        robot_state: dict,
        env_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample a batch of (v, steer) command pairs for each env.

        Args:
            robot_state: dict-like with:
                - "base_lin_vel": [N_env, 3] linear velocity in base frame
                - "front_steer":  [N_env, 2] left/right steering angles
            env_ids: 1D tensor of env indices to sample for, shape [B].

        Returns:
            actions: tensor of shape [B, Nv*Ns, 2],
                     where each row contains all (v, steer) pairs
                     for that env.
        """
        device = self.device
        env_ids = env_ids.to(device=device, dtype=torch.long)

        base_lin_vel = robot_state["base_lin_vel"].to(device)
        front_steer = robot_state["front_steer"].to(device)

        v_cur = base_lin_vel[env_ids, 0]

        steer_cur = front_steer[env_ids].mean(dim=-1)

        dt = self.dt

        v_low = torch.clamp(
            v_cur - self.max_accel * dt,
            min=self.min_speed,
            max=self.max_speed,
        )
        v_high = torch.clamp(
            v_cur + self.max_accel * dt,
            min=self.min_speed,
            max=self.max_speed,
        )

        steer_low = torch.clamp(
            steer_cur - self.max_steer_rate * dt,
            min=self.min_steer,
            max=self.max_steer,
        )
        steer_high = torch.clamp(
            steer_cur + self.max_steer_rate * dt,
            min=self.min_steer,
            max=self.max_steer,
        )

        Nv = self.num_speed_samples
        Ns = self.num_steer_samples

        t_v = torch.linspace(0.0, 1.0, Nv, device=device)
        t_s = torch.linspace(0.0, 1.0, Ns, device=device)

        v_samples = v_low.unsqueeze(1) * (1.0 - t_v) + v_high.unsqueeze(1) * t_v

        steer_samples = (
            steer_low.unsqueeze(1) * (1.0 - t_s) + steer_high.unsqueeze(1) * t_s
        )

        V = v_samples.unsqueeze(2).expand(-1, Nv, Ns)
        S = steer_samples.unsqueeze(1).expand(-1, Nv, Ns)

        actions = torch.stack([V, S], dim=-1)
        actions = actions.reshape(actions.shape[0], -1, 2)

        return actions

    def step(self, target_pos, env_ids):
        """
         forward

        Args:
            target_pos:
            env_ids: ID

        Returns:
            torch.Tensor: tensor (len(env_ids), action_dim)
        """

        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(self.device)

        occupancy = self.generate_occupancy(env_ids)
        dilated_occupancy = self.dilate_occupancy(occupancy, 0.5, 0.05)

        vis_grid = (1 - dilated_occupancy) * 255
        vis_grid = vis_grid.cpu().numpy().astype(np.uint8)

        robot_state = self.robot_manager.get_robot_state(noise_flag=False)[0][
            self.robot_name
        ]
        candidates = self.sample_controls(robot_state, env_ids)
        print("candidates: ", candidates)

        if self.robot_name in self.robot_manager.single_action_space:
            action_space = self.robot_manager.single_action_space[self.robot_name]

            action_dim = gym.spaces.utils.flatdim(action_space)
        else:
            action_dim = 2

        num_envs = len(env_ids)
        zero_actions = torch.zeros(
            (num_envs, action_dim), dtype=torch.float32, device=self.device
        )

        return zero_actions

    def update_obstacles(
        self, obstacle_avoidance_path_list, obstacle_ignore_path_list, env_ids
    ):
        pass

    def reset_idx(self, env_ids):
        pass
