




"""
This script demonstrates the base environment concept that combines a scene with an action,
observation and event manager for a floating cube.
"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="This script demonstrates how to use the concept of an Environment.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to spawn.")


AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass






@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Example scene configuration."""


    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", debug_vis=False)


    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 5)),
    )


    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )







class CubeActionTerm(ActionTerm):
    """Simple action term that implements a PD controller to track a target position."""

    _asset: RigidObject
    """The articulation asset on which the action term is applied."""

    def __init__(self, cfg: ActionTermCfg, env: ManagerBasedEnv):

        super().__init__(cfg, env)

        self._raw_actions = torch.zeros(env.num_envs, 3, device=self.device)
        self._processed_actions = torch.zeros(env.num_envs, 3, device=self.device)
        self._vel_command = torch.zeros(self.num_envs, 6, device=self.device)

        self.p_gain = 5.0
        self.d_gain = 0.5

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._raw_actions.shape[1]

    @property
    def raw_actions(self) -> torch.Tensor:

        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations
    """

    def process_actions(self, actions: torch.Tensor):

        self._raw_actions[:] = actions

        self._processed_actions[:] = self._raw_actions[:]

    def apply_actions(self):

        pos_error = self._processed_actions - (self._asset.data.root_pos_w - self._env.scene.env_origins)
        vel_error = -self._asset.data.root_lin_vel_w

        self._vel_command[:, :3] = self.p_gain * pos_error + self.d_gain * vel_error
        self._asset.write_root_velocity_to_sim(self._vel_command)


@configclass
class CubeActionTermCfg(ActionTermCfg):
    """Configuration for the cube action term."""

    class_type: type = CubeActionTerm







def base_position(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""

    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w - env.scene.env_origins







@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = CubeActionTermCfg(asset_name="cube")


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""


        position = ObsTerm(func=base_position, params={"asset_cfg": SceneEntityCfg("cube")})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True


    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
            },
            "asset_cfg": SceneEntityCfg("cube"),
        },
    )







@configclass
class CubeEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""


    scene: MySceneCfg = MySceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5, replicate_physics=True)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""

        self.decimation = 2

        self.sim.dt = 0.01
        self.sim.physics_material = self.scene.terrain.physics_material


def main():
    """Main function."""


    env = ManagerBasedEnv(cfg=CubeEnvCfg())


    target_position = torch.rand(env.num_envs, 3, device=env.device) * 2
    target_position[:, 2] += 2.0

    target_position -= env.scene.env_origins


    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():

            if count % 300 == 0:
                env.reset()
                count = 0


            obs, _ = env.step(target_position)

            error = torch.norm(obs["policy"] - target_position).mean().item()
            print(f"[Step: {count:04d}]: Mean position error: {error:.4f}")

            count += 1


if __name__ == "__main__":

    main()

    simulation_app.close()
