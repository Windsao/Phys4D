




"""
This script creates a simple environment with a floating cube. The cube is controlled by a PD
controller to track an arbitrary target position.

While going through this tutorial, we recommend you to pay attention to how a custom action term
is defined. The action term is responsible for processing the raw actions and applying them to the
scene entities.

We also define an event term called 'randomize_scale' that randomizes the scale of
the cube. This event term has the mode 'prestartup', which means that it is applied on the USD stage
before the simulation starts. Additionally, the flag 'replicate_physics' is set to False,
which means that the cube is not replicated across multiple environments but rather each
environment gets its own cube instance.

The rest of the environment is similar to the previous tutorials.

.. code-block:: bash

    # Run the script
    ./isaaclab.sh -p scripts/tutorials/03_envs/create_cube_base_env.py --num_envs 32

"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Tutorial on creating a floating cube environment.")
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
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass






class CubeActionTerm(ActionTerm):
    """Simple action term that implements a PD controller to track a target position.

    The action term is applied to the cube asset. It involves two steps:

    1. **Process the raw actions**: Typically, this includes any transformations of the raw actions
       that are required to map them to the desired space. This is called once per environment step.
    2. **Apply the processed actions**: This step applies the processed actions to the asset.
       It is called once per simulation step.

    In this case, the action term simply applies the raw actions to the cube asset. The raw actions
    are the desired target positions of the cube in the environment frame. The pre-processing step
    simply copies the raw actions to the processed actions as no additional processing is required.
    The processed actions are then applied to the cube asset by implementing a PD controller to
    track the target position.
    """

    _asset: RigidObject
    """The articulation asset on which the action term is applied."""

    def __init__(self, cfg: CubeActionTermCfg, env: ManagerBasedEnv):

        super().__init__(cfg, env)

        self._raw_actions = torch.zeros(env.num_envs, 3, device=self.device)
        self._processed_actions = torch.zeros(env.num_envs, 3, device=self.device)
        self._vel_command = torch.zeros(self.num_envs, 6, device=self.device)

        self.p_gain = cfg.p_gain
        self.d_gain = cfg.d_gain

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
    """The class corresponding to the action term."""

    p_gain: float = 5.0
    """Proportional gain of the PD controller."""
    d_gain: float = 0.5
    """Derivative gain of the PD controller."""







def base_position(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""

    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w - env.scene.env_origins







@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Example scene configuration.

    The scene comprises of a ground plane, light source and floating cubes (gravity disabled).
    """


    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", debug_vis=False)


    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 5)),
    )


    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2000.0),
    )







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





    randomize_scale = EventTerm(
        func=mdp.randomize_rigid_body_scale,
        mode="prestartup",
        params={
            "scale_range": {"x": (0.5, 1.5), "y": (0.5, 1.5), "z": (0.5, 1.5)},
            "asset_cfg": SceneEntityCfg("cube"),
        },
    )




    randomize_color = EventTerm(
        func=mdp.randomize_visual_color,
        mode="prestartup",
        params={
            "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
            "asset_cfg": SceneEntityCfg("cube"),
            "mesh_name": "geometry/mesh",
            "event_name": "rep_cube_randomize_color",
        },
    )







@configclass
class CubeEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""





    scene: MySceneCfg = MySceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5, replicate_physics=False)


    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""

        self.decimation = 2

        self.sim.dt = 0.01
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.render_interval = 2
        self.sim.device = args_cli.device

        self.viewer.eye = (5.0, 5.0, 5.0)
        self.viewer.lookat = (0.0, 0.0, 2.0)


def main():
    """Main function."""


    env_cfg = CubeEnvCfg()
    env = ManagerBasedEnv(cfg=env_cfg)


    target_position = torch.rand(env.num_envs, 3, device=env.device) * 2
    target_position[:, 2] += 2.0

    target_position -= env.scene.env_origins


    count = 0
    obs, _ = env.reset()
    while simulation_app.is_running():
        with torch.inference_mode():

            if count % 300 == 0:
                count = 0
                obs, _ = env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")

            obs, _ = env.step(target_position)

            error = torch.norm(obs["policy"] - target_position).mean().item()
            print(f"[Step: {count:04d}]: Mean position error: {error:.4f}")

            count += 1


    env.close()


if __name__ == "__main__":

    main()

    simulation_app.close()
