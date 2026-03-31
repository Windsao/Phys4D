




import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.classic.humanoid.mdp as mdp




from isaaclab_assets.robots.ant import ANT_CFG


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with an ant robot."""


    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )


    robot = ANT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )







@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=[".*"], scale=7.5)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""

        base_height = ObsTerm(func=mdp.base_pos_z)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        base_yaw_roll = ObsTerm(func=mdp.base_yaw_roll)
        base_angle_to_target = ObsTerm(func=mdp.base_angle_to_target, params={"target_pos": (1000.0, 0.0, 0.0)})
        base_up_proj = ObsTerm(func=mdp.base_up_proj)
        base_heading_proj = ObsTerm(func=mdp.base_heading_proj, params={"target_pos": (1000.0, 0.0, 0.0)})
        joint_pos_norm = ObsTerm(func=mdp.joint_pos_limit_normalized)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.2)
        feet_body_forces = ObsTerm(
            func=mdp.body_incoming_wrench,
            scale=0.1,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=["front_left_foot", "front_right_foot", "left_back_foot", "right_back_foot"]
                )
            },
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True


    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}},
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-0.1, 0.1),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""


    progress = RewTerm(func=mdp.progress_reward, weight=1.0, params={"target_pos": (1000.0, 0.0, 0.0)})

    alive = RewTerm(func=mdp.is_alive, weight=0.5)

    upright = RewTerm(func=mdp.upright_posture_bonus, weight=0.1, params={"threshold": 0.93})

    move_to_target = RewTerm(
        func=mdp.move_to_target_bonus, weight=0.5, params={"threshold": 0.8, "target_pos": (1000.0, 0.0, 0.0)}
    )

    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.005)

    energy = RewTerm(func=mdp.power_consumption, weight=-0.05, params={"gear_ratio": {".*": 15.0}})

    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits_penalty_ratio, weight=-0.1, params={"threshold": 0.99, "gear_ratio": {".*": 15.0}}
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""


    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    torso_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.31})


@configclass
class AntEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the MuJoCo-style Ant walking environment."""


    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=5.0, clone_in_fabric=True)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()

    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""

        self.decimation = 2
        self.episode_length_s = 16.0

        self.sim.dt = 1 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2

        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.0
