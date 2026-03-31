




from isaaclab.assets import DeformableObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.spawners import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

import isaaclab_tasks.manager_based.manipulation.lift.mdp as mdp

from . import joint_pos_env_cfg




from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG







@configclass
class FrankaCubeLiftEnvCfg(joint_pos_env_cfg.FrankaCubeLiftEnvCfg):
    def __post_init__(self):

        super().__post_init__()



        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )


@configclass
class FrankaCubeLiftEnvCfg_PLAY(FrankaCubeLiftEnvCfg):
    def __post_init__(self):

        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        self.observations.policy.enable_corruption = False







@configclass
class FrankaTeddyBearLiftEnvCfg(FrankaCubeLiftEnvCfg):
    def __post_init__(self):

        super().__post_init__()

        self.scene.object = DeformableObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.5, 0, 0.05), rot=(0.707, 0, 0, 0.707)),
            spawn=UsdFileCfg(
                usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Objects/Teddy_Bear/teddy_bear.usd",
                scale=(0.01, 0.01, 0.01),
            ),
        )


        self.scene.robot.actuators["panda_hand"].effort_limit_sim = 50.0
        self.scene.robot.actuators["panda_hand"].stiffness = 40.0
        self.scene.robot.actuators["panda_hand"].damping = 10.0



        self.scene.replicate_physics = False


        self.events.reset_object_position = EventTerm(
            func=mdp.reset_nodal_state_uniform,
            mode="reset",
            params={
                "position_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("object"),
            },
        )




        self.terminations.object_dropping = None
        self.rewards.reaching_object = None
        self.rewards.lifting_object = None
        self.rewards.object_goal_tracking = None
        self.rewards.object_goal_tracking_fine_grained = None
        self.observations.policy.object_position = None
