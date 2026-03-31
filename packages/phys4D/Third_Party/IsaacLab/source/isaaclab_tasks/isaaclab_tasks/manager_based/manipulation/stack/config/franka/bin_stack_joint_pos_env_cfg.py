





import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.stack import mdp
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events
from isaaclab_tasks.manager_based.manipulation.stack.stack_env_cfg import StackEnvCfg




from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG


@configclass
class EventCfg:
    """Configuration for events."""

    init_franka_arm_pose = EventTerm(
        func=franka_stack_events.set_default_joint_pose,

        mode="reset",
        params={
            "default_pose": [0.0444, -0.1894, -0.1107, -2.5148, 0.0044, 2.3775, 0.6952, 0.0400, 0.0400],
        },
    )

    randomize_franka_joint_state = EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


    reset_blue_bin_pose = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={

            "pose_range": {"x": (0.4, 0.4), "y": (0.0, 0.0), "z": (0.0203, 0.0203), "yaw": (0.0, 0.0)},
            "min_separation": 0.0,
            "asset_cfgs": [SceneEntityCfg("blue_sorting_bin")],
        },
    )


    reset_cube_1_pose = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.4, 0.4), "y": (0.0, 0.0), "z": (0.0203, 0.0203), "yaw": (0.0, 0.0)},
            "min_separation": 0.0,
            "asset_cfgs": [SceneEntityCfg("cube_1")],
        },
    )


    reset_cube_pose = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.65, 0.70), "y": (-0.18, 0.18), "z": (0.0203, 0.0203), "yaw": (-1.0, 1.0, 0)},
            "min_separation": 0.1,
            "asset_cfgs": [SceneEntityCfg("cube_2"), SceneEntityCfg("cube_3")],
        },
    )


@configclass
class FrankaBinStackEnvCfg(StackEnvCfg):
    def __post_init__(self):

        super().__post_init__()


        self.events = EventCfg()


        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]


        self.scene.table.spawn.semantic_tags = [("class", "table")]


        self.scene.plane.semantic_tags = [("class", "ground")]


        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )

        self.gripper_joint_names = ["panda_finger_.*"]
        self.gripper_open_val = 0.04
        self.gripper_threshold = 0.005


        cube_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=40,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )


        self.scene.blue_sorting_bin = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/BlueSortingBin",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.0203), rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=UsdFileCfg(
                usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/nut_pour_task/nut_pour_assets/sorting_bin_blue.usd",
                scale=(1.1, 1.6, 3.3),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            ),
        )



        self.scene.cube_1 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_1",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.025), rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=cube_properties,
            ),
        )


        self.scene.cube_2 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_2",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.85, 0.25, 0.0203), rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=cube_properties,
            ),
        )


        self.scene.cube_3 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_3",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.85, -0.25, 0.0203), rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/green_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=cube_properties,
            ),
        )


        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.0),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
            ],
        )
