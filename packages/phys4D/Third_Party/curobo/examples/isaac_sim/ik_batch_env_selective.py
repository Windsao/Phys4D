











try:

    import isaacsim
except ImportError:
    pass


import torch

a = torch.zeros(4, device="cuda:0")


import os


import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--headless_mode",
    type=str,
    default=None,
    help="To run headless, use one of [native, websocket], webrtc might not work.",
)

parser.add_argument(
    "--visualize_spheres",
    action="store_true",
    help="When True, visualizes robot spheres",
    default=False,
)

parser.add_argument("--robot", type=str, default="franka.yml", help="robot configuration to load")
args = parser.parse_args()

from omni.isaac.kit import SimulationApp

os.environ.setdefault("CUROBO_TORCH_CUDA_GRAPH_RESET", "1")

simulation_app = SimulationApp(
    {
        "headless": args.headless_mode is not None,
        "width": "1920",
        "height": "1080",
    }
)

import carb
import numpy as np
from helper import add_extensions, add_robot_to_scene
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid


from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.kit import SimulationApp


from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig


def draw_points(pose, success):

    try:
        from omni.isaac.debug_draw import _debug_draw
    except ImportError:
        from isaacsim.util.debug_draw import _debug_draw

    draw = _debug_draw.acquire_debug_draw_interface()
    N = 100

    draw.clear_points()
    cpu_pos = pose.position.cpu().numpy()
    b, _ = cpu_pos.shape
    point_list = []
    colors = []
    for i in range(b):

        point_list += [(cpu_pos[i, 0], cpu_pos[i, 1], cpu_pos[i, 2])]
        if success[i].item():
            colors += [(0, 1, 0, 0.25)]
        else:
            colors += [(1, 0, 0, 0.25)]
    sizes = [40.0 for _ in range(b)]

    draw.draw_points(point_list, colors, sizes)


def main():
    usd_help = UsdHelper()
    act_distance = 0.2

    n_envs = 4

    ik_env_ids = [0, 2]
    n_ik_envs = len(ik_env_ids)


    my_world = World(stage_units_in_meters=1.0)
    my_world.scene.add_default_ground_plane()

    stage = my_world.stage
    usd_help.load_stage(stage)
    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")


    target_list = []
    offset_y = 2.5
    radius = 0.1
    pose = Pose.from_list([0, 0, 0, 1, 0, 0, 0])
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), args.robot))["robot_cfg"]

    robot_list = []

    for i in range(n_envs):
        if i > 0:
            pose.position[0, 1] += offset_y
        usd_help.add_subroot("/World", "/World/world_" + str(i), pose)

        target = cuboid.VisualCuboid(
            "/World/world_" + str(i) + "/target",
            position=np.array([0.5, 0, 0.5]) + pose.position[0].cpu().numpy(),
            orientation=np.array([0, 1, 0, 0]),
            color=np.array([1.0, 0, 0]),
            size=0.05,
        )
        target_list.append(target)
        r = add_robot_to_scene(
            robot_cfg,
            my_world,
            "/World/world_" + str(i) + "/",
            robot_name="robot_" + str(i),
            position=pose.position[0].cpu().numpy(),
            initialize_world=False,
        )
        robot_list.append(r[0])
    setup_curobo_logger("warn")
    my_world.initialize_physics()



    tensor_args = TensorDeviceType()
    robot_file = "franka.yml"

    world_file = ["collision_test.yml", "collision_thin_walls.yml", "collision_test.yml", "collision_thin_walls.yml"]
    world_cfg_list = []
    for i in range(n_envs):
        world_cfg = WorldConfig.from_dict(
            load_yaml(join_path(get_world_configs_path(), world_file[i]))
        )
        world_cfg.objects[0].pose[2] -= 0.02
        world_cfg.randomize_color(r=[0.2, 0.3], b=[0.0, 0.05], g=[0.2, 0.3])
        usd_help.add_world_to_stage(world_cfg, base_frame="/World/world_" + str(i))
        world_cfg_list.append(world_cfg)


    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg_list,
        rotation_threshold=0.05,
        position_threshold=0.005,
        num_seeds=20,
        self_collision_check=True,
        self_collision_opt=True,
        tensor_args=tensor_args,
        use_cuda_graph=True,
        collision_checker_type=CollisionCheckerType.MESH,
        collision_cache={"obb": 10, "mesh": 10},
    )
    ik_solver = IKSolver(ik_config)

    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

    print("warming up...")

    add_extensions(simulation_app, args.headless_mode)

    i = 0
    max_distance = 0.5
    env_query_idx = torch.tensor(ik_env_ids, device=tensor_args.device, dtype=torch.int32)
    prev_goal = None
    cmd_plan = [None] * n_envs
    art_controllers = [r.get_articulation_controller() for r in robot_list]
    cmd_idx = 0
    past_goal = None

    while simulation_app.is_running():
        my_world.step(render=True)
        if not my_world.is_playing():
            if i % 100 == 0:
                print("**** Click Play to start simulation *****")
            i += 1
            continue
        step_index = my_world.current_time_step_index

        if step_index <= 10:

            for robot in robot_list:
                robot._articulation_view.initialize()
                idx_list = [robot.get_dof_index(x) for x in j_names]
                robot.set_joint_positions(default_config, idx_list)

                robot._articulation_view.set_max_efforts(
                    values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
                )
        if step_index < 20:
            continue
        sp_buffer = []
        sq_buffer = []
        for k in target_list:
            sph_position, sph_orientation = k.get_local_pose()
            sp_buffer.append(sph_position)
            sq_buffer.append(sph_orientation)

        ik_goal = Pose(
            position=tensor_args.to_device(sp_buffer),
            quaternion=tensor_args.to_device(sq_buffer),
        )
        if prev_goal is None:
            prev_goal = ik_goal.clone()
        if past_goal is None:
            past_goal = ik_goal.clone()
        sim_js_names = robot_list[0].dof_names


        full_js_all = None
        for i in range(n_envs):
            sim_js = robot_list[i].get_joints_state()
            if sim_js is None:
                continue
            if full_js_all is None:
                full_js_all = JointState(
                    position=tensor_args.to_device(sim_js.positions).view(1, -1),
                    velocity=tensor_args.to_device(sim_js.velocities).view(1, -1) * 0.0,
                    acceleration=tensor_args.to_device(sim_js.velocities).view(1, -1) * 0.0,
                    jerk=tensor_args.to_device(sim_js.velocities).view(1, -1) * 0.0,
                    joint_names=sim_js_names,
                )
            else:
                cu_js = JointState(
                    position=tensor_args.to_device(sim_js.positions).view(1, -1),
                    velocity=tensor_args.to_device(sim_js.velocities).view(1, -1) * 0.0,
                    acceleration=tensor_args.to_device(sim_js.velocities).view(1, -1) * 0.0,
                    jerk=tensor_args.to_device(sim_js.velocities).view(1, -1) * 0.0,
                    joint_names=sim_js_names,
                )
                full_js_all = full_js_all.stack(cu_js)

        if full_js_all is None:
            continue


        plan_goal_pose = Pose(
            position=ik_goal.position[ik_env_ids],
            quaternion=ik_goal.quaternion[ik_env_ids],
        )


        prev_goal_plan = Pose(
            position=prev_goal.position[ik_env_ids],
            quaternion=prev_goal.quaternion[ik_env_ids],
        )
        past_goal_plan = Pose(
            position=past_goal.position[ik_env_ids],
            quaternion=past_goal.quaternion[ik_env_ids],
        )

        prev_distance_plan = plan_goal_pose.distance(prev_goal_plan)
        past_distance_plan = plan_goal_pose.distance(past_goal_plan)


        need_ik = (
            (torch.sum(prev_distance_plan[0] > 1e-2) > 0 if prev_distance_plan[0].shape[0] > 0 else False)
            or (torch.sum(prev_distance_plan[1] > 1e-2) > 0 if prev_distance_plan[1].shape[0] > 0 else False)
        ) and (
            (torch.sum(past_distance_plan[0]) == 0.0)
            and (torch.sum(past_distance_plan[1]) == 0.0)
        ) and torch.max(torch.abs(full_js_all.velocity[ik_env_ids])) < 0.2


        need_ik = need_ik and any(cmd_plan[env_id] is None for env_id in ik_env_ids)

        if need_ik:

            result = ik_solver.solve_batch_env(
                plan_goal_pose,
                env_query_idx=env_query_idx
            )

            if torch.count_nonzero(result.success) > 0:

                for idx, env_id in enumerate(ik_env_ids):
                    if result.success[idx]:
                        prev_goal.position[env_id].copy_(plan_goal_pose.position[idx])
                        prev_goal.quaternion[env_id].copy_(plan_goal_pose.quaternion[idx])


                for idx, env_id in enumerate(ik_env_ids):
                    if result.success[idx] and result.js_solution is not None:



                        cmd_plan[env_id] = result.js_solution[idx]


                        idx_list = []
                        common_js_names = []
                        for x in sim_js_names:
                            if x in cmd_plan[env_id].joint_names:
                                idx_list.append(robot_list[env_id].get_dof_index(x))
                                common_js_names.append(x)

                        cmd_plan[env_id] = cmd_plan[env_id].get_ordered_joint_state(common_js_names)

                cmd_idx = 0
                print(
                    f"[Selective IK] Solved IK for envs {ik_env_ids}: "
                    f"success={result.success.cpu().numpy()}, "
                    f"time={result.solve_time:.4f}s"
                )
            else:
                print(
                    f"[Selective IK] IK solving failed for envs {ik_env_ids}"
                )


        for env_id in range(n_envs):
            if cmd_plan[env_id] is not None:


                cmd_state = cmd_plan[env_id]
                if len(cmd_state.position.shape) == 1:

                    position = cmd_state.position.unsqueeze(0)
                    velocity = cmd_state.velocity.unsqueeze(0) if cmd_state.velocity is not None else None
                else:
                    position = cmd_state.position[0]
                    velocity = cmd_state.velocity[0] if cmd_state.velocity is not None else None


                idx_list = []
                for x in sim_js_names:
                    if x in cmd_state.joint_names:
                        idx_list.append(robot_list[env_id].get_dof_index(x))

                art_action = ArticulationAction(
                    position.cpu().numpy(),
                    velocity.cpu().numpy() if velocity is not None else None,
                    joint_indices=idx_list,
                )

                art_controllers[env_id].apply_action(art_action)

                cmd_plan[env_id] = None


        past_goal.copy_(ik_goal)

        for _ in range(2):
            my_world.step(render=False)


if __name__ == "__main__":
    main()

