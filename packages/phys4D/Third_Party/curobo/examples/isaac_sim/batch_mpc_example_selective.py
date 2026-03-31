












try:

    import isaacsim
except ImportError:
    pass



import torch

a = torch.zeros(4, device="cuda:0")


import argparse
import time



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

simulation_app = SimulationApp(
    {
        "headless": args.headless_mode is not None,
        "width": "1920",
        "height": "1080",
    }
)




import os


import carb
import numpy as np
from helper import add_robot_to_scene
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid
from omni.isaac.core.utils.types import ArticulationAction


from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper








EXT_DIR = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__))))
DATA_DIR = os.path.join(EXT_DIR, "data")




from typing import Optional


from helper import add_extensions, add_robot_to_scene



from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.types.state import JointState
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig




def draw_points(rollouts: torch.Tensor):
    if rollouts is None:
        return

    import random


    try:
        from omni.isaac.debug_draw import _debug_draw
    except ImportError:
        from isaacsim.util.debug_draw import _debug_draw
    draw = _debug_draw.acquire_debug_draw_interface()
    N = 100

    draw.clear_points()
    cpu_rollouts = rollouts.cpu().numpy()
    b, h, _ = cpu_rollouts.shape
    point_list = []
    colors = []
    for i in range(b):

        point_list += [
            (cpu_rollouts[i, j, 0], cpu_rollouts[i, j, 1], cpu_rollouts[i, j, 2]) for j in range(h)
        ]
        colors += [(1.0 - (i + 1.0 / b), 0.3 * (i + 1.0 / b), 0.0, 0.1) for _ in range(h)]
    sizes = [10.0 for _ in range(b * h)]
    draw.draw_points(point_list, colors, sizes)


def main():
    usd_help = UsdHelper()
    n_envs = 4


    plan_env_ids = [1, 2]
    n_plan_envs = len(plan_env_ids)


    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage

    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")

    stage = my_world.stage
    my_world.scene.add_default_ground_plane()


    usd_help.load_stage(stage)


    target_list = []
    offset_y = 2.5
    pose = Pose.from_list([0, 0, 0, 1, 0, 0, 0])
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), args.robot))["robot_cfg"]

    robot_list = []
    robot_prim_path_list = []

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
        r, robot_path = add_robot_to_scene(
            robot_cfg,
            my_world,
            "/World/world_" + str(i) + "/",
            robot_name="robot_" + str(i),
            position=pose.position[0].cpu().numpy(),
            initialize_world=False,
        )
        robot_list.append(r)
        robot_prim_path_list.append(robot_path)

    setup_curobo_logger("warn")
    my_world.initialize_physics()

    past_pose = None
    n_obstacle_cuboids = 30
    n_obstacle_mesh = 10

    tensor_args = TensorDeviceType()

    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]
    robot_cfg["kinematics"]["collision_sphere_buffer"] += 0.02

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

    init_curobo = False

    mpc_config = MpcSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg_list,
        use_cuda_graph=True,
        use_cuda_graph_metrics=True,
        use_cuda_graph_full_step=False,
        self_collision_check=True,
        collision_checker_type=CollisionCheckerType.MESH,
        collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
        use_mppi=True,
        use_lbfgs=False,
        use_es=False,
        store_rollouts=True,
        step_dt=0.02,
    )

    mpc = MpcSolver(mpc_config)

    retract_cfg = mpc.rollout_fn.dynamics_model.retract_config.clone().unsqueeze(0)
    joint_names = mpc.rollout_fn.joint_names


    retract_cfg_all = retract_cfg.repeat(n_envs, 1)


    retract_cfg_plan = retract_cfg.repeat(n_plan_envs, 1)

    state = mpc.rollout_fn.compute_kinematics(
        JointState.from_position(retract_cfg_plan, joint_names=joint_names)
    )
    current_state_plan = JointState.from_position(retract_cfg_plan, joint_names=joint_names)
    retract_pose_plan = Pose(state.ee_pos_seq, quaternion=state.ee_quat_seq)


    plan_env_query_idx = torch.tensor(plan_env_ids, device=tensor_args.device, dtype=torch.int32)
    goal = Goal(
        current_state=current_state_plan,
        goal_state=JointState.from_position(retract_cfg_plan, joint_names=joint_names),
        goal_pose=retract_pose_plan,
    )


    goal_buffer = mpc.setup_solve_batch_env(goal, 1)

    goal_buffer = goal_buffer.create_index_buffers(
        batch_size=n_plan_envs,
        batch_env=True,
        batch_retract=False,
        num_seeds=1,
        tensor_args=tensor_args,
        env_query_idx=plan_env_query_idx,
    )
    mpc.update_goal(goal_buffer)
    mpc_result = mpc.step(current_state_plan, max_attempts=2)

    init_world = False
    cmd_state_full = None
    step = 0
    add_extensions(simulation_app, args.headless_mode)
    art_controllers = [r.get_articulation_controller() for r in robot_list]
    past_pose_list = [None] * n_envs


    cmd_state_all = [None] * n_envs








    while simulation_app.is_running():
        if not init_world:
            for _ in range(10):
                my_world.step(render=True)
            init_world = True
        draw_points(mpc.get_visual_rollouts())

        my_world.step(render=True)
        if not my_world.is_playing():
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

        if not init_curobo:
            init_curobo = True
        step += 1
        step_index = step
        if step_index % 1000 == 0:
            print("Updating world")
            world_update_start_time = time.time()
            torch.cuda.synchronize()
            for i, robot in enumerate(robot_list):
                robot_prim_path = robot_prim_path_list[i]
                obstacles = usd_help.get_obstacles_from_stage(
                    only_paths=["/World/world_" + str(i)],
                    ignore_substring=[
                        robot_prim_path,
                        "/World/world_" + str(i) + "/target",
                        "/World/defaultGroundPlane",
                        "/curobo",
                    ],
                    reference_prim_path=robot_prim_path,
                )

                if len(world_cfg_list[i].objects) > 0:
                    obstacles.add_obstacle(world_cfg_list[i].objects[0])
                mpc.world_coll_checker.load_collision_model(obstacles, env_idx=i)
            torch.cuda.synchronize()
            world_update_time = time.time() - world_update_start_time
            print(f"[Timing] World collision update time: {world_update_time*1000:.2f} ms")


        sp_buffer = []
        sq_buffer = []
        for k in target_list:
            cube_position, cube_orientation = k.get_local_pose()
            sp_buffer.append(cube_position)
            sq_buffer.append(cube_orientation)

        ik_goal_all = Pose(
            position=tensor_args.to_device(sp_buffer),
            quaternion=tensor_args.to_device(sq_buffer),
        )


        target_moved = False
        for i in range(n_envs):
            if past_pose_list[i] is None:
                past_pose_list[i] = sp_buffer[i] + 1.0
            if np.linalg.norm(sp_buffer[i] - past_pose_list[i]) > 1e-3:
                target_moved = True
                past_pose_list[i] = sp_buffer[i]




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


        if step_index % 2000 == 0:
            plan_env_ids = [0, 1]
        elif step_index % 2000 == 1000:
            plan_env_ids = [2, 3]


        current_plan_env_ids = plan_env_ids
        current_n_plan_envs = len(current_plan_env_ids)

        plan_start_state = JointState(
            position=full_js_all.position[current_plan_env_ids],
            velocity=full_js_all.velocity[current_plan_env_ids],
            acceleration=full_js_all.acceleration[current_plan_env_ids],
            jerk=full_js_all.jerk[current_plan_env_ids],
            joint_names=full_js_all.joint_names,
        )

        plan_start_state = plan_start_state.get_ordered_joint_state(mpc.rollout_fn.joint_names)


        ik_goal_plan = Pose(
            position=ik_goal_all.position[current_plan_env_ids],
            quaternion=ik_goal_all.quaternion[current_plan_env_ids],
        )


        if target_moved:

            goal_buffer.goal_pose.copy_(ik_goal_plan)
            mpc.update_goal(goal_buffer)



        plan_env_query_idx = torch.tensor(current_plan_env_ids, device=tensor_args.device, dtype=torch.int32)




        if goal_buffer.batch_world_idx is None or goal_buffer.batch_world_idx.shape[0] != current_n_plan_envs:

            goal_buffer = goal_buffer.create_index_buffers(
                batch_size=current_n_plan_envs,
                batch_env=True,
                batch_retract=False,
                num_seeds=1,
                tensor_args=tensor_args,
                env_query_idx=plan_env_query_idx,
            )

            mpc.update_goal(goal_buffer)
        else:


            goal_buffer.batch_world_idx = plan_env_query_idx.unsqueeze(-1)


        goal_buffer.current_state.copy_(plan_start_state)
        goal_buffer.goal_pose.copy_(ik_goal_plan)






        mpc.update_goal(goal_buffer)


        torch.cuda.synchronize()
        mpc_step_start_time = time.time()
        mpc_result = mpc.step(plan_start_state, max_attempts=2)
        torch.cuda.synchronize()
        mpc_step_time = time.time() - mpc_step_start_time


        succ = True
        cmd_state_full = mpc_result.js_action


        for batch_idx, env_id in enumerate(current_plan_env_ids):
            if cmd_state_full is not None:
                cmd_state_all[env_id] = cmd_state_full[batch_idx]


        for env_id in range(n_envs):
            if cmd_state_all[env_id] is not None:
                sim_js_names = robot_list[env_id].dof_names
                common_js_names = []
                idx_list = []
                for x in sim_js_names:
                    if x in cmd_state_all[env_id].joint_names:
                        idx_list.append(robot_list[env_id].get_dof_index(x))
                        common_js_names.append(x)


                cmd_state = cmd_state_all[env_id].get_ordered_joint_state(common_js_names)

                art_action = ArticulationAction(
                    cmd_state.position.view(-1).cpu().numpy(),

                    joint_indices=idx_list,
                )

                if step_index % 1000 == 0 and env_id == 0:
                    if env_id in current_plan_env_ids:
                        batch_idx = current_plan_env_ids.index(env_id)
                        print(
                            f"[Metrics] Env {env_id} - Feasible: {mpc_result.metrics.feasible[batch_idx].item()}, "
                            f"Pose error: {mpc_result.metrics.pose_error[batch_idx].item():.4f}"
                        )

                if succ:

                    for _ in range(1):
                        art_controllers[env_id].apply_action(art_action)

                else:
                    carb.log_warn(f"No action is being taken for env {env_id}.")


        if step_index % 100 == 0:
            print(f"[Timing] MPC step time: {mpc_step_time*1000:.2f} ms (planning for {current_n_plan_envs} envs: {current_plan_env_ids})")




if __name__ == "__main__":
    main()
    simulation_app.close()

