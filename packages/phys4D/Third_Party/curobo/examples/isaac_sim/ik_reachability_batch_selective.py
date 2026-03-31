











try:

    import isaacsim
except ImportError:
    pass


import torch

a = torch.zeros(4, device="cuda:0")


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

simulation_app = SimulationApp(
    {
        "headless": args.headless_mode is not None,
        "width": "1920",
        "height": "1080",
    }
)

from typing import Dict


import carb
import numpy as np
from helper import add_extensions, add_robot_to_scene
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid, sphere


from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel


from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.types.state import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import (
    get_assets_path,
    get_filename,
    get_path_of_dir,
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig










def get_pose_grid(n_x, n_y, n_z, max_x, max_y, max_z):
    x = np.linspace(-max_x, max_x, n_x)
    y = np.linspace(-max_y, max_y, n_y)
    z = np.linspace(0, max_z, n_z)
    x, y, z = np.meshgrid(x, y, z, indexing="ij")

    position_arr = np.zeros((n_x * n_y * n_z, 3))
    position_arr[:, 0] = x.flatten()
    position_arr[:, 1] = y.flatten()
    position_arr[:, 2] = z.flatten()
    return position_arr



_accumulated_points = []
_accumulated_colors = []
_accumulated_sizes = []

def draw_points(pose, success, env_offset=None, clear_first=False):

    try:
        from omni.isaac.debug_draw import _debug_draw
    except ImportError:
        from isaacsim.util.debug_draw import _debug_draw

    global _accumulated_points, _accumulated_colors, _accumulated_sizes

    draw = _debug_draw.acquire_debug_draw_interface()

    if clear_first:
        _accumulated_points = []
        _accumulated_colors = []
        _accumulated_sizes = []
        draw.clear_points()

    cpu_pos = pose.position.cpu().numpy()
    b, _ = cpu_pos.shape
    point_list = []
    colors = []
    for i in range(b):

        pos = cpu_pos[i]
        if env_offset is not None:

            pos = pos + env_offset
        point_list += [(pos[0], pos[1], pos[2])]
        if success[i].item():
            colors += [(0, 1, 0, 0.8)]
        else:
            colors += [(1, 0, 0, 0.2)]
    sizes = [40.0 for _ in range(b)]


    _accumulated_points.extend(point_list)
    _accumulated_colors.extend(colors)
    _accumulated_sizes.extend(sizes)


    if len(_accumulated_points) > 0:
        draw.draw_points(_accumulated_points, _accumulated_colors, _accumulated_sizes)


def main():
    usd_help = UsdHelper()

    n_envs = 4

    ik_env_ids = [0, 2]
    n_ik_envs = len(ik_env_ids)


    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage

    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")

    setup_curobo_logger("warn")
    past_pose = None
    n_obstacle_cuboids = 30
    n_obstacle_mesh = 10


    target_pose = None

    tensor_args = TensorDeviceType()

    robot_cfg = load_yaml(join_path(get_robot_configs_path(), args.robot))["robot_cfg"]

    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

    robot_list = []
    robot_prim_path_list = []
    target_list = []
    offset_y = 2.5
    pose = Pose.from_list([0, 0, 0, 1, 0, 0, 0])


    stage = my_world.stage
    usd_help.load_stage(stage)
    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")


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

    my_world.scene.add_default_ground_plane()
    my_world.initialize_physics()


    world_file = ["collision_test.yml", "collision_thin_walls.yml", "collision_test.yml", "collision_thin_walls.yml"]
    world_cfg_list = []
    for i in range(n_envs):
        world_cfg_table = WorldConfig.from_dict(
            load_yaml(join_path(get_world_configs_path(), world_file[i]))
        )
        world_cfg_table.objects[0].pose[2] -= 0.02
        world_cfg_table.randomize_color(r=[0.2, 0.3], b=[0.0, 0.05], g=[0.2, 0.3])
        usd_help.add_world_to_stage(world_cfg_table, base_frame="/World/world_" + str(i))
        world_cfg_list.append(world_cfg_table)


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
        collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
    )
    ik_solver = IKSolver(ik_config)


    position_grid_offset = tensor_args.to_device(get_pose_grid(10, 10, 5, 0.5, 0.5, 0.5))
    n_grid_points = position_grid_offset.shape[0]


    fk_state = ik_solver.fk(ik_solver.get_retract_config().view(1, -1))
    base_goal_pose = fk_state.ee_pose



    base_pos = base_goal_pose.position[0]
    base_quat = base_goal_pose.quaternion[0]



    goal_pos_all = base_pos.unsqueeze(0).unsqueeze(0).repeat(n_envs, n_grid_points, 1)
    goal_quat_all = base_quat.unsqueeze(0).unsqueeze(0).repeat(n_envs, n_grid_points, 1)


    for env_id in range(n_envs):
        goal_pos_all[env_id] += position_grid_offset

    goal_pose_all = Pose(position=goal_pos_all, quaternion=goal_quat_all)



    env_query_idx = torch.tensor(ik_env_ids, device=tensor_args.device, dtype=torch.int32)



    plan_goal_pose = Pose(
        position=goal_pose_all.position[ik_env_ids],
        quaternion=goal_pose_all.quaternion[ik_env_ids],
    )

    result = ik_solver.solve_batch_env_goalset(plan_goal_pose, env_query_idx=env_query_idx)

    print("Curobo is Ready")
    print(f"Initial IK reachability: {torch.count_nonzero(result.success).item()}/{len(result.success)} poses reachable")
    add_extensions(simulation_app, args.headless_mode)

    cmd_plan = [None] * n_envs
    cmd_idx = 0
    i = 0
    spheres = [None] * n_envs
    past_goal_poses = [None] * n_envs


    env_base_positions = []
    for env_id in range(n_envs):
        base_pose = Pose.from_list([0, 0, 0, 1, 0, 0, 0])
        if env_id > 0:
            for j in range(env_id):
                base_pose.position[0, 1] += offset_y
        env_base_positions.append(base_pose.position[0].cpu().numpy())

    while simulation_app.is_running():
        my_world.step(render=True)
        if not my_world.is_playing():
            if i % 100 == 0:
                print("**** Click Play to start simulation *****")
            i += 1
            continue

        step_index = my_world.current_time_step_index
        if step_index <= 10:

            for env_id, robot in enumerate(robot_list):
                robot._articulation_view.initialize()
                idx_list = [robot.get_dof_index(x) for x in j_names]
                robot.set_joint_positions(default_config, idx_list)

                robot._articulation_view.set_max_efforts(
                    values=np.array([5000 for _ in range(len(idx_list))]), joint_indices=idx_list
                )
        if step_index < 20:
            continue

        if step_index == 50 or step_index % 500 == 0.0:

            for env_id, robot_path in enumerate(robot_prim_path_list):
                print(f"Updating world {env_id}, reading w.r.t.", robot_path)
                obstacles = usd_help.get_obstacles_from_stage(
                    only_paths=[f"/World/world_{env_id}"],
                    reference_prim_path=robot_path,
                    ignore_substring=[
                        robot_path,
                        f"/World/world_{env_id}/target",
                        "/World/defaultGroundPlane",
                        "/curobo",
                    ],
                ).get_collision_check_world()
                print(f"Env {env_id} obstacles: {[x.name for x in obstacles.objects]}")


                if env_id == n_envs - 1:
                    ik_solver.update_world(obstacles)
                    print("Updated World")
                    carb.log_info("Synced CuRobo world from stage.")



        target_positions = []
        target_orientations = []

        for env_id in range(n_envs):

            cube_position_world, cube_orientation = target_list[env_id].get_world_pose()


            cube_position_local = cube_position_world - env_base_positions[env_id]
            target_positions.append(cube_position_local)
            target_orientations.append(cube_orientation)


        need_recompute = False
        for idx, env_id in enumerate(ik_env_ids):
            if past_goal_poses[env_id] is None:
                past_goal_poses[env_id] = target_positions[env_id]
                need_recompute = True
            elif np.linalg.norm(target_positions[env_id] - past_goal_poses[env_id]) > 1e-3:
                need_recompute = True
                break

        if need_recompute:

            for env_id in range(n_envs):
                ik_goal = Pose(
                    position=tensor_args.to_device(target_positions[env_id]),
                    quaternion=tensor_args.to_device(target_orientations[env_id]),
                )


                goal_pose_all.position[env_id] = ik_goal.position[:] + position_grid_offset
                goal_pose_all.quaternion[env_id] = ik_goal.quaternion[:]



            plan_goal_pose = Pose(
                position=goal_pose_all.position[ik_env_ids],
                quaternion=goal_pose_all.quaternion[ik_env_ids],
            )


            result = ik_solver.solve_batch_env_goalset(plan_goal_pose, env_query_idx=env_query_idx)

            succ = torch.any(result.success)
            print(
                f"[Selective IK Reachability] Envs {ik_env_ids}: "
                f"Poses: {plan_goal_pose.batch}, "
                f"Reachable: {torch.count_nonzero(result.success).item()}, "
                f"Time(s): {result.solve_time:.4f}"
            )







            for idx, env_id in enumerate(ik_env_ids):
                env_goal_pose = plan_goal_pose[idx]



                env_success = torch.zeros(n_grid_points, dtype=torch.bool, device=result.success.device)


                if result.success[idx]:
                    if result.goalset_index is not None and idx < len(result.goalset_index):
                        selected_goal_idx = result.goalset_index[idx].item()
                        if selected_goal_idx < n_grid_points:
                            env_success[selected_goal_idx] = True
                            print(f"[IK Reachability] Env {env_id}: Selected goal {selected_goal_idx} from {n_grid_points} goals")


                env_offset = np.array([0.0, 0.0, 0.0])
                if env_id > 0:
                    base_pose = Pose.from_list([0, 0, 0, 1, 0, 0, 0])
                    for j in range(env_id):
                        base_pose.position[0, 1] += offset_y
                    env_offset = base_pose.position[0].cpu().numpy()


                draw_points(env_goal_pose, env_success, env_offset=env_offset, clear_first=(idx == 0))


            for idx, env_id in enumerate(ik_env_ids):
                past_goal_poses[env_id] = target_positions[env_id]




            if succ:

                for idx, env_id in enumerate(ik_env_ids):
                    if result.success[idx]:


                        try:
                            cmd_plan[env_id] = result.js_solution[idx]
                            print(f"[IK Reachability] Env {env_id} (idx {idx}): Success, got solution")
                        except (IndexError, AttributeError) as e:
                            print(f"[IK Reachability] Env {env_id} (idx {idx}): Error accessing solution: {e}")
                            print(f"  result.js_solution type: {type(result.js_solution)}")
                            if hasattr(result.js_solution, 'position'):
                                print(f"  result.js_solution.position shape: {result.js_solution.position.shape}")
                            continue


                        sim_js_names = robot_list[env_id].dof_names
                        idx_list = []
                        common_js_names = []
                        for x in sim_js_names:
                            if x in cmd_plan[env_id].joint_names:
                                idx_list.append(robot_list[env_id].get_dof_index(x))
                                common_js_names.append(x)

                        cmd_plan[env_id] = cmd_plan[env_id].get_ordered_joint_state(common_js_names)
                    else:
                        print(f"[IK Reachability] Env {env_id} (idx {idx}): IK failed")


        for env_id in range(n_envs):
            if cmd_plan[env_id] is not None:

                cmd_state = cmd_plan[env_id]
                if len(cmd_state.position.shape) == 1:
                    position = cmd_state.position
                else:
                    position = cmd_state.position[0]

                sim_js_names = robot_list[env_id].dof_names
                idx_list = []
                for x in sim_js_names:
                    if x in cmd_state.joint_names:
                        idx_list.append(robot_list[env_id].get_dof_index(x))

                if len(idx_list) > 0:
                    robot_list[env_id].set_joint_positions(position.cpu().numpy(), idx_list)
                    if step_index % 100 == 0:
                        print(f"[IK Reachability] Applied solution to env {env_id}, {len(idx_list)} joints")
                else:
                    print(f"[IK Reachability] Warning: No matching joints for env {env_id}")

                cmd_plan[env_id] = None


        if args.visualize_spheres and step_index % 2 == 0:
            for env_id, robot in enumerate(robot_list):
                sim_js = robot.get_joints_state()
                if sim_js is None:
                    continue
                sim_js_names = robot.dof_names
                cu_js = JointState(
                    position=tensor_args.to_device(sim_js.positions),
                    velocity=tensor_args.to_device(sim_js.velocities) * 0.0,
                    acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
                    jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
                    joint_names=sim_js_names,
                )
                cu_js = cu_js.get_ordered_joint_state(ik_solver.kinematics.joint_names)

                sph_list = ik_solver.kinematics.get_robot_as_spheres(cu_js.position)


                env_offset = np.array([0.0, 0.0, 0.0])
                if env_id > 0:
                    base_pose = Pose.from_list([0, 0, 0, 1, 0, 0, 0])
                    for j in range(env_id):
                        base_pose.position[0, 1] += offset_y
                    env_offset = base_pose.position[0].cpu().numpy()

                if spheres[env_id] is None:
                    spheres[env_id] = []

                    for si, s in enumerate(sph_list[0]):
                        sphere_pos = np.ravel(s.position) + env_offset
                        sp = sphere.VisualSphere(
                            prim_path=f"/curobo/robot_sphere_{env_id}_{si}",
                            position=sphere_pos,
                            radius=float(s.radius),
                            color=np.array([0, 0.8, 0.2]),
                        )
                        spheres[env_id].append(sp)
                else:
                    for si, s in enumerate(sph_list[0]):
                        sphere_pos = np.ravel(s.position) + env_offset
                        spheres[env_id][si].set_world_pose(position=sphere_pos)
                        spheres[env_id][si].set_radius(float(s.radius))

    simulation_app.close()


if __name__ == "__main__":
    main()

