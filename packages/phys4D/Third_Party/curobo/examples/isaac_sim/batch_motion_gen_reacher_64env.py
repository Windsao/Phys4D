











try:

    import isaacsim
except ImportError:
    pass


import torch

a = torch.zeros(4, device="cuda:0")


import os
import time


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
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig


def main():
    usd_help = UsdHelper()
    act_distance = 0.2

    n_envs = 4
    grid_size = 2
    batch_size = 2



    env_batches = []
    for i in range(0, n_envs, batch_size):
        batch = list(range(i, min(i + batch_size, n_envs)))
        env_batches.append(batch)

    my_world = World(stage_units_in_meters=1.0)
    my_world.scene.add_default_ground_plane()

    stage = my_world.stage
    usd_help.load_stage(stage)
    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")


    stage = my_world.stage


    target_list = []
    target_material_list = []
    offset_x = 2.5
    offset_y = 2.5
    radius = 0.1
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), args.robot))["robot_cfg"]

    robot_list = []

    for i in range(n_envs):

        row = i // grid_size
        col = i % grid_size


        pos_x = col * offset_x
        pos_y = row * offset_y

        pose = Pose.from_list([pos_x, pos_y, 0, 1, 0, 0, 0])
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

    world_file = ["collision_thin_walls.yml", "collision_test.yml"]
    world_cfg_list = []
    for i in range(n_envs):

        world_file_idx = i % len(world_file)
        world_cfg = WorldConfig.from_dict(
            load_yaml(join_path(get_world_configs_path(), world_file[world_file_idx]))
        )
        world_cfg.objects[0].pose[2] -= 0.02
        world_cfg.randomize_color(r=[0.2, 0.3], b=[0.0, 0.05], g=[0.2, 0.3])
        usd_help.add_world_to_stage(world_cfg, base_frame="/World/world_" + str(i))
        world_cfg_list.append(world_cfg)



    empty_world_cfg_list = [WorldConfig() for _ in range(batch_size)]

    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        empty_world_cfg_list,
        tensor_args,
        collision_checker_type=CollisionCheckerType.MESH,
        use_cuda_graph=True,
        interpolation_dt=0.03,
        collision_cache={"obb": 10, "mesh": 10},
        collision_activation_distance=0.025,
        maximum_trajectory_dt=0.25,
        n_collision_envs=batch_size,
    )
    motion_gen = MotionGen(motion_gen_config)



    world_cfg_loaded = [None] * n_envs

    batch_env_idx_to_env_id = [None] * batch_size
    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

    print("warming up...")






    add_extensions(simulation_app, args.headless_mode)
    config = RobotWorldConfig.load_from_config(
        robot_file, world_cfg_list, collision_activation_distance=act_distance
    )
    model = RobotWorld(config)
    i = 0
    max_distance = 0.5
    x_sph = torch.zeros((n_envs, 1, 1, 4), device=tensor_args.device, dtype=tensor_args.dtype)
    x_sph[..., 3] = radius
    env_query_idx = torch.arange(n_envs, device=tensor_args.device, dtype=torch.int32)
    plan_config = MotionGenPlanConfig(
        enable_graph=False, max_attempts=2, enable_finetune_trajopt=True
    )
    prev_goal = None
    cmd_plan = [None] * n_envs
    art_controllers = [r.get_articulation_controller() for r in robot_list]
    cmd_idx = 0
    past_goal = None
    need_plan_envs = list(range(n_envs))
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


                cu_js = JointState(
                    position=tensor_args.to_device(np.zeros((1, len(sim_js_names)), dtype=np.float32)),
                    velocity=tensor_args.to_device(np.zeros((1, len(sim_js_names)), dtype=np.float32)),
                    acceleration=tensor_args.to_device(np.zeros((1, len(sim_js_names)), dtype=np.float32)),
                    jerk=tensor_args.to_device(np.zeros((1, len(sim_js_names)), dtype=np.float32)),
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

            if full_js_all is None:
                full_js_all = cu_js
            else:
                full_js_all = full_js_all.stack(cu_js)

        if full_js_all is None:
            continue


        if full_js_all.position.shape[0] != n_envs:
            print(f"[Planning Status] WARNING: full_js_all has {full_js_all.position.shape[0]} envs, expected {n_envs}")
            continue


        prev_distance_all = ik_goal.distance(prev_goal)
        past_distance_all = ik_goal.distance(past_goal)


        max_pos_dist = torch.max(prev_distance_all[0]).item()
        max_quat_dist = torch.max(prev_distance_all[1]).item()
        if max_pos_dist > 1e-4 or max_quat_dist > 1e-4:
            print(f"[Planning Status] Max distances - pos: {max_pos_dist:.4f}, quat: {max_quat_dist:.4f}")

            significant_changes = []
            for env_id in range(n_envs):
                pos_dist = prev_distance_all[0][env_id].item()
                quat_dist = prev_distance_all[1][env_id].item()
                if pos_dist > 1e-3 or quat_dist > 1e-3:
                    significant_changes.append((env_id, pos_dist, quat_dist))
            if len(significant_changes) > 0:
                print(f"[Planning Status] Envs with distance > 1e-3: {significant_changes[:10]}")


        is_first_planning = all(cmd_plan[env_id] is None for env_id in range(n_envs))


        if is_first_planning:
            print(f"[Planning Status] is_first_planning=True (all plans are None)")
        else:
            plans_count = sum(1 for env_id in range(n_envs) if cmd_plan[env_id] is not None)




        newly_added = []
        goal_changed_envs = []
        for env_id in range(n_envs):

            pos_dist = prev_distance_all[0][env_id].item()
            quat_dist = prev_distance_all[1][env_id].item()
            goal_changed = (pos_dist > 1e-2 or quat_dist > 1e-2)


            if pos_dist > 1e-4 or quat_dist > 1e-4:
                print(f"[Planning Status] Env {env_id}: pos_dist={pos_dist:.4f}, quat_dist={quat_dist:.4f}, goal_changed={goal_changed}, in_need_plan={env_id in need_plan_envs}")

            if goal_changed:
                goal_changed_envs.append(env_id)

                if env_id not in need_plan_envs:
                    need_plan_envs.append(env_id)
                    newly_added.append(env_id)

                cmd_plan[env_id] = None


        if len(goal_changed_envs) > 0:
            print(f"[Planning Status] Goal changed envs: {goal_changed_envs}")
            for env_id in goal_changed_envs:
                pos_dist = prev_distance_all[0][env_id].item()
                quat_dist = prev_distance_all[1][env_id].item()
                print(f"  Env {env_id}: pos_dist={pos_dist:.4f}, quat_dist={quat_dist:.4f}")


        if len(need_plan_envs) > 0:
            print(f"[Planning Status] need_plan_envs: {need_plan_envs} (total: {len(need_plan_envs)})")
            if len(newly_added) > 0:
                print(f"[Planning Status] Newly added envs: {newly_added}")
        elif len(goal_changed_envs) > 0:
            print(f"[Planning Status] WARNING: goal_changed_envs={goal_changed_envs} but need_plan_envs is empty!")


        if 1 not in need_plan_envs and prev_distance_all[0][1].item() > 1e-2:
            print(f"[Planning Status] WARNING: Env 1 has distance {prev_distance_all[0][1].item():.4f} but not in need_plan_envs!")
            print(f"[Planning Status]   is_first_planning={is_first_planning}, cmd_plan[1]={cmd_plan[1] is not None}")


        if len(need_plan_envs) > 0:


            ready_to_plan = []
            not_ready_reasons = {}
            for env_id in need_plan_envs:

                past_goal_stable = (
                    past_distance_all[0][env_id] < 1e-3 and
                    past_distance_all[1][env_id] < 1e-3
                )


                if not past_goal_stable:
                    pos_dist = past_distance_all[0][env_id].item()
                    quat_dist = past_distance_all[1][env_id].item()
                    not_ready_reasons[env_id] = f"past_goal_not_stable (dist: {pos_dist:.4f}, {quat_dist:.4f})"


                if past_goal_stable:
                    ready_to_plan.append(env_id)


            if len(ready_to_plan) > 0:
                print(f"[Planning Status] Ready to plan: {ready_to_plan} (total: {len(ready_to_plan)})")
            if len(not_ready_reasons) > 0:
                print(f"[Planning Status] Not ready: {not_ready_reasons}")

            if len(ready_to_plan) == 0:

                pass
            else:

                if len(ready_to_plan) > batch_size:
                    envs_to_plan = ready_to_plan[:batch_size]
                else:
                    envs_to_plan = ready_to_plan.copy()



                plan_start_state = JointState(
                    position=full_js_all.position[envs_to_plan],
                    velocity=full_js_all.velocity[envs_to_plan],
                    acceleration=full_js_all.acceleration[envs_to_plan],
                    jerk=full_js_all.jerk[envs_to_plan],
                    joint_names=full_js_all.joint_names,
                )
                plan_goal_pose = Pose(
                    position=ik_goal.position[envs_to_plan],
                    quaternion=ik_goal.quaternion[envs_to_plan],
                )

                plan_start_state = plan_start_state.get_ordered_joint_state(motion_gen.kinematics.joint_names)


                plan_start_time = time.time()




                batch_env_idx_mapping = {}
                next_batch_idx = 0

                for env_id in envs_to_plan:

                    if world_cfg_loaded[env_id] is not None:

                        batch_env_idx = world_cfg_loaded[env_id]
                        batch_env_idx_mapping[env_id] = batch_env_idx

                        batch_env_idx_to_env_id[batch_env_idx] = env_id
                    else:


                        if next_batch_idx < batch_size:
                            batch_env_idx = next_batch_idx
                            next_batch_idx += 1
                        else:


                            batch_env_idx = next_batch_idx % batch_size
                            next_batch_idx += 1

                            old_env_id = batch_env_idx_to_env_id[batch_env_idx]
                            if old_env_id is not None:
                                world_cfg_loaded[old_env_id] = None


                        motion_gen.world_coll_checker.load_collision_model(
                            world_cfg_list[env_id],
                            env_idx=batch_env_idx,
                            fix_cache_reference=motion_gen.use_cuda_graph,
                        )
                        world_cfg_loaded[env_id] = batch_env_idx
                        batch_env_idx_to_env_id[batch_env_idx] = env_id
                        batch_env_idx_mapping[env_id] = batch_env_idx
                        print(f"[Lazy Loading] Loaded world collision for env {env_id} into batch slot {batch_env_idx}")




                plan_env_query_idx_list = [batch_env_idx_mapping[env_id] for env_id in envs_to_plan]
                plan_env_query_idx = torch.tensor(plan_env_query_idx_list, device=tensor_args.device, dtype=torch.int32)
                if plan_env_query_idx.dim() == 0:

                    plan_env_query_idx = plan_env_query_idx.unsqueeze(0)
                elif plan_env_query_idx.dim() > 1:
                    plan_env_query_idx = plan_env_query_idx.squeeze()


                print(f"[Planning Debug] envs_to_plan: {envs_to_plan}")
                print(f"[Planning Debug] plan_start_state.position.shape: {plan_start_state.position.shape}")
                print(f"[Planning Debug] plan_goal_pose.position.shape: {plan_goal_pose.position.shape}")
                print(f"[Planning Debug] plan_env_query_idx: {plan_env_query_idx}, shape: {plan_env_query_idx.shape}")
                print(f"[Planning Debug] full_js_all.position.shape: {full_js_all.position.shape}")
                for batch_idx, env_id in enumerate(envs_to_plan):
                    world_file_idx = env_id % len(world_file)
                    world_name = world_file[world_file_idx]
                    start_pos = plan_start_state.position[batch_idx].cpu().numpy()
                    goal_pos = plan_goal_pose.position[batch_idx].cpu().numpy()



                    full_js_env = JointState(
                        position=full_js_all.position[env_id:env_id+1],
                        velocity=full_js_all.velocity[env_id:env_id+1],
                        acceleration=full_js_all.acceleration[env_id:env_id+1],
                        jerk=full_js_all.jerk[env_id:env_id+1],
                        joint_names=full_js_all.joint_names,
                    )
                    full_js_env_ordered = full_js_env.get_ordered_joint_state(motion_gen.kinematics.joint_names)
                    full_js_pos = full_js_env_ordered.position[0].cpu().numpy()
                    print(f"[Planning Debug] Batch idx {batch_idx} -> Env {env_id} (world: {world_name})")
                    print(f"[Planning Debug]   Start joint pos (from plan_start_state): {start_pos[:3]}... (first 3 joints, shape: {start_pos.shape})")
                    print(f"[Planning Debug]   Start joint pos (from full_js_all[{env_id}]): {full_js_pos[:3]}... (first 3 joints, shape: {full_js_pos.shape})")
                    if start_pos.shape == full_js_pos.shape:
                        print(f"[Planning Debug]   Match: {np.allclose(start_pos, full_js_pos, atol=1e-5)}")
                    else:
                        print(f"[Planning Debug]   Shape mismatch: plan_start_state has {start_pos.shape[0]} joints, full_js_all has {full_js_pos.shape[0]} joints")
                    print(f"[Planning Debug]   Goal position: {goal_pos}")
                print(f"[Planning Debug] plan_start_state.joint_names: {plan_start_state.joint_names[:5]}...")


                result = motion_gen.plan_batch_env(
                    plan_start_state,
                    plan_goal_pose,
                    plan_config.clone(),
                    env_query_idx=plan_env_query_idx
                )
                plan_time = time.time() - plan_start_time
                print(f"[Planning Status] Planned envs: {envs_to_plan}, time: {plan_time:.4f}s, success: {torch.sum(result.success).item()}/{len(envs_to_plan)}")

                if torch.count_nonzero(result.success) > 0:


                    trajs = None


                    if result.interpolated_plan is not None:
                        try:

                            if isinstance(result.interpolated_plan, list):
                                if result.path_buffer_last_tstep is not None:
                                    trajs = result.get_paths()

                            elif hasattr(result.interpolated_plan, 'position'):

                                pos_shape = result.interpolated_plan.position.shape
                                print(f"[Planning Status] Extracting from batch JointState (shape: {pos_shape})")




                                if len(pos_shape) == 2:


                                    trajs = []
                                    if result.success[0] if len(result.success) > 0 else False:
                                        single_js = result.interpolated_plan

                                        if result.path_buffer_last_tstep is not None and len(result.path_buffer_last_tstep) > 0:
                                            last_tstep = result.path_buffer_last_tstep[0].item()
                                            traj = single_js.trim_trajectory(0, last_tstep)
                                        else:
                                            traj = single_js
                                        trajs.append(traj)
                                    else:
                                        trajs.append(None)
                                elif len(pos_shape) == 3:

                                    trajs = []
                                    batch_size = pos_shape[0]
                                    for s in range(batch_size):
                                        if s < len(result.success) and result.success[s]:


                                            single_batch_js = JointState(
                                                position=result.interpolated_plan.position[s],
                                                velocity=result.interpolated_plan.velocity[s] if result.interpolated_plan.velocity is not None else None,
                                                acceleration=result.interpolated_plan.acceleration[s] if result.interpolated_plan.acceleration is not None else None,
                                                jerk=result.interpolated_plan.jerk[s] if result.interpolated_plan.jerk is not None else None,
                                                joint_names=result.interpolated_plan.joint_names,
                                            )

                                            if result.path_buffer_last_tstep is not None and s < len(result.path_buffer_last_tstep):
                                                last_tstep = result.path_buffer_last_tstep[s].item()
                                                traj = single_batch_js.trim_trajectory(0, last_tstep)
                                            else:

                                                traj = single_batch_js
                                            trajs.append(traj)
                                        else:
                                            trajs.append(None)
                                else:
                                    print(f"[Planning Status] Warning: Unexpected interpolated_plan shape: {pos_shape}")
                                    trajs = None
                        except (ValueError, AttributeError, TypeError) as e:
                            print(f"[Planning Status] Warning: get_paths() failed: {e}")
                            trajs = None


                    if trajs is None:
                        if result.optimized_plan is not None:
                            print(f"[Planning Status] Using optimized_plan as fallback")

                            trajs = []
                            for s in range(len(result.success)):
                                if result.success[s]:


                                    traj = JointState(
                                        position=result.optimized_plan.position[s:s+1],
                                        velocity=result.optimized_plan.velocity[s:s+1] if result.optimized_plan.velocity is not None else None,
                                        acceleration=result.optimized_plan.acceleration[s:s+1] if result.optimized_plan.acceleration is not None else None,
                                        jerk=result.optimized_plan.jerk[s:s+1] if result.optimized_plan.jerk is not None else None,
                                        joint_names=result.optimized_plan.joint_names,
                                    )
                                    trajs.append(traj)
                                else:
                                    trajs.append(None)
                        else:
                            print(f"[Planning Status] Error: No valid trajectory found in result (interpolated_plan={result.interpolated_plan is not None}, optimized_plan={result.optimized_plan is not None})")
                            trajs = [None] * len(result.success)

                    for s in range(len(result.success)):
                        if result.success[s] and trajs[s] is not None:
                            actual_env_id = envs_to_plan[s]

                            prev_goal.position[actual_env_id].copy_(plan_goal_pose.position[s])
                            prev_goal.quaternion[actual_env_id].copy_(plan_goal_pose.quaternion[s])


                            cmd_plan[actual_env_id] = motion_gen.get_full_js(trajs[s])

                            idx_list = []
                            common_js_names = []
                            for x in sim_js_names:
                                if x in cmd_plan[actual_env_id].joint_names:
                                    idx_list.append(robot_list[actual_env_id].get_dof_index(x))
                                    common_js_names.append(x)

                            cmd_plan[actual_env_id] = cmd_plan[actual_env_id].get_ordered_joint_state(common_js_names)


                            if actual_env_id in need_plan_envs:
                                need_plan_envs.remove(actual_env_id)

                    cmd_idx = 0
                else:
                    print(
                        f"[Batch Planner] Plan attempt failed for envs {envs_to_plan}: status={result.status}, valid_query={result.valid_query}"
                    )

                    if hasattr(result, 'ik_result') and result.ik_result is not None:
                        ik_success = result.ik_result.success
                        print(f"[Batch Planner] IK success: {ik_success}")
                        if torch.count_nonzero(ik_success) == 0:
                            print(f"[Batch Planner] IK failed for all seeds. Position error: {result.ik_result.position_error}, Rotation error: {result.ik_result.rotation_error}")

                    if hasattr(result, 'trajopt_result') and result.trajopt_result is not None:
                        print(f"[Batch Planner] TrajOpt feasible: {result.trajopt_result.feasible if hasattr(result.trajopt_result, 'feasible') else 'N/A'}")


        for s in range(len(cmd_plan)):
            if cmd_plan[s] is not None and cmd_idx < len(cmd_plan[s].position):
                cmd_state = cmd_plan[s][cmd_idx]



                idx_list_s = []
                common_js_names_s = []
                for x in sim_js_names:
                    if x in cmd_plan[s].joint_names:
                        idx_list_s.append(robot_list[s].get_dof_index(x))
                        common_js_names_s.append(x)



                pos_np = cmd_state.position.view(-1).cpu().numpy()
                pos_np[7] = 0.0
                pos_np[8] = 0.0
                vel_np = cmd_state.velocity.view(-1).cpu().numpy() if cmd_state.velocity is not None else None
                print(vel_np.shape)

                art_action = ArticulationAction(
                    pos_np,
                    vel_np,
                    joint_indices=idx_list_s,
                )


                art_controllers[s].apply_action(art_action)
            else:
                cmd_plan[s] = None
        cmd_idx += 1
        past_goal.copy_(ik_goal)

        for _ in range(2):
            my_world.step(render=False)


if __name__ == "__main__":
    main()

