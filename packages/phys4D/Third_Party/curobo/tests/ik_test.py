










import torch


from curobo.geom.sdf.world import (
    CollisionCheckerType,
    WorldCollisionConfig,
    WorldPrimitiveCollision,
)
from curobo.geom.sdf.world_mesh import WorldMeshCollision
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig


def test_basic_ik():
    tensor_args = TensorDeviceType()

    config_file = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
    urdf_file = config_file["robot_cfg"]["kinematics"][
        "urdf_path"
    ]
    base_link = config_file["robot_cfg"]["kinematics"]["base_link"]
    ee_link = config_file["robot_cfg"]["kinematics"]["ee_link"]
    robot_cfg = RobotConfig.from_basic(urdf_file, base_link, ee_link, tensor_args)

    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        None,
        rotation_threshold=0.05,
        position_threshold=0.005,
        num_seeds=30,
        self_collision_check=False,
        self_collision_opt=False,
        tensor_args=tensor_args,
    )
    ik_solver = IKSolver(ik_config)
    b_size = 10
    q_sample = ik_solver.sample_configs(b_size)
    kin_state = ik_solver.fk(q_sample)
    goal = Pose(kin_state.ee_position, kin_state.ee_quaternion)
    result = ik_solver.solve_batch(goal)
    success = result.success

    assert torch.count_nonzero(success).item() >= 1.0


def test_full_config_collision_free_ik():
    tensor_args = TensorDeviceType()
    world_file = "collision_cubby.yml"

    robot_file = "franka.yml"
    robot_cfg = RobotConfig.from_dict(
        load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    )
    world_cfg = WorldConfig.from_dict(load_yaml(join_path(get_world_configs_path(), world_file)))
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        rotation_threshold=0.05,
        position_threshold=0.005,
        num_seeds=30,
        self_collision_check=True,
        self_collision_opt=True,
        tensor_args=tensor_args,
    )
    ik_solver = IKSolver(ik_config)
    b_size = 10

    q_sample = ik_solver.sample_configs(b_size)
    kin_state = ik_solver.fk(q_sample)
    goal = Pose(kin_state.ee_position, kin_state.ee_quaternion)
    result = ik_solver.solve(goal)

    success = result.success
    assert torch.count_nonzero(success).item() >= 9.0


def test_attach_object_full_config_collision_free_ik():
    tensor_args = TensorDeviceType()
    world_file = "collision_cubby.yml"

    robot_file = "franka.yml"
    robot_cfg = RobotConfig.from_dict(
        load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    )
    world_cfg = WorldConfig.from_dict(load_yaml(join_path(get_world_configs_path(), world_file)))
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        rotation_threshold=0.05,
        position_threshold=0.005,
        num_seeds=30,
        self_collision_check=True,
        self_collision_opt=True,
        tensor_args=tensor_args,
    )
    ik_solver = IKSolver(ik_config)
    b_size = 10

    q_sample = ik_solver.sample_configs(b_size)
    kin_state = ik_solver.fk(q_sample)
    goal = Pose(kin_state.ee_position, kin_state.ee_quaternion)
    result = ik_solver.solve(goal)

    success = result.success
    assert torch.count_nonzero(success).item() >= 9.0

    q_sample = ik_solver.sample_configs(b_size)
    kin_state = ik_solver.fk(q_sample)
    goal = Pose(kin_state.ee_position, kin_state.ee_quaternion)



    result = ik_solver.solve(goal)
    success = result.success
    assert torch.count_nonzero(success).item() >= 9.0


def test_batch_env_ik():
    tensor_args = TensorDeviceType()
    world_files = ["collision_cubby.yml", "collision_test.yml"]

    robot_file = "franka.yml"
    robot_cfg = RobotConfig.from_dict(
        load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    )
    w_list = [
        WorldConfig.from_dict(load_yaml(join_path(get_world_configs_path(), world_file)))
        for world_file in world_files
    ]
    world_ccheck = WorldPrimitiveCollision(WorldCollisionConfig(tensor_args, n_envs=2))

    world_ccheck.load_batch_collision_model(w_list)
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        world_coll_checker=world_ccheck,
        rotation_threshold=0.05,
        position_threshold=0.005,
        num_seeds=100,
        self_collision_check=True,
        self_collision_opt=True,
        tensor_args=tensor_args,
        use_cuda_graph=True,
    )
    ik_solver = IKSolver(ik_config)
    b_size = 2

    q_sample = ik_solver.sample_configs(b_size)
    kin_state = ik_solver.fk(q_sample)
    goal = Pose(kin_state.ee_position, kin_state.ee_quaternion)
    result = ik_solver.solve_batch_env(goal)

    success = result.success
    assert torch.count_nonzero(success).item() >= 1.0


def test_batch_env_mesh_ik():
    tensor_args = TensorDeviceType()
    world_files = ["collision_table.yml", "collision_table.yml"]

    robot_file = "franka.yml"
    robot_cfg = RobotConfig.from_dict(
        load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    )
    w_list = [
        WorldConfig.from_dict(
            load_yaml(join_path(get_world_configs_path(), world_file))
        ).get_mesh_world()
        for world_file in world_files
    ]
    world_ccheck = WorldMeshCollision(
        WorldCollisionConfig(tensor_args, checker_type=CollisionCheckerType.MESH, n_envs=2)
    )


    world_ccheck.load_batch_collision_model(w_list)
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        world_coll_checker=world_ccheck,
        rotation_threshold=0.05,
        position_threshold=0.005,
        num_seeds=100,
        self_collision_check=True,
        self_collision_opt=True,
        tensor_args=tensor_args,
        use_cuda_graph=True,
    )
    ik_solver = IKSolver(ik_config)
    b_size = 2

    q_sample = ik_solver.sample_configs(b_size)
    kin_state = ik_solver.fk(q_sample)
    goal = Pose(kin_state.ee_position, kin_state.ee_quaternion)
    result = ik_solver.solve_batch_env(goal)

    success = result.success
    assert torch.count_nonzero(success).item() >= 1.0
