










import time


import torch


from curobo.geom.types import WorldConfig
from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.wrap.reacher.trajopt import TrajOptSolver, TrajOptSolverConfig


def plot_js(trajectory: JointState):

    import matplotlib.pyplot as plt

    _, axs = plt.subplots(4, 1)
    q = trajectory.position.cpu().numpy()
    qd = trajectory.velocity.cpu().numpy()
    qdd = trajectory.acceleration.cpu().numpy()
    qddd = None
    if trajectory.jerk is not None:
        qddd = trajectory.jerk.cpu().numpy()

    for i in range(q.shape[-1]):
        axs[0].plot(q[:, i], label=str(i))
        axs[1].plot(qd[:, i], label=str(i))
        axs[2].plot(qdd[:, i], label=str(i))
        if qddd is not None:
            axs[3].plot(qddd[:, i], label=str(i))
    plt.legend()
    plt.show()


def plot_traj(trajectory):

    import matplotlib.pyplot as plt

    _, axs = plt.subplots(1, 1)
    q = trajectory

    for i in range(q.shape[-1]):
        axs.plot(q[:, i], label=str(i))
    plt.legend()
    plt.show()


def demo_trajopt_collision_free():
    PLOT = True
    tensor_args = TensorDeviceType()
    world_file = "collision_table.yml"

    robot_file = "franka.yml"
    robot_cfg = RobotConfig.from_dict(
        load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    )
    world_cfg = WorldConfig.from_dict(load_yaml(join_path(get_world_configs_path(), world_file)))

    trajopt_config = TrajOptSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args,
        use_cuda_graph=False,
    )
    trajopt_solver = TrajOptSolver(trajopt_config)
    q_start = trajopt_solver.retract_config

    q_goal = q_start.clone() + 0.1

    kin_state = trajopt_solver.fk(q_goal)
    goal_pose = Pose(kin_state.ee_position, kin_state.ee_quaternion)
    goal_state = JointState.from_position(q_goal)
    current_state = JointState.from_position(q_start)















    print("Running Goal Pose trajopt")
    js_goal = Goal(goal_pose=goal_pose, current_state=current_state)
    result = trajopt_solver.solve_single(js_goal)
    print(result.success)
    if PLOT:
        plot_js(result.solution)


    print("Running Goal Pose Set trajopt")





if __name__ == "__main__":



    demo_trajopt_collision_free()
