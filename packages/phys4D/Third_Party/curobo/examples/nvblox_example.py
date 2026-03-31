
















from curobo.geom.sdf.world import CollisionCheckerType
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig


def plot_traj(trajectory):

    import matplotlib.pyplot as plt

    _, axs = plt.subplots(1, 1)
    q = trajectory

    for i in range(q.shape[-1]):
        axs.plot(q[:, i], label=str(i))
    plt.legend()
    plt.show()


def plot_iters_traj(trajectory, d_id=1, dof=7, seed=0):

    import matplotlib.pyplot as plt

    _, axs = plt.subplots(len(trajectory), 1)
    if len(trajectory) == 1:
        axs = [axs]
    for k in range(len(trajectory)):
        q = trajectory[k]

        for i in range(len(q)):
            axs[k].plot(
                q[i][seed, :-1, d_id].cpu(),
                "r+-",
                label=str(i),
                alpha=0.1 + min(0.9, float(i) / (len(q))),
            )
    plt.legend()
    plt.show()


def plot_iters_traj_3d(trajectory, d_id=1, dof=7, seed=0):

    import matplotlib.pyplot as plt

    ax = plt.axes(projection="3d")
    c = 0
    h = trajectory[0][0].shape[1] - 1
    x = [x for x in range(h)]

    for k in range(len(trajectory)):
        q = trajectory[k]

        for i in range(len(q)):

            ax.scatter3D(
                x, [c for _ in range(h)], q[i][seed, :h, d_id].cpu(), c=q[i][seed, :, d_id].cpu()
            )

            c += 1

    plt.show()


def demo_motion_gen_nvblox():
    PLOT = True
    tensor_args = TensorDeviceType()
    world_file = "collision_nvblox.yml"
    robot_file = "franka.yml"
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_file,
        world_file,
        tensor_args,
        trajopt_tsteps=32,
        collision_checker_type=CollisionCheckerType.BLOX,
        use_cuda_graph=False,
        num_trajopt_seeds=2,
        num_graph_seeds=2,
        evaluate_interpolated_trajectory=True,
    )
    goals = tensor_args.to_device(
        [
            [0.5881, 0.0589, 0.3055],
            [0.5881, 0.4155, 0.3055],
            [0.5881, 0.4155, 0.1238],
            [0.5881, -0.4093, 0.1238],
            [0.7451, 0.0287, 0.2539],
        ]
    ).view(-1, 3)

    motion_gen = MotionGen(motion_gen_config)
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    robot_cfg = RobotConfig.from_dict(robot_cfg, tensor_args)
    motion_gen.warmup()
    print("ready")





if __name__ == "__main__":
    demo_motion_gen_nvblox()
