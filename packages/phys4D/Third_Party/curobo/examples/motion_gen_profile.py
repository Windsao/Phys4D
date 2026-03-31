
















from torch.profiler import ProfilerActivity, profile


from curobo.geom.sdf.world import CollisionCheckerType
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig


def plot_traj(trajectory):

    import matplotlib.pyplot as plt

    _, axs = plt.subplots(1, 1)
    q = trajectory

    for i in range(q.shape[-1]):
        axs.plot(q[:, i], label=str(i))
    plt.legend()
    plt.show()


def demo_motion_gen():
    PLOT = False
    tensor_args = TensorDeviceType()
    world_file = "collision_test.yml"
    robot_file = "franka.yml"
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_file,
        world_file,
        tensor_args,
        trajopt_tsteps=40,
        collision_checker_type=CollisionCheckerType.PRIMITIVE,
        use_cuda_graph=False,
    )
    motion_gen = MotionGen(motion_gen_config)
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    robot_cfg = RobotConfig.from_dict(robot_cfg, tensor_args)
    retract_cfg = robot_cfg.cspace.retract_config
    state = motion_gen.rollout_fn.compute_kinematics(
        JointState.from_position(retract_cfg.view(1, -1))
    )

    retract_pose = Pose(state.ee_pos_seq.squeeze(), quaternion=state.ee_quat_seq.squeeze())
    start_state = JointState.from_position(retract_cfg.view(1, -1) + 0.5)
    result = motion_gen.plan(start_state, retract_pose, enable_graph=False)


    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        result = motion_gen.plan(start_state, retract_pose, enable_graph=False)

    print("Exporting the trace..")
    prof.export_chrome_trace("trace.json")
    exit(10)
    traj = result.raw_plan
    print("Trajectory Generated: ", result.success)
    if PLOT:
        plot_traj(traj.cpu().numpy())


if __name__ == "__main__":
    demo_motion_gen()
