



"""Launch Isaac Sim Simulator first."""


import sys

if sys.platform != "win32":
    import pinocchio
    import pinocchio as pin
else:
    import pinocchio
    import pinocchio as pin

from isaaclab.app import AppLauncher


simulation_app = AppLauncher(headless=True).app

"""Unit tests for NullSpacePostureTask with simplified robot configuration using Pink library directly."""

import numpy as np

import pytest
from pink.configuration import Configuration
from pink.tasks import FrameTask
from pinocchio.robot_wrapper import RobotWrapper

from isaaclab.controllers.pink_ik.null_space_posture_task import NullSpacePostureTask


class TestNullSpacePostureTaskSimplifiedRobot:
    """Test cases for NullSpacePostureTask with simplified robot configuration."""

    @pytest.fixture
    def num_joints(self):
        """Number of joints in the simplified robot."""
        return 20

    @pytest.fixture
    def joint_configurations(self):
        """Pre-generated joint configurations for testing."""

        np.random.seed(42)

        return {
            "random": np.random.uniform(-0.5, 0.5, 20),
            "controlled_only": np.array([0.5] * 5 + [0.0] * 15),
            "sequential": np.linspace(0.1, 2.0, 20),
        }

    @pytest.fixture
    def robot_urdf(self):
        """Load the simplified test robot URDF file."""
        import os

        current_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.join(current_dir, "simplified_test_robot.urdf")
        return urdf_path

    @pytest.fixture
    def robot_configuration(self, robot_urdf):
        """Simplified robot wrapper."""
        wrapper = RobotWrapper.BuildFromURDF(robot_urdf, None, root_joint=None)
        return Configuration(wrapper.model, wrapper.data, wrapper.q0)

    @pytest.fixture
    def tasks(self):
        """pink tasks."""
        return [
            FrameTask("left_hand_pitch_link", position_cost=1.0, orientation_cost=1.0),
            NullSpacePostureTask(
                cost=1.0,
                controlled_frames=["left_hand_pitch_link"],
                controlled_joints=[
                    "waist_yaw_joint",
                    "waist_pitch_joint",
                    "waist_roll_joint",
                    "left_shoulder_pitch_joint",
                    "left_shoulder_roll_joint",
                ],
            ),
        ]

    def test_null_space_jacobian_zero_end_effector_velocity(
        self, robot_configuration, tasks, joint_configurations, num_joints
    ):
        """Test that velocities projected through null space Jacobian result in zero end-effector velocity."""

        robot_configuration.q = joint_configurations["random"]


        frame_task = tasks[0]

        position = np.array([0.5, 0.3, 0.8])
        quaternion = pin.Quaternion(1.0, 0.0, 0.0, 0.0)
        target_pose = pin.SE3(quaternion, position)
        frame_task.set_target(target_pose)


        null_space_task = tasks[1]
        target_posture = np.zeros(num_joints)
        null_space_task.set_target(target_posture)


        null_space_jacobian = null_space_task.compute_jacobian(robot_configuration)


        frame_task_jacobian = frame_task.compute_jacobian(robot_configuration)


        for _ in range(10):

            random_velocity = np.random.randn(num_joints) * 0.1


            null_space_velocity = null_space_jacobian @ random_velocity


            ee_velocity = frame_task_jacobian @ null_space_velocity


            assert np.allclose(ee_velocity, np.zeros(6), atol=1e-7), f"End-effector velocity not zero: {ee_velocity}"

    def test_null_space_jacobian_properties(self, robot_configuration, tasks, joint_configurations, num_joints):
        """Test mathematical properties of the null space Jacobian."""

        robot_configuration.q = joint_configurations["random"]


        frame_task = tasks[0]

        position = np.array([0.3, 0.4, 0.6])
        quaternion = pin.Quaternion(0.707, 0.0, 0.0, 0.707)
        target_pose = pin.SE3(quaternion, position)
        frame_task.set_target(target_pose)


        null_space_task = tasks[1]
        target_posture = np.zeros(num_joints)
        target_posture[0:5] = [0.1, -0.1, 0.2, -0.2, 0.0]
        null_space_task.set_target(target_posture)


        null_space_jacobian = null_space_task.compute_jacobian(robot_configuration)
        ee_jacobian = robot_configuration.get_frame_jacobian("left_hand_pitch_link")



        null_space_projection = null_space_jacobian @ ee_jacobian.T
        assert np.allclose(
            null_space_projection, np.zeros_like(null_space_projection), atol=1e-7
        ), f"Null space projection of end-effector Jacobian not zero: {null_space_projection}"

    def test_null_space_jacobian_identity_when_no_frame_tasks(
        self, robot_configuration, joint_configurations, num_joints
    ):
        """Test that null space Jacobian is identity when no frame tasks are defined."""

        null_space_task = NullSpacePostureTask(cost=1.0, controlled_frames=[], controlled_joints=[])


        robot_configuration.q = joint_configurations["sequential"]


        target_posture = np.zeros(num_joints)
        null_space_task.set_target(target_posture)


        null_space_jacobian = null_space_task.compute_jacobian(robot_configuration)


        expected_identity = np.eye(num_joints)
        assert np.allclose(
            null_space_jacobian, expected_identity
        ), f"Null space Jacobian should be identity when no frame tasks defined: {null_space_jacobian}"

    def test_null_space_jacobian_consistency_across_configurations(
        self, robot_configuration, tasks, joint_configurations, num_joints
    ):
        """Test that null space Jacobian is consistent across different joint configurations."""

        test_configs = [
            np.zeros(num_joints),
            joint_configurations["controlled_only"],
            joint_configurations["random"],
        ]


        frame_task = tasks[0]

        position = np.array([0.3, 0.3, 0.5])
        quaternion = pin.Quaternion(1.0, 0.0, 0.0, 0.0)
        target_pose = pin.SE3(quaternion, position)
        frame_task.set_target(target_pose)


        null_space_task = tasks[1]
        target_posture = np.zeros(num_joints)
        null_space_task.set_target(target_posture)

        jacobians = []
        for config in test_configs:
            robot_configuration.q = config
            jacobian = null_space_task.compute_jacobian(robot_configuration)
            jacobians.append(jacobian)


            ee_jacobian = robot_configuration.get_frame_jacobian("left_hand_pitch_link")


            random_velocity = np.random.randn(num_joints) * 0.1
            null_space_velocity = jacobian @ random_velocity
            ee_velocity = ee_jacobian @ null_space_velocity

            assert np.allclose(
                ee_velocity, np.zeros(6), atol=1e-7
            ), f"End-effector velocity not zero for configuration {config}: {ee_velocity}"

    def test_compute_error_without_target(self, robot_configuration, joint_configurations):
        """Test that compute_error raises ValueError when no target is set."""
        null_space_task = NullSpacePostureTask(
            cost=1.0,
            controlled_frames=["left_hand_pitch_link"],
            controlled_joints=["waist_yaw_joint", "waist_pitch_joint"],
        )

        robot_configuration.q = joint_configurations["sequential"]


        with pytest.raises(ValueError, match="No posture target has been set"):
            null_space_task.compute_error(robot_configuration)

    def test_joint_masking(self, robot_configuration, joint_configurations, num_joints):
        """Test that joint mask correctly filters only controlled joints."""

        controlled_joint_names = ["waist_pitch_joint", "left_shoulder_pitch_joint", "left_elbow_pitch_joint"]


        null_space_task = NullSpacePostureTask(
            cost=1.0, controlled_frames=["left_hand_pitch_link"], controlled_joints=controlled_joint_names
        )


        joint_names = robot_configuration.model.names.tolist()[1:]
        joint_indexes = [joint_names.index(jn) for jn in controlled_joint_names]


        current_config = joint_configurations["sequential"]
        target_config = np.zeros(num_joints)

        robot_configuration.q = current_config
        null_space_task.set_target(target_config)


        error = null_space_task.compute_error(robot_configuration)



        expected_error = np.zeros(num_joints)
        for i in joint_indexes:
            expected_error[i] = current_config[i]

        assert np.allclose(
            error, expected_error, atol=1e-7
        ), f"Joint mask not working correctly: expected {expected_error}, got {error}"

    def test_empty_controlled_joints(self, robot_configuration, joint_configurations, num_joints):
        """Test behavior when controlled_joints is empty."""
        null_space_task = NullSpacePostureTask(
            cost=1.0, controlled_frames=["left_hand_pitch_link"], controlled_joints=[]
        )

        current_config = joint_configurations["sequential"]
        target_config = np.zeros(num_joints)

        robot_configuration.q = current_config
        null_space_task.set_target(target_config)


        error = null_space_task.compute_error(robot_configuration)
        expected_error = np.zeros(num_joints)
        assert np.allclose(error, expected_error), f"Error should be zero when no joints controlled: {error}"

    def test_set_target_from_configuration(self, robot_configuration, joint_configurations):
        """Test set_target_from_configuration method."""
        null_space_task = NullSpacePostureTask(
            cost=1.0,
            controlled_frames=["left_hand_pitch_link"],
            controlled_joints=["waist_yaw_joint", "waist_pitch_joint"],
        )


        test_config = joint_configurations["sequential"]
        robot_configuration.q = test_config


        null_space_task.set_target_from_configuration(robot_configuration)


        assert null_space_task.target_q is not None
        assert np.allclose(null_space_task.target_q, test_config)

    def test_multiple_frame_tasks(self, robot_configuration, joint_configurations, num_joints):
        """Test null space projection with multiple frame tasks."""

        null_space_task = NullSpacePostureTask(
            cost=1.0,
            controlled_frames=["left_hand_pitch_link", "right_hand_pitch_link"],
            controlled_joints=["waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"],
        )

        current_config = joint_configurations["sequential"]
        robot_configuration.q = current_config


        null_space_jacobian = null_space_task.compute_jacobian(robot_configuration)


        jacobian_left_hand = robot_configuration.get_frame_jacobian("left_hand_pitch_link")
        jacobian_right_hand = robot_configuration.get_frame_jacobian("right_hand_pitch_link")


        for _ in range(5):
            random_velocity = np.random.randn(num_joints) * 0.1
            null_space_velocity = null_space_jacobian @ random_velocity


            ee_velocity_left = jacobian_left_hand @ null_space_velocity
            ee_velocity_right = jacobian_right_hand @ null_space_velocity

            assert np.allclose(
                ee_velocity_left, np.zeros(6), atol=1e-7
            ), f"Left hand velocity not zero: {ee_velocity_left}"
            assert np.allclose(
                ee_velocity_right, np.zeros(6), atol=1e-7
            ), f"Right hand velocity not zero: {ee_velocity_right}"
