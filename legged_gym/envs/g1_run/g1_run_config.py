from legged_gym import *
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class G1RunCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = (
            80  # 9 (IMU+cmd) + 23 (pos) + 23 (vel) + 23 (action) + 2 (phase) = 80
        )
        num_privileged_obs = 83  # 80 + 3 (base_lin_vel)
        num_actions = 23

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        measure_heights = False

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.8]  # x,y,z [m]
        default_joint_angles = {
            "left_hip_pitch_joint": -0.1,
            "left_hip_roll_joint": 0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.3,
            "left_ankle_pitch_joint": -0.2,
            "left_ankle_roll_joint": 0,
            "right_hip_pitch_joint": -0.1,
            "right_hip_roll_joint": 0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.3,
            "right_ankle_pitch_joint": -0.2,
            "right_ankle_roll_joint": 0,
            "waist_yaw_joint": 0.0,
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.5,  # Slightly bent elbow
            "left_wrist_roll_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.5,
            "right_wrist_roll_joint": 0.0,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = "P"
        stiffness = {
            "hip_yaw": 100,
            "hip_roll": 100,
            "hip_pitch": 120,
            "knee": 180,
            "ankle": 40,
            "waist": 100,
            "shoulder": 40,
            "elbow": 40,
            "wrist": 20,
        }
        damping = {
            "hip_yaw": 2.0,
            "hip_roll": 2.0,
            "hip_pitch": 3.0,
            "knee": 5.0,
            "ankle": 2.0,
            "waist": 2.0,
            "shoulder": 1.0,
            "elbow": 1.0,
            "wrist": 0.5,
        }
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        name = "g1_run"
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/unitree_robotics/g1_description/g1_23dof.urdf"
        foot_name = "ankle_roll"
        key_bodies = ["ankle_roll", "knee", "hip", "pelvis", "shoulder", "elbow"]
        penalize_contacts_on = ["hip", "knee", "shoulder", "elbow", "wrist", "pelvis"]
        terminate_after_contacts_on = ["pelvis"]
        base_link_name = "pelvis"
        self_collisions = 0
        flip_visual_attachments = False
        dof_names = [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
        ]

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = False
        friction_range = [0.1, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1.0, 3.0]
        push_robots = False
        push_interval_s = 5
        max_push_vel_xy = 1.0

    class commands(LeggedRobotCfg.commands):
        curriculum = True
        max_curriculum = 2.0
        num_commands = 4  # lin_vel_x, lin_vel_y, ang_vel_yaw, heading
        resampling_time = 10.0  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [-0.5, 0.5]  # high speed will be reached through curriculum
            lin_vel_y = [-0.1, 0.1]
            ang_vel_yaw = [-0.3, 0.3]
            heading = [-3.14, 3.14]

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78

        class scales(LeggedRobotCfg.rewards.scales):
            tracking_lin_vel = 1.5  # Encourage following velocity strictly
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = (
                -0.1
            )  # Heavily penalize torso twisting (encourages arms to counter yaw)
            orientation = -2.0  # Keep torso upright
            base_height = -10.0
            dof_acc = -2.5e-7
            dof_vel = -5e-4
            feet_air_time = 2.0  # Encourage flight phase
            collision = 0.0
            action_rate = -0.01
            dof_pos_limits = -5.0
            alive = 0.15
            hip_pos = -1.0
            contact_no_vel = -0.2
            feet_swing_height = (
                -2.0
            )  # Clear ground to avoid tripping, reduced for high speed running
            contact = 0.18
            arm_swing = 0.02


class G1RunCfgPPO(LeggedRobotCfgPPO):
    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [64]
        critic_hidden_dims = [64]
        activation = "elu"
        rnn_type = "lstm"
        rnn_hidden_size = 64
        rnn_num_layers = 1

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = "ActorCriticRecurrent"
        max_iterations = 10000
        run_name = ""
        experiment_name = "g1_run"
