from legged_gym import *
from legged_gym.envs.base.legged_robot_ts_config import (
    LeggedRobotTSCfg,
    LeggedRobotTSCfgPPO,
)
from legged_gym.envs.base.common_cfgs import Go2RoughCommonCfg


class Go2TSTerrainCfg(Go2RoughCommonCfg):
    class env(Go2RoughCommonCfg.env):
        num_envs = 4096
        num_observations = 45  # num_obs
        num_privileged_obs = 29  # 1 friction + 4 height_var + 4 height_range + 4 height_mean + 4 height_var_ema + 12 normal_vector
        frame_stack = 15
        num_history_obs = int(num_observations * frame_stack)
        num_latent_dims = 3
        c_frame_stack = 5
        single_critic_obs_len = num_observations + 31 + 81 + 17 + 3
        num_critic_obs = c_frame_stack * single_critic_obs_len
        num_actions = 12
        env_spacing = 0.5

    class terrain(Go2RoughCommonCfg.terrain):
        mesh_type = "heightfield"
        curriculum = True
        terrain_proportions = [0.5, 0.5, 0.0, 0.0, 0.0]
        slope_threshold = 0.0

    class init_state(Go2RoughCommonCfg.init_state):
        pass

    class control(Go2RoughCommonCfg.control):
        pass

    class asset(Go2RoughCommonCfg.asset):
        pass

    class rewards(Go2RoughCommonCfg.rewards):
        base_height_target = 0.3

        class scales(Go2RoughCommonCfg.rewards.scales):
            base_height = 0.0
            dof_close_to_default = -0.05
            tracking_lin_vel = 1.5
            tracking_ang_vel = 1.0
            action_rate = -0.005
            action_smoothness = -0.005
            dof_power = -2.0e-5
            dof_acc = -2.0e-8
            lin_vel_z = -0.5

    class commands(Go2RoughCommonCfg.commands):
        curriculum = True
        max_curriculum = 2.0
        num_commands = 4
        resampling_time = 10.0
        heading_command = True

        class ranges(Go2RoughCommonCfg.commands.ranges):
            lin_vel_x = [-0.5, 0.5]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-1.5, 1.5]
            heading = [-3.14, 3.14]

    class domain_rand(Go2RoughCommonCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.05, 1.7]
        terrain_friction_ranges = {0.5: [0.05, 1.7], 1.0: [0.2, 1.7]}
        randomize_base_mass = False
        added_mass_range = [-1.0, 1.0]
        push_robots = False
        push_interval_s = 10
        max_push_vel_xy = 1.0
        randomize_com_displacement = False
        com_pos_x_range = [-0.03, 0.03]
        com_pos_y_range = [-0.03, 0.03]
        com_pos_z_range = [-0.03, 0.03]
        randomize_pd_gain = False
        kp_range = [0.8, 1.2]
        kd_range = [0.8, 1.2]
        randomize_joint_armature = False
        joint_armature_range = [0.015, 0.025]
        randomize_joint_friction = False
        joint_friction_range = [0.01, 0.02]
        randomize_joint_damping = False
        joint_damping_range = [0.25, 0.3]


class Go2TSTerrainCfgPPO(LeggedRobotTSCfgPPO):
    class policy(LeggedRobotTSCfgPPO.policy):
        critic_hidden_dims = [1024, 256, 128]
        privilege_encoder_hidden_dims = [256, 128]
        history_encoder_type = "MLP"  # "MLP" or "TCN"
        history_encoder_hidden_dims = [256, 128]  # for MLP
        history_encoder_channel_dims = [1, 1, 1, 1]  # for TCN
        history_encoder_dilation = [1, 1, 2, 1]  # for TCN
        history_encoder_stride = [1, 2, 1, 2]  # for TCN
        history_encoder_final_layer_dim = 128  # for TCN
        kernel_size = 5

    class algorithm(LeggedRobotTSCfgPPO.algorithm):
        encoder_lr = 2.0e-4
        num_encoder_epochs = 2

    class runner(LeggedRobotTSCfgPPO.runner):
        run_name = "ts_terrain"
        if SIMULATOR == "genesis":
            run_name += "_genesis"
        elif SIMULATOR == "isaacgym":
            run_name += "_isaacgym"
        elif SIMULATOR == "isaaclab":
            run_name += "_isaaclab"
        experiment_name = "go2_rough_terrain"
        save_interval = 100
        max_iterations = 3000
