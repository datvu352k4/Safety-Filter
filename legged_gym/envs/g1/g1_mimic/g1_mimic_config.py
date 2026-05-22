from legged_gym import *
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym.envs.base.common_cfgs import G1MimicCommonCfg

"""
Unitree G1 DeepMimic environment configuration file.

Attention:
    1. This task is only valid in Genesis and IsaacLab simulators. IsaacGym has some
        bugs related to the rigid body state update after resetting the environment, 
        preventing the task from working correctly.
"""
class G1MimicCfg(G1MimicCommonCfg):
    class env(LeggedRobotCfg.env):
        frame_stack = 5
        ref_motion_frame_stack = 1
        ref_motion_single_obs = 125
        num_single_obs = 151 + int(ref_motion_single_obs * ref_motion_frame_stack)
        num_observations = int(num_single_obs * frame_stack)
        c_frame_stack = 5
        num_single_critic_obs = num_single_obs + 17
        num_privileged_obs = int(num_single_critic_obs * c_frame_stack)
        num_actions = 29
        # reference motion file, should be a .pkl file containing a dictionary
        motion_file = '02_01_walk_stageii_60hz_isaacgym.pkl'
        episode_length_s = 10 
        debug_draw_key_body_points = True # draw key body points for mimic tasks
        max_projected_gravity = -0.3
        
    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.2, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 2.]
        push_robots = True
        push_interval_s = 10
        max_push_vel_xy = 0.5
        randomize_com_displacement = True
        com_pos_x_range = [-0.05, 0.05]
        com_pos_y_range = [-0.05, 0.05]
        com_pos_z_range = [-0.05, 0.05]
        # Apply random push forces to the links of the robot
        push_links = True
        max_push_force = 10.0 # [N], maximum magnitude of the random push force applied to each link
        push_links_interval_s = 2.0 # time interval between random pushes

    class control(G1MimicCommonCfg.control):
        dt = 1/60.0

    class asset(G1MimicCommonCfg.asset):
        pass

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78
        tracking_dof_pos_sigma = 4.0
        tracking_dof_vel_sigma = 100.0
        tracking_ref_root_pose_sigma = 0.2
        tracking_ref_root_vel_sigma = 1.0
        tracking_ref_key_pos_sigma = 0.1
        only_positive_rewards = False
        class scales(LeggedRobotCfg.rewards.scales):
            # limits
            dof_pos_limits = -5.0
            # tasks
            tracking_ref_dof_pos = 0.5 * 2
            tracking_ref_dof_vel = 0.1 * 2
            tracking_ref_root_pose = 0.5 * 2
            tracking_ref_root_vel = 0.1 * 2
            tracking_ref_key_pos = 0.15 * 2
            # regularization
            ang_vel_xy = -0.05
            dof_acc = -5.e-8
            dof_power = -5.e-6
            collision = -1.0
            action_rate = -0.01
            feet_slip = -0.5
    
    class normalization(LeggedRobotCfg.normalization):
        clip_actions = 100.0
    
    class sim(LeggedRobotCfg.sim):
        dt = 1/240.0

class G1MimicCfgPPO(LeggedRobotCfgPPO):
    class policy(LeggedRobotCfgPPO.policy):
        clip_actions = G1MimicCfg.normalization.clip_actions
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [1024, 256, 128]
        activation = 'elu'
        
    class runner(LeggedRobotCfgPPO.runner):
        num_steps_per_env = 32
        max_iterations = 3000
        run_name = f'{SIMULATOR}'
        experiment_name = 'g1_mimic'
        save_interval = 500
