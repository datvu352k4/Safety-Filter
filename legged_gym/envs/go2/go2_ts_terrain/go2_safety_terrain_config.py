"""
Safety Filter Configuration — Baseline (per technical report).

Reward architecture:
    R1 foot_slip      = 40.0 — phạt trượt chân, deadzone = 0.15 + 0.6×|yaw|
    R2 vel_alpha      = 1.0  — thưởng mean(alpha), tránh collapse về 0
    R3 smooth         = 1.0  — Asymmetric: 5.0×Δα² tăng tốc / 0.1×Δα² phanh
    R4 orientation    = 20.0 — phạt ngã
    R5 tracking_error = 3.5  — phạt không đạt tốc độ mục tiêu

Actor input:  6D  (z_t(3) + last_alpha(3))
Critic input: 72D (z_t(3)+fric(1)+lin_vel(3)+ang_vel(3)+proj_grav(3)+base_z(1)
                   +dir(3)+alpha(3)+height_var(4)+height_rel_flat(36)+normal(12))
Alpha range:  [0.2, 1.0]
"""

from legged_gym.envs.go2.go2_ts_terrain.go2_ts_terrain_config import (
    Go2TSTerrainCfg,
    Go2TSTerrainCfgPPO,
)


class Go2SafetyTerrainCfg(Go2TSTerrainCfg):

    class env(Go2TSTerrainCfg.env):
        num_actions = 3  # Alpha: vx, vy, yaw
        num_observations = 6  # z_t(3) + last_alpha(3)
        num_privileged_obs = 72  # 72D critic (per report)
        num_history_obs = 0

    class terrain(Go2TSTerrainCfg.terrain):
        curriculum = False
        max_init_terrain_level = 7

    class domain_rand(Go2TSTerrainCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.05, 1.7]
        terrain_friction_ranges = {0.5: [0.05, 1.7], 1.0: [0.2, 1.7]}

    class rewards(Go2TSTerrainCfg.rewards):
        base_height_target = 0.3

        class scales:
            safety_foot_slip = 20.0
            safety_vel_alpha = 1.5
            safety_smooth = 1.5
            safety_orientation = 30.0


class Go2SafetyTerrainCfgPPO(Go2TSTerrainCfgPPO):

    class policy(Go2TSTerrainCfgPPO.policy):
        actor_hidden_dims = [256, 128]
        critic_hidden_dims = [512, 256, 128]
        init_noise_std = 0.3

    class algorithm(Go2TSTerrainCfgPPO.algorithm):
        entropy_coef = 0.003
        learning_rate = 3e-4
        max_grad_norm = 1.0

    class runner(Go2TSTerrainCfgPPO.runner):
        run_name = "safety_filter_terrain_baseline"
        experiment_name = "go2_safety_terrain"
        max_iterations = 5000
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        save_interval = 100
