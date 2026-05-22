"""
Safety Filter v2 Configuration — Physics-Based.

Reward architecture:
    R1 foot_slip  = 4.0  — phạt trượt chân (physics-based, dominant)
    R2 vel_alpha  = 1.0  — thưởng vel × alpha (chống collapse)
    R3 smooth     = 0.2  — giữ alpha mượt
"""

from legged_gym.envs.go2.go2_ts.go2_ts_config import Go2TSCfg, Go2TSCfgPPO


class Go2SafetyCfg(Go2TSCfg):

    class env(Go2TSCfg.env):
        num_actions = 3  # alpha cho 3 trục: vx, vy, yaw

        # Actor (deployment): z_t(1) + vel(3) + height(1) + alpha_prev(3) = 8
        num_observations = 8

        # Critic (training): base_obs(45) + vel(3) + z(1) + z_t(1) + alpha(3) + dir(3) = 56
        num_privileged_obs = 56
        num_history_obs = 0

    class domain_rand(Go2TSCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.05, 2.0]

    class rewards(Go2TSCfg.rewards):
        base_height_target = 0.3

        class scales:
            # ═══════════════════════════════════════════════════════════
            # v2 PHYSICS-BASED REWARDS
            #
            # R1 phạt trượt trực tiếp → policy TỰ KHÁM PHÁ alpha tối ưu.
            # R2 chống alpha collapse → đảm bảo robot di chuyển.
            # R3 giữ mượt → deployment ổn định.
            #
            # Equilibrium (alpha tối ưu, không trượt):
            #   R1 ≈ 0, R2 ≈ +0.6 → total ≈ +0.6/step → DƯƠNG ✅
            # ═══════════════════════════════════════════════════════════

            safety_foot_slip = 50.0  # R1: phat truot chan TẬN DIỆT (50.0) vì đã gỡ bỏ hoàn toàn grace_mask bảo kê
            safety_vel_alpha = 1.0  # R2: thưởng vel × alpha
            safety_smooth = 0.2  # R3: giữ alpha mượt

    class terrain(Go2TSCfg.terrain):
        num_rows = 2
        num_cols = 2
        terrain_length = 8.0
        terrain_width = 8.0


class Go2SafetyCfgPPO(Go2TSCfgPPO):

    class policy(Go2TSCfgPPO.policy):
        actor_hidden_dims = [64, 32]
        critic_hidden_dims = [256, 128, 64]
        init_noise_std = 0.3

    class algorithm(Go2TSCfgPPO.algorithm):
        entropy_coef = 0.001
        learning_rate = 3e-4
        max_grad_norm = 1.0

    class runner(Go2TSCfgPPO.runner):
        run_name = "safety_filter_v2"
        experiment_name = "go2_safety"
        max_iterations = 5000
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        save_interval = 200
