from legged_gym.envs.go2.go2_ts.go2_ts_config import Go2TSCfg, Go2TSCfgPPO


class Go2SafetyCfg(Go2TSCfg):
    class env(Go2TSCfg.env):
        num_actions = 3  # hệ số scale alpha cho x, y, yaw

        # Kích thước = 99 (z_t) + 3 (v_current) + 3 (last_actions) = 105
        num_observations = 105

        num_privileged_obs = None  # Mạng an toàn không cần privileged_obs
        num_history_obs = 0  # Không dùng frame_stack cho mạng Safety

    class domain_rand(Go2TSCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.5]  # Từ cực trơn (băng) đến nhám

    class rewards(Go2TSCfg.rewards):
        class scales:
            safety_alpha_max = 1.0  # Đã tăng lên để tạo lực kéo mạnh
            safety_tracking = 3.0  # Điểm bám sát cơ bản
            safety_slip_penalty = 1.5  # (MỚI) Hệ số phạt khi trượt
            survival = 0.1
            fall_penalty = -5.0  # Đã giảm bớt độ "sốc"
            action_penalty = 0.05  # Phạt rất nhẹ để tạo lực kéo mượt


class Go2SafetyCfgPPO(Go2TSCfgPPO):
    class policy(Go2TSCfgPPO.policy):
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [256, 128, 64]
        init_noise_std = 0.5  # Bắt đầu với noise nhỏ để đỡ giật cục

    class algorithm(Go2TSCfgPPO.algorithm):
        entropy_coef = 0.001  # Bóp nghẹt phần thưởng khám phá
        learning_rate = 1e-4  # Học chậm lại cho chắc cú
        max_grad_norm = 1.0  # Chống nổ gradient

    class runner(Go2TSCfgPPO.runner):
        run_name = "safety_filter"
        experiment_name = "go2_safety"
        max_iterations = 3000
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
