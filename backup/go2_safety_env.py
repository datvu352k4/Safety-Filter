import torch


class Go2SafetyEnvWrapper:
    def __init__(self, base_env, ll_policy, ll_model, cfg):
        self.base_env = base_env
        self.ll_policy = ll_policy
        self.ll_model = ll_model
        self.cfg = cfg
        self.device = base_env.device

        self.num_envs = base_env.num_envs
        self.num_obs = cfg.env.num_observations
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions
        self.max_episode_length = base_env.max_episode_length

        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.reset_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.episode_length_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.privileged_obs_buf = None
        self.extras = {}

        self.target_dummy_commands = torch.zeros(self.num_envs, 3, device=self.device)
        self.dummy_commands = torch.zeros(self.num_envs, 3, device=self.device)
        self.alpha_scale = torch.zeros(self.num_envs, 3, device=self.device)

        # Buffer lưu action của bước trước để mạng học mượt hơn
        self.last_actions = torch.zeros(
            self.num_envs, self.num_actions, device=self.device
        )

        self.env_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        # FIX LỖI 3: Trả về None chuẩn theo config
        return None

    def reset(self):
        base_obs_new, _, base_obs_history_new, _ = self.base_env.get_observations()
        self._compute_safety_obs(base_obs_new, base_obs_history_new)
        return self.obs_buf, self.obs_buf

    def _compute_safety_obs(self, base_obs, base_obs_history):
        with torch.no_grad():
            try:
                z_t = self.ll_model.history_encoder(base_obs_history)
            except AttributeError:
                z_t = self.ll_model.actor.history_encoder(base_obs_history)

        v_current = self.base_env.simulator.base_lin_vel[:, :2]
        yaw_current = self.base_env.simulator.base_ang_vel[:, 2].unsqueeze(1)
        vel_state = torch.cat([v_current, yaw_current], dim=-1)

        # Cung cấp DUMMY CMD và LAST ACTIONS để mạng biết ngữ cảnh
        self.obs_buf = torch.cat([z_t, vel_state, self.last_actions], dim=-1)

    def step(self, actions):
        # 1. BẮT BỆNH THỜI GIAN: Lưu lại alpha cũ trước khi tính alpha mới
        self.previous_alpha = self.alpha_scale.clone()

        # 2. Tính alpha mới cho bước hiện tại
        self.alpha_scale = (torch.tanh(actions) + 1.0) / 2.0

        # 3. FIX VẤN ĐỀ 1 (Claude): Lưu alpha vào last_actions để nhét vào Observation
        self.last_actions = self.alpha_scale.clone()
        # 3. Cập nhật Dummy Target
        self.env_steps += 1
        resample_envs = self.env_steps % 150 == 0
        if resample_envs.any():
            num_resample = resample_envs.sum()
            self.target_dummy_commands[resample_envs, 0] = (
                torch.rand(num_resample, device=self.device) * 2.4
            ) - 1.2
            self.target_dummy_commands[resample_envs, 1] = (
                torch.rand(num_resample, device=self.device) * 2.0
            ) - 1.0
            self.target_dummy_commands[resample_envs, 2] = (
                torch.rand(num_resample, device=self.device) * 3.0
            ) - 1.5

        # Lọc Low-pass cho mượt
        self.dummy_commands = (
            0.95 * self.dummy_commands + 0.05 * self.target_dummy_commands
        )

        # 4. Tính và truyền safe_cmd
        safe_cmd = self.dummy_commands * self.alpha_scale
        self.base_env.commands[:, :3] = safe_cmd

        # ==========================================================
        # 5. CHẠY VẬT LÝ VÀ FIX LỖI STALE OBSERVATION
        with torch.no_grad():
            # Lấy obs trước khi chạy để tính lệnh low-level
            base_obs_pre, _, base_obs_history_pre, _ = self.base_env.get_observations()
            ll_actions = self.ll_policy(base_obs_pre, base_obs_history_pre)

            # Chạy 1 bước vật lý
            _, _, _, _, base_rews, base_dones, base_infos = self.base_env.step(
                ll_actions
            )

            # Lấy obs SAU KHI VẬT LÝ ĐÃ CHẠY để cấp cho mạng Safety
            base_obs_new, _, base_obs_history_new, _ = self.base_env.get_observations()
        # ==========================================================

        # 6. Cập nhật Safety Obs và Reset buffer
        self._compute_safety_obs(base_obs_new, base_obs_history_new)
        self.reset_buf = base_dones
        self.episode_length_buf += 1

        self._compute_rewards()

        # FIX LỖI 4: Xóa sạch rác (lệnh cũ) khi robot ngã để tránh giật đùng đùng
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.episode_length_buf[env_ids] = 0
            self.env_steps[env_ids] = 0
            self.dummy_commands[env_ids] = 0.0
            self.target_dummy_commands[env_ids] = 0.0
            self.alpha_scale[env_ids] = 0.0
            self.last_actions[env_ids] = 0.0

        return self.obs_buf, self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _compute_rewards(self):
        v_current = self.base_env.simulator.base_lin_vel[:, :2]
        yaw_current = self.base_env.simulator.base_ang_vel[:, 2].unsqueeze(1)
        actual_vel = torch.cat([v_current, yaw_current], dim=-1)

        safe_cmd = self.dummy_commands * self.alpha_scale

        err_x = safe_cmd[:, 0] - actual_vel[:, 0]
        err_y = safe_cmd[:, 1] - actual_vel[:, 1]
        err_yaw = safe_cmd[:, 2] - actual_vel[:, 2]

        # Vẫn khuếch đại lỗi Y và Yaw
        weighted_error_sq = err_x**2 + 3.0 * err_y**2 + 3.0 * err_yaw**2

        # --- 1. Điểm Tracking (Đã nới lỏng temperature thành 0.25) ---
        tracking_score = torch.exp(-weighted_error_sq / 0.25)
        reward_tracking = tracking_score * self.cfg.rewards.scales.safety_tracking

        # --- 2. Thưởng dũng cảm (Độc lập, không nhân với tracking_score) ---
        alpha_magnitude = torch.sum(self.alpha_scale, dim=-1)
        reward_alpha = alpha_magnitude * self.cfg.rewards.scales.safety_alpha_max

        # --- 3. Phạt Trượt / Khắc tinh của đường băng (Slip Detection) ---
        # Khi tracking_score thấp (trượt), cụm (1.0 - tracking_score) sẽ lớn.
        # Nếu lúc này alpha_magnitude vẫn lớn, hình phạt sẽ bị nhân lên cực mạnh!
        slip_signal = (1.0 - tracking_score) * alpha_magnitude
        reward_slip = -slip_signal * self.cfg.rewards.scales.safety_slip_penalty

        # --- 4. Sinh tồn & Ngã ---
        reward_survival = (~self.reset_buf).float() * self.cfg.rewards.scales.survival
        reward_fall = self.reset_buf.float() * self.cfg.rewards.scales.fall_penalty

        alpha_delta_sq = torch.sum(
            torch.square(self.alpha_scale - self.previous_alpha), dim=-1
        )
        reward_action_penalty = -alpha_delta_sq * self.cfg.rewards.scales.action_penalty

        self.rew_buf = (
            reward_tracking
            + reward_alpha
            + reward_slip
            + reward_survival
            + reward_fall
            + reward_action_penalty
        )
