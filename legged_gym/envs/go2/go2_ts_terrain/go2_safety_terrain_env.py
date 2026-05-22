import math
import torch


class Go2SafetyTerrainEnvWrapper:
    """Wrapper for safety filter training on terrain."""

    # -- Constants --
    MAX_CMD_VEL = 2.0  # Max X velocity (m/s)
    MAX_CMD_LAT = 1.0  # Max Y velocity (m/s)
    MAX_CMD_YAW = 1.5  # Max Yaw rate (rad/s)

    REWARD_CLIP = 5.0  # Reward clipping threshold

    # Contact threshold for reliable stance detection (Newtons)
    CONTACT_THRESHOLD = 10

    # Mô phỏng MPPI: Chu kỳ đổi lệnh dao động ngẫu nhiên từ 20 đến 150 step
    # Vừa tạo cơ hội chay bứt tốc max, vừa tạo những cú bẻ lái/phanh gấp bất ngờ.

    # Initial grace period after command change (steps)
    GRACE_STEPS = 8

    def __init__(self, base_env, ll_policy, ll_model, cfg):
        self.base_env = base_env
        self.ll_policy = ll_policy
        self.ll_model = ll_model
        self.cfg = cfg
        self.device = base_env.device

        self.num_envs = base_env.num_envs
        self.num_obs = cfg.env.num_observations  # 6: z_t(3) + alpha_prev(3)
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions
        self.max_episode_length = base_env.max_episode_length

        # Buffers
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device)
        self.privileged_obs_buf = torch.zeros(
            self.num_envs, self.num_privileged_obs, device=self.device
        )
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.reset_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.episode_length_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.extras = {}

        # Alpha initialization
        self.alpha_scale = torch.ones(self.num_envs, 3, device=self.device)
        self.last_actions = torch.ones(self.num_envs, 3, device=self.device)
        self._prev_alpha = torch.ones(self.num_envs, 3, device=self.device)

        # Track fall status
        self.is_fallen_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        # Randomized resample intervals per environment
        self.resample_intervals = torch.randint(
            20, 150, (self.num_envs,), device=self.device
        )

        # Command direction initialization
        self.training_dir = torch.ones(self.num_envs, 3, device=self.device)
        self.env_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        # Grace period counter
        self.steps_since_resample = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        # Latent buffer
        self._num_latent_dims = base_env.cfg.env.num_latent_dims
        self._z_t = torch.zeros(
            self.num_envs, self._num_latent_dims, device=self.device
        )

        # EMA buffer for terrain roughness (same alpha as locomotion env)
        self.height_var_ema = torch.zeros(self.num_envs, 4, device=self.device)
        self.HEIGHT_VAR_EMA_ALPHA = 0.05

        self._V_MAX = torch.tensor(
            [self.MAX_CMD_VEL, self.MAX_CMD_LAT, self.MAX_CMD_YAW],
            device=self.device,
        ).unsqueeze(0)

        self._feet_indices = base_env.simulator._feet_indices

        # Logging sums
        self.episode_sums = {
            "rew_foot_slip": torch.zeros(self.num_envs, device=self.device),
            "rew_vel_alpha": torch.zeros(self.num_envs, device=self.device),
            "rew_smooth": torch.zeros(self.num_envs, device=self.device),
            "rew_orientation": torch.zeros(self.num_envs, device=self.device),
            "rew_tracking": torch.zeros(self.num_envs, device=self.device),
            "mean_alpha": torch.zeros(self.num_envs, device=self.device),
            "mean_slip": torch.zeros(self.num_envs, device=self.device),
            "mean_vel_x": torch.zeros(self.num_envs, device=self.device),
            "mean_vel_yaw": torch.zeros(self.num_envs, device=self.device),
            "mean_z_t": torch.zeros(self.num_envs, device=self.device),
        }

        self._resample_direction(torch.arange(self.num_envs, device=self.device))

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def reset(self):
        base_obs, _, base_obs_history, _ = self.base_env.get_observations()
        self._compute_safety_obs(base_obs, base_obs_history)
        return self.obs_buf, self.privileged_obs_buf

    def step(self, actions):
        # 1. Update alpha from policy actions
        self._apply_alpha(actions)

        # 2. Randomly resample command direction per environment
        resample_mask = self.steps_since_resample >= self.resample_intervals
        if resample_mask.any():
            self._resample_direction(resample_mask.nonzero(as_tuple=True)[0])

        # 3. Apply scaled commands to base environment
        safe_cmd = self.alpha_scale * self._V_MAX * self.training_dir
        self.base_env.commands[:, :3] = safe_cmd

        # 4. Advance locomotion policy
        with torch.no_grad():
            base_obs, _, base_obs_history, _ = self.base_env.get_observations()
            ll_actions = self.ll_policy(base_obs, base_obs_history)
            _, _, _, _, _, base_dones, _ = self.base_env.step(ll_actions)
            base_obs_new, _, base_obs_history_new, _ = self.base_env.get_observations()

        # 5. Update observations, rewards, and handle resets
        self._compute_safety_obs(base_obs_new, base_obs_history_new)
        self.reset_buf = base_dones
        self.episode_length_buf += 1
        self.env_steps += 1

        self._compute_rewards()
        self._handle_resets()

        return (
            self.obs_buf,
            self.privileged_obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
        )

    def _apply_alpha(self, actions):
        """Convert policy output to alpha (Direct execution for End-to-End learned smoothing)."""
        # actions are raw output from Actor NN (pre-tanh)
        raw = (torch.tanh(actions) + 1.0) / 2.0
        self.alpha_scale = raw.clamp(0.2, 1.0)  # [0.2, 1.0] range as per report
        self.last_actions = self.alpha_scale.clone()

    def _resample_direction(self, env_ids):
        """Randomly sample new command directions."""
        n = env_ids.shape[0]
        roll = torch.rand(n, device=self.device)
        new_dir = torch.zeros(n, 3, device=self.device)

        # Cấu hình phân phối lệnh theo yêu cầu người dùng:
        # 25% X, 25% Y, 25% Yaw, 15% X+Y, 10% Zero

        # 1. Chỉ chạy X (25%)
        mask_x = roll < 0.25
        if mask_x.any():
            new_dir[mask_x, 0] = (
                torch.randint(0, 2, (mask_x.sum().item(),), device=self.device) * 2 - 1
            ).float()

        # 2. Chỉ chạy Y (25%)
        mask_y = (roll >= 0.25) & (roll < 0.50)
        if mask_y.any():
            new_dir[mask_y, 1] = (
                torch.randint(0, 2, (mask_y.sum().item(),), device=self.device) * 2 - 1
            ).float()

        # 3. Chỉ quay Yaw (25%)
        mask_w = (roll >= 0.50) & (roll < 0.75)
        if mask_w.any():
            new_dir[mask_w, 2] = (
                torch.randint(0, 2, (mask_w.sum().item(),), device=self.device) * 2 - 1
            ).float()

        # 4. Kết hợp X + Y (15%)
        mask_xy = (roll >= 0.75) & (roll < 0.90)
        if mask_xy.any():
            new_dir[mask_xy, 0] = (
                torch.randint(0, 2, (mask_xy.sum().item(),), device=self.device) * 2 - 1
            ).float()
            new_dir[mask_xy, 1] = (
                torch.randint(0, 2, (mask_xy.sum().item(),), device=self.device) * 2 - 1
            ).float()

        # 5. Đứng yên (10%) : [0.90 - 1.0] -> Đã được khởi tạo là zeros ở trên.

        self.training_dir[env_ids] = new_dir
        self.steps_since_resample[env_ids] = 0

        # Sinh thời gian sống ngẫu nhiên mới cho chu kỳ tiếp theo (20 - 150 step)
        self.resample_intervals[env_ids] = torch.randint(
            20, 150, (len(env_ids),), device=self.device
        )

    def _compute_safety_obs(self, base_obs, base_obs_history):
        """Compute actor and critic observations."""
        with torch.no_grad():
            try:
                z_t = self.ll_model.history_encoder(base_obs_history)
            except AttributeError:
                z_t = self.ll_model.actor.history_encoder(base_obs_history)

        self._z_t = z_t
        base_lin_vel = self.base_env.simulator.base_lin_vel[:, :3]
        base_ang_vel = self.base_env.simulator.base_ang_vel[:, :3]
        projected_gravity = self.base_env.simulator.projected_gravity[:, :3]
        base_z = self.base_env.simulator.base_pos[:, 2].unsqueeze(1)
        friction = self.base_env.simulator._friction_values

        # Terrain statistics (baseline per technical report)
        height_rel = (
            self.base_env.simulator.feet_pos[:, :, 2].unsqueeze(-1)
            - self.base_env.simulator.height_around_feet
        ).clip(
            -1.0, 1.0
        )  # (N, 4, 9)
        height_var = torch.var(height_rel, dim=-1)  # (N, 4)
        height_rel_flat = height_rel.view(height_rel.shape[0], -1)  # (N, 36)
        normal_vecs = self.base_env.simulator.normal_vector_around_feet  # (N, 12)

        # Actor obs: z_t(3) + last_alpha(3) = 6D
        self.obs_buf = torch.cat([z_t, self.last_actions], dim=-1)

        # Critic privileged obs (72D):
        #   z_t(3) + fric(1) + lin_vel(3) + ang_vel(3) + proj_grav(3) + base_z(1)
        #   + dir(3) + alpha(3) + height_var(4) + height_rel_flat(36) + normal(12) = 72
        self.privileged_obs_buf = torch.cat(
            [
                z_t,  # 3
                friction,  # 1
                base_lin_vel,  # 3
                base_ang_vel,  # 3
                projected_gravity,  # 3
                base_z,  # 1
                self.training_dir,  # 3
                self.last_actions,  # 3
                height_var,  # 4
                height_rel_flat,  # 36
                normal_vecs,  # 12
            ],
            dim=-1,
        )

    def _compute_rewards(self):
        """Compute Reward components."""
        self.steps_since_resample += 1
        s = self.cfg.rewards.scales
        sim = self.base_env.simulator

        alpha_mean = self.alpha_scale.mean(dim=-1)
        z_t_val = self._z_t.mean(dim=-1)

        # Grace mask
        grace_mask = (self.steps_since_resample.float() / self.GRACE_STEPS).clamp(
            0.0, 1.0
        )

        # R1: Foot slip penalty (Deadzone phân tách trục, chỉ nới lỏng khi Xoay)
        # Chân đã chạm đất (Stance) thì vận tốc so với mặt đất phải tiến về 0.
        # Chúng ta chỉ cho phép "miết" chân khi robot đang thực hiện lệnh xoay (Yaw).
        W_MAG = self.training_dir[:, 2].abs()
        YAW_ALLOWANCE = 0.05 * W_MAG  # Bán kính chân Go2 ~0.25m

        # Deadzone cơ sở cực thấp (0.15) để bắt trượt Ice khi phanh gấp
        DZ_X = (0.32 + YAW_ALLOWANCE).unsqueeze(1)
        DZ_Y = (0.08 + YAW_ALLOWANCE).unsqueeze(1)

        feet_vel_xy = sim._feet_vel[
            :, :, :2
        ]  # (N, 4, 2) - Luôn dùng [:, :, :2] vì _feet_vel đã lọc 4 chân
        feet_contact_force = sim._link_contact_forces[:, self._feet_indices, :]
        contact_mask = torch.norm(feet_contact_force, dim=-1) > self.CONTACT_THRESHOLD

        # Tính trượt riêng cho từng trục (Robot Frame)
        slip_x = (feet_vel_xy[:, :, 0].abs() - DZ_X).clamp(min=0.0)
        slip_y = (feet_vel_xy[:, :, 1].abs() - DZ_Y).clamp(min=0.0)

        # Tổng hợp trượt (chỉ xét chân đang chạm đất)
        total_slip = ((slip_x + slip_y) * contact_mask).sum(dim=-1)

        rew_slip = -(total_slip * grace_mask) * s.safety_foot_slip

        # R2: Alpha Maximization Reward
        # Khuyến khích Alpha lúc nào cũng phải to nhất (1.0)
        # Sử dụng tổng (sum) thay vì trung bình (mean) để mỗi trục tự chịu trách nhiệm về điểm số của mình,
        # tránh tình trạng "hy sinh X để ăn điểm Y/Yaw".
        rew_alpha = torch.sum(self.alpha_scale, dim=-1) * (s.safety_vel_alpha / 3.0)

        # R3: Asymmetric Smoothness penalty (End-to-End LPF)
        if hasattr(self, "_prev_alpha"):
            d_alpha = self.alpha_scale - self._prev_alpha
            # Phạt nặng (x5) nếu d_alpha > 0 (Tăng tốc quá nhanh)
            # Không phạt (x0.1) nếu d_alpha < 0 (Cho phép phanh gấp để sinh tồn)
            penalty_smooth = torch.where(
                d_alpha > 0,
                d_alpha**2 * 1.5,
                d_alpha**2 * 0.1,  # baseline: 5.0 accel / 0.1 brake
            )
            rew_smooth = -penalty_smooth.sum(dim=-1) * s.safety_smooth
        else:
            rew_smooth = torch.zeros(self.num_envs, device=self.device)
        self._prev_alpha = self.alpha_scale.clone()

        # R4: Orientation penalty
        yaw_cmd_mag = self.training_dir[:, 2].abs()
        orientation_penalty_scale = (1.0 - 0.5 * yaw_cmd_mag).clamp(min=0.5)

        is_fallen = sim.projected_gravity[:, 2] > 0.0
        self.reset_buf |= is_fallen
        self.base_env.reset_buf |= is_fallen
        rew_orientation = torch.where(
            is_fallen,
            -s.safety_orientation * orientation_penalty_scale,
            torch.zeros_like(self.rew_buf),
        )

        # Aggregate (R1+R2+R3+R4)
        rew_slip = rew_slip.clamp(
            -50.0, 0.0
        )  # Tháo bỏ giới hạn phạt (trước đây bị kịch trần -5.0)
        rew_alpha = rew_alpha.clamp(0.0, self.REWARD_CLIP)
        rew_smooth = rew_smooth.clamp(-2.0, 0.0)
        self.rew_buf = rew_slip + rew_alpha + rew_smooth + rew_orientation

        # Logging
        self.episode_sums["rew_foot_slip"] += rew_slip
        self.episode_sums["rew_vel_alpha"] += rew_alpha
        self.episode_sums["rew_smooth"] += rew_smooth
        self.episode_sums["rew_orientation"] += rew_orientation
        self.episode_sums["mean_alpha"] += alpha_mean
        self.episode_sums["mean_slip"] += total_slip
        self.episode_sums["mean_vel_x"] += sim.base_lin_vel[:, 0]
        self.episode_sums["mean_vel_yaw"] += sim.base_ang_vel[:, 2]
        self.episode_sums["mean_z_t"] += z_t_val

    def _handle_resets(self):
        """Handle resets and logging."""
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) == 0:
            return

        self.base_env.reset_idx(env_ids)
        self.extras["episode"] = {}
        ep_len = self.episode_length_buf[env_ids].float().clamp(min=1)
        for key, buf in self.episode_sums.items():
            self.extras["episode"][f"{key}_per_step"] = (
                (buf[env_ids] / ep_len).mean().item()
            )
            buf[env_ids] = 0.0

        self.episode_length_buf[env_ids] = 0
        self.env_steps[env_ids] = 0
        self.alpha_scale[env_ids] = 1.0
        self.last_actions[env_ids] = 1.0
        self._z_t[env_ids] = 0.0
        self._prev_alpha[env_ids] = 1.0
        self._resample_direction(env_ids)
