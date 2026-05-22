"""
Safety Filter v2 — Physics-Based Adaptive Speed Limiter.

Muc dich:
    Huan luyen 1 policy nho (MLP 64->32->3) de dau ra alpha in [0,1]
    lam SPEED LIMIT cho tung truc (vx, vy, yaw).
    Alpha duoc dieu khien boi z_t - latent estimate tu
    history encoder cua locomotion policy.

Logic tai deployment:
    v_limit = alpha x V_max
    final_cmd = clip(MPPI_cmd, +/-v_limit)

Logic tai training:
    Luon giao lenh o V_max (hard case) theo huong ngau nhien.
    Policy hoc: z_t nay -> alpha bao nhieu la an toan?

Reward design (v2 - Physics-Based):
    R1 foot_slip  (scale=6.0) - phat truot chan truc tiep tu physics
    R2 vel_alpha  (scale=1.0) - chong collapse, thuong dung capacity
    R3 smooth     (scale=0.2) - giu alpha thay doi muot

Thay doi so voi v1 (Guide-Based):
    v1 dung sigmoid(-k*z_t+b) hardcode mapping z_t->alpha.
    v2 de policy TU KHAM PHA mapping tu physics feedback (truot chan).
    -> Scales tu nhien khi z_t mo rong sang multi-dim (terrain, payload...).

Thay doi so voi v2.0:
    - Curriculum: tang combined (vx+vy+yaw) len 50% vi day la kich ban
      MPPI thuc te gay truot tren ma sat thap.
    - foot_slip scale: 4.0 -> 6.0 de buc policy bao thu hon.
    - Grace period: 50 steps an xa sau khi doi huong.
    - Squared slip: deadzone cho micro-slip.
    - Alpha floor 0.3: ngan PPO zero-out penalty.
"""

import torch


class Go2SafetyEnvWrapper:
    """Wrapper bien base locomotion env thanh safety filter training env."""

    # -- Hang so --
    MAX_CMD_VEL = 2.0  # m/s  -- van toc toi da truc X
    MAX_CMD_LAT = 1.0  # m/s  -- van toc toi da truc Y
    MAX_CMD_YAW = 1.5  # rad/s -- toc do quay toi da

    REWARD_CLIP = 5.0  # clamp reward thanh phan

    # Contact force threshold: chan "chịu tải thực sự" khi lực nén > 20N
    # Loại bỏ hoàn toàn vận tốc ảo lúc mới nhấc/hạ chân (pha swing)
    CONTACT_THRESHOLD = 15

    # Resample huong training moi 150 steps (~3.0 giay o 50Hz)
    # Tra lai thoi gian on dinh de chay deu kiem diem thuong R2
    RESAMPLE_INTERVAL = 120

    # Grace period: 0 steps. Không châm chước nữa!
    # Vì đã có Threshold 15N và deadzone 0.15m/s lọc hết nhiễu vật lý rồi.
    # Trượt trên băng lúc phanh gấp là phạt NGAY LẬP TỨC.
    GRACE_STEPS = 2

    def __init__(self, base_env, ll_policy, ll_model, cfg):
        self.base_env = base_env
        self.ll_policy = ll_policy
        self.ll_model = ll_model
        self.cfg = cfg
        self.device = base_env.device

        self.num_envs = base_env.num_envs
        self.num_obs = cfg.env.num_observations  # 8
        self.num_privileged_obs = cfg.env.num_privileged_obs  # 56
        self.num_actions = cfg.env.num_actions  # 3
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

        # alpha in [0,1] x 3 truc: speed limit fraction
        self.alpha_scale = torch.zeros(self.num_envs, 3, device=self.device)
        self.last_actions = torch.zeros(self.num_envs, 3, device=self.device)

        # Huong training {-1, 0, +1} x 3 truc -- magnitude luon V_max
        self.training_dir = torch.ones(self.num_envs, 3, device=self.device)
        self.env_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        # Grace period counter: steps ke tu lan doi huong cuoi
        self.steps_since_resample = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        # z_t cache
        self._z_t = torch.zeros(self.num_envs, 1, device=self.device)

        # V_MAX tensor (1, 3)
        self._V_MAX = torch.tensor(
            [self.MAX_CMD_VEL, self.MAX_CMD_LAT, self.MAX_CMD_YAW],
            device=self.device,
        ).unsqueeze(0)

        # Feet indices tu simulator (4 chan: FL, FR, RL, RR)
        self._feet_indices = base_env.simulator._feet_indices

        # Logging
        self.episode_sums = {
            "rew_foot_slip": torch.zeros(self.num_envs, device=self.device),
            "rew_vel_alpha": torch.zeros(self.num_envs, device=self.device),
            "rew_smooth": torch.zeros(self.num_envs, device=self.device),
            "mean_alpha": torch.zeros(self.num_envs, device=self.device),
            "mean_slip": torch.zeros(self.num_envs, device=self.device),
            "mean_vel_x": torch.zeros(self.num_envs, device=self.device),
            "mean_z_t": torch.zeros(self.num_envs, device=self.device),
        }

    # =====================================================================
    # PUBLIC API
    # =====================================================================

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def reset(self):
        base_obs, _, base_obs_history, _ = self.base_env.get_observations()
        self._compute_safety_obs(base_obs, base_obs_history)
        return self.obs_buf, self.privileged_obs_buf

    def step(self, actions):
        # 1. Cap nhat alpha tu output policy
        self._apply_alpha(actions)

        # 2. Doi huong training ngau nhien
        self._resample_direction()

        # 3. Tao lenh = alpha x V_max x direction
        safe_cmd = self.alpha_scale * self._V_MAX * self.training_dir
        self.base_env.commands[:, :3] = safe_cmd

        # 4. Chay locomotion policy 1 step
        with torch.no_grad():
            base_obs, _, base_obs_history, _ = self.base_env.get_observations()
            ll_actions = self.ll_policy(base_obs, base_obs_history)
            _, _, _, _, _, base_dones, _ = self.base_env.step(ll_actions)
            base_obs_new, _, base_obs_history_new, _ = self.base_env.get_observations()

        # 5. Cap nhat observations va rewards
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

    # =====================================================================
    # ALPHA: ASYMMETRIC LOW-PASS FILTER
    # =====================================================================

    def _apply_alpha(self, actions):
        """
        Chuyen output policy (unbounded) -> alpha in [0,1] voi LPF bat doi xung.

        Tang alpha cham (rate=0.15): tranh tang ga dot ngot.
        Giam alpha nhanh (rate=0.50): phan ung kip khi phat hien truot.
        """
        prev = self.alpha_scale.clone()
        raw = (torch.tanh(actions) + 1.0) / 2.0
        delta = raw - prev
        rate = torch.where(
            delta > 0,
            torch.full_like(delta, 0.15),  # tang cham
            torch.full_like(delta, 0.50),  # giam nhanh
        )
        self.alpha_scale = (prev + rate * delta).clamp(0.0, 1.0)
        self.last_actions = self.alpha_scale.clone()

    # =====================================================================
    # DIRECTION RESAMPLING
    # =====================================================================

    def _resample_direction(self):
        """
        Moi RESAMPLE_INTERVAL steps, doi huong lenh ngau nhien.

        Curriculum Huan Luyen Phanh (Brake Training):
        Eps tap trung hoan toan vao viec dat toc do MAX tren mem truc,
        sau do dot ngot Lat Lenh (dao chieu) hoac Phanh (Zero).
        Khong xai combined de robot cam nhan ro ret nhat "cu truot dai" cua tung truc!

          30%  vx-only (+1 hoac -1)
          30%  vy-only (+1 hoac -1)
          30%  yaw-only (+1 hoac -1)
          10%  cmd = 0  (Buoc phanh hoan toan)
        """
        resample = (self.env_steps % self.RESAMPLE_INTERVAL == 0) & (self.env_steps > 0)
        if not resample.any():
            return

        idx = resample.nonzero(as_tuple=True)[0]
        n = idx.shape[0]
        roll = torch.rand(n, device=self.device)

        new_dir = torch.zeros(n, 3, device=self.device)

        # 30%: vx-only
        mask_x = roll < 0.30
        if mask_x.any():
            new_dir[mask_x, 0] = (
                torch.randint(0, 2, (mask_x.sum(),), device=self.device) * 2 - 1
            ).float()

        # 30%: vy-only
        mask_y = (roll >= 0.30) & (roll < 0.60)
        if mask_y.any():
            new_dir[mask_y, 1] = (
                torch.randint(0, 2, (mask_y.sum(),), device=self.device) * 2 - 1
            ).float()

        # 30%: yaw-only
        mask_w = (roll >= 0.60) & (roll < 0.90)
        if mask_w.any():
            new_dir[mask_w, 2] = (
                torch.randint(0, 2, (mask_w.sum(),), device=self.device) * 2 - 1
            ).float()

        # 10%: zero (phanh hãm) -- đã khởi tạo bằng zero, không cần gán thêm.

        self.training_dir[idx] = new_dir
        self.steps_since_resample[idx] = 0  # reset grace counter

    # =====================================================================
    # OBSERVATIONS
    # =====================================================================

    def _compute_safety_obs(self, base_obs, base_obs_history):
        """
        Actor obs (8D) -- dung khi deploy:
            z_t(1) + base_vel(3) + height(1) + alpha_prev(3)

        Critic obs (56D) -- chi dung khi train:
            base_obs(45) + base_vel(3) + height(1) + z_t(1) + alpha(3) + dir(3)
        """
        with torch.no_grad():
            try:
                z_t = self.ll_model.history_encoder(base_obs_history)
            except AttributeError:
                z_t = self.ll_model.actor.history_encoder(base_obs_history)

        self._z_t = z_t

        base_lin_vel = self.base_env.simulator.base_lin_vel[:, :3]
        base_z = self.base_env.simulator.base_pos[:, 2].unsqueeze(1)

        self.obs_buf = torch.cat([z_t, base_lin_vel, base_z, self.last_actions], dim=-1)
        self.privileged_obs_buf = torch.cat(
            [base_obs, base_lin_vel, base_z, z_t, self.last_actions, self.training_dir],
            dim=-1,
        )

    # =====================================================================
    # REWARDS
    # =====================================================================

    def _compute_rewards(self):
        """
        Physics-based rewards -- khong dung sigmoid guide.

        R1: Foot Slip Penalty (scale=6.0)
            - Do truc tiep van toc chan khi dang cham dat.
            - Dung slip^2 (binh phuong): deadzone cho micro-slip.
            - Grace mask: mien tru 50 steps sau khi doi huong.
            - Alpha floor 0.3: ngan PPO zero-out penalty.

        R2: Vel x Alpha (scale=1.0)
            - Thuong khi robot di dung huong voi alpha cao.
            - Ngan alpha collapse ve 0.

        R3: Smoothness (scale=0.2)
            - Phat thay doi alpha dot ngot giua 2 steps.
        """
        self.steps_since_resample += 1
        s = self.cfg.rewards.scales
        sim = self.base_env.simulator

        safe_cmd = self.alpha_scale * self._V_MAX * self.training_dir
        v_xy = sim.base_lin_vel[:, :2]
        yaw_vel = sim.base_ang_vel[:, 2]
        actual_vel = torch.cat([v_xy, yaw_vel.unsqueeze(1)], dim=-1)

        alpha_mean = self.alpha_scale.mean(dim=-1)
        z_t_val = self._z_t.squeeze(-1)

        # --- DÙNG GRACE MASK 10 STEPS (~0.2s) ---
        # Cho phép robot có một khoảng đệm vật lý cực ngắn để dừng lại khi phanh gấp.
        # Hết 0.2s mà vẫn còn trượt (cày trên băng) là trảm (penalty thẳng tay)!
        grace_mask = (self.steps_since_resample.float() / self.GRACE_STEPS).clamp(
            0.0, 1.0
        )

        # --- R1: FOOT SLIP PENALTY (physics-based, scale=6.0) -------------
        # Dung LINEAR SLIP kem deadzone = 0.05m/s thay vi binh phuong.
        # Ly do: binh phuong lam penalty vo tinh nho di 5-10 lan khi slip < 1.0
        # khien R1 qua yeu so voi R2.
        #
        # Alpha floor 0.3: ngan PPO "lach luat" ep alpha=0 de huy penalty
        feet_vel_xy = sim._feet_vel[:, :, :2]  # (N, 4, 2)
        feet_contact_force = sim._link_contact_forces[
            :, self._feet_indices, :
        ]  # (N, 4, 3)
        in_contact = (feet_contact_force.norm(dim=-1) > self.CONTACT_THRESHOLD).float()

        slip_per_foot = feet_vel_xy.norm(dim=-1) * in_contact  # (N, 4)
        slip_per_foot = (slip_per_foot - 0.15).clamp(
            min=0.0
        )  # deadzone 0.15 m/s (che phủ nhiễu hình học xoay khớp của tâm bàn chân)
        total_slip = slip_per_foot.sum(dim=-1)  # (N,)

        # BỎ nhân với alpha_weight! Phạt trực tiếp trên tổng slip!
        # DÙNG NHẸ grace_mask (0.2s) để tránh phạt oan cú nảy vật lý đầu tiên khi phanh.
        rew_slip = -(total_slip * grace_mask) * s.safety_foot_slip

        # --- R2: VEL x ALPHA (chong collapse, scale=1.0) ------------------
        # Product: vel_normalized x alpha.
        # Alpha cao + robot di nhanh dung huong -> reward lon.
        # Alpha = 0 -> reward = 0 -> policy tim alpha duong.
        vel_in_dir = actual_vel * self.training_dir
        vel_norm = (vel_in_dir / self._V_MAX).clamp(0.0, 1.0)
        # SỬ DỤNG training_dir THAY VÌ safe_cmd ĐỂ XÁC ĐỊNH cmd_active!
        # Nếu dùng safe_cmd, PPO ép alpha=0 -> cmd_active=0 -> loại bỏ Y/W
        # khỏi num_active để "ăn gian" 100% thưởng từ X.
        # SỬ DỤNG training_dir THAY VÌ safe_cmd ĐỂ XÁC ĐỊNH cmd_active!
        # Dùng trực tiếp self.alpha_scale thay vì vel_norm * alpha.
        # Lý do: Base policy không thể đạt MaxX + MaxY + MaxYaw cùng lúc về mặt vật lý.
        # Nếu bắt chẹt vel_norm, an toàn bộ lọc bị R2 thê thảm, kéo theo reward âm.
        cmd_active = self.training_dir.abs() > 0.5
        product = torch.where(
            cmd_active, self.alpha_scale, torch.zeros_like(self.alpha_scale)
        )
        any_active = cmd_active.any(dim=-1).float()
        num_active = cmd_active.sum(dim=-1).clamp(min=1.0)
        rew_vel = (product.sum(dim=-1) / num_active) * any_active * s.safety_vel_alpha

        # --- R3: SMOOTHNESS (scale=0.2) ------------------------------------
        # Phat |Delta_alpha|^2 giua 2 steps lien tiep.
        if hasattr(self, "_prev_alpha"):
            d_alpha = self.alpha_scale - self._prev_alpha
            rew_smooth = -(d_alpha**2).sum(dim=-1) * s.safety_smooth
        else:
            rew_smooth = torch.zeros(self.num_envs, device=self.device)
        self._prev_alpha = self.alpha_scale.clone()

        # --- AGGREGATE ---------------------------------------------------
        rew_slip = rew_slip.clamp(-self.REWARD_CLIP, 0.0)
        rew_vel = rew_vel.clamp(0.0, self.REWARD_CLIP)
        rew_smooth = rew_smooth.clamp(-1.0, 0.0)

        self.rew_buf = rew_slip + rew_vel + rew_smooth

        # Logging
        self.episode_sums["rew_foot_slip"] += rew_slip
        self.episode_sums["rew_vel_alpha"] += rew_vel
        self.episode_sums["rew_smooth"] += rew_smooth
        self.episode_sums["mean_alpha"] += alpha_mean
        self.episode_sums["mean_slip"] += total_slip
        self.episode_sums["mean_vel_x"] += sim.base_lin_vel[:, 0]
        self.episode_sums["mean_z_t"] += z_t_val

    # =====================================================================
    # RESET
    # =====================================================================

    def _handle_resets(self):
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) == 0:
            return

        self.extras["episode"] = {}
        ep_len = self.episode_length_buf[env_ids].float().clamp(min=1)
        for key, buf in self.episode_sums.items():
            self.extras["episode"][f"{key}_per_step"] = (
                (buf[env_ids] / ep_len).mean().item()
            )
            buf[env_ids] = 0.0

        self.episode_length_buf[env_ids] = 0
        self.env_steps[env_ids] = 0
        self.steps_since_resample[env_ids] = 0
        self.training_dir[env_ids] = 1.0
        self.alpha_scale[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self._z_t[env_ids] = 0.0
        if hasattr(self, "_prev_alpha"):
            self._prev_alpha[env_ids] = 0.0
