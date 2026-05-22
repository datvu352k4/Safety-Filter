from legged_gym import *
from legged_gym.envs.base.legged_robot_ts import *


class Go2TSTerrain(LeggedRobotTS):
    # EMA decay: alpha=0.05 → time constant ≈ 20 steps ≈ 2 gait cycles at 2 m/s
    HEIGHT_VAR_EMA_ALPHA = 0.05

    def compute_observations(self):
        # Base observations
        self.obs_buf = torch.cat(
            (
                self.commands[:, :3] * self.commands_scale,
                self.simulator.projected_gravity,
                self.simulator.base_ang_vel * self.obs_scales.ang_vel,
                (self.simulator.dof_pos - self.simulator.default_dof_pos)
                * self.obs_scales.dof_pos,
                self.simulator.dof_vel * self.obs_scales.dof_vel,
                self.actions,
            ),
            dim=-1,
        )

        domain_randomization_info = torch.cat(
            (
                self.simulator._friction_values,
                self.simulator._added_base_mass,
                self.simulator._base_com_bias,
                self.simulator._rand_push_vels[:, :2],
                self.simulator._kp_scale,
                self.simulator._kd_scale,
            ),
            dim=-1,
        )

        # Critic observations
        critic_obs = torch.cat(
            (
                self.obs_buf,
                domain_randomization_info,
                self.simulator.base_lin_vel * self.obs_scales.lin_vel,
            ),
            dim=-1,
        )
        if self.cfg.asset.obtain_link_contact_states:
            critic_obs = torch.cat(
                (
                    critic_obs,
                    self.simulator.link_contact_states,
                ),
                dim=-1,
            )
        if self.cfg.terrain.measure_heights:
            heights = (
                torch.clip(
                    self.simulator.base_pos[:, 2].unsqueeze(1)
                    - 0.5
                    - self.simulator.measured_heights,
                    -1,
                    1.0,
                )
                * self.obs_scales.height_measurements
            )
            critic_obs = torch.cat((critic_obs, heights), dim=-1)
        self.critic_obs_deque.append(critic_obs)
        self.critic_obs_buf = torch.cat(
            [self.critic_obs_deque[i] for i in range(self.critic_obs_deque.maxlen)],
            dim=-1,
        )

        # Add noise
        if self.add_noise:
            self.obs_buf += (
                2 * torch.rand_like(self.obs_buf) - 1
            ) * self.noise_scale_vec

        # Update observation history
        self.obs_history_deque.append(self.obs_buf)
        self.obs_history = torch.cat(
            [self.obs_history_deque[i] for i in range(self.obs_history_deque.maxlen)],
            dim=-1,
        )

        # Privileged observations for teacher encoder
        if self.num_privileged_obs is not None:
            height_rel = (
                self.simulator.feet_pos[:, :, 2].unsqueeze(-1)
                - self.simulator.height_around_feet
            ).clip(
                -1.0, 1.0
            )  # (N, 4, 9)

            # Instantaneous per-foot statistics (flat: ≈0 | gravel: >>0)
            height_var   = torch.var(height_rel, dim=-1)                                   # (N, 4)
            height_range = height_rel.max(dim=-1).values - height_rel.min(dim=-1).values  # (N, 4)
            height_mean  = height_rel.mean(dim=-1)                                         # (N, 4)

            # EMA-smoothed roughness per foot — accumulates evidence over ~20 steps
            # Robust at high speed (some feet in swing → instantaneous var≈0 even on gravel)
            self.height_var_ema = (
                (1.0 - self.HEIGHT_VAR_EMA_ALPHA) * self.height_var_ema
                + self.HEIGHT_VAR_EMA_ALPHA * height_var.detach()
            )  # (N, 4)

            self.privileged_obs_buf = torch.cat(
                (
                    self.simulator._friction_values,             # 1   GT friction
                    height_var,                                  # 4   instantaneous var per foot
                    height_range,                                # 4   instantaneous range per foot
                    height_mean,                                 # 4   height offset per foot
                    self.height_var_ema,                         # 4   EMA-smoothed var per foot
                    self.simulator.normal_vector_around_feet,    # 12  surface normal per foot
                ),
                dim=-1,
            )  # Total: 29 dim

    def _init_buffers(self):
        super()._init_buffers()
        # EMA buffer for smoothed per-foot height variance (terrain roughness)
        self.height_var_ema = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device
        )

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        # Reset EMA so newly spawned envs start fresh (no residual from previous terrain)
        self.height_var_ema[env_ids] = 0.0

    def _reset_dofs(self, env_ids):
        """Reset DOF positions and velocities."""
        dof_pos = torch.zeros(
            (len(env_ids), self.num_actions), dtype=torch.float, device=self.device
        )
        dof_vel = torch.zeros(
            (len(env_ids), self.num_actions), dtype=torch.float, device=self.device
        )

        dof_pos[:, [0, 3, 6, 9]] = self.simulator.default_dof_pos[
            :, [0, 3, 6, 9]
        ] + torch_rand_float(-0.2, 0.2, (len(env_ids), 4), self.device)
        dof_pos[:, [1, 4, 7, 10]] = self.simulator.default_dof_pos[
            :, [1, 4, 7, 10]
        ] + torch_rand_float(-0.4, 0.4, (len(env_ids), 4), self.device)
        dof_pos[:, [2, 5, 8, 11]] = self.simulator.default_dof_pos[
            :, [2, 5, 8, 11]
        ] + torch_rand_float(-0.4, 0.4, (len(env_ids), 4), self.device)

        self.simulator.reset_dofs(env_ids, dof_pos, dof_vel)

    def _get_noise_scale_vec(self):
        """Calculate observation noise scale vector."""
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = 0.0  # commands
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[9:21] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[21:33] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[33:45] = 0.0  # previous actions
        return noise_vec

    def _reward_feet_air_time(self):
        """Reward long steps / air time."""
        contact = (
            self.simulator.link_contact_forces[:, self.simulator.feet_indices, 2] > 1.0
        )
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.0) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.25) * first_contact, dim=1)
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_foot_clearance(self):
        """Reward foot clearance during swing."""
        foot_vel_xy_norm = torch.norm(self.simulator.feet_vel[:, :, :2], dim=-1)
        clearance_error = torch.sum(
            foot_vel_xy_norm
            * torch.square(
                self.simulator.feet_pos[:, :, 2]
                - torch.mean(self.simulator.height_around_feet, dim=-1)
                - self.cfg.rewards.foot_clearance_target
                - self.cfg.rewards.foot_height_offset
            ),
            dim=-1,
        )
        return torch.exp(
            -clearance_error / self.cfg.rewards.foot_clearance_tracking_sigma
        )

    def _reward_hip_pos(self):
        """Reward hip joint position close to default."""
        hip_joint_indices = [0, 3, 6, 9]
        dof_pos_error = torch.sum(
            torch.square(
                self.simulator.dof_pos[:, hip_joint_indices]
                - self.simulator.default_dof_pos[:, hip_joint_indices]
            ),
            dim=-1,
        )
        return dof_pos_error

    def _reward_thigh_pos(self):
        """Reward thigh joint position close to default."""
        thigh_joint_indices = [1, 4, 7, 10]
        dof_pos_error = torch.sum(
            torch.square(
                self.simulator.dof_pos[:, thigh_joint_indices]
                - self.simulator.default_dof_pos[:, thigh_joint_indices]
            ),
            dim=-1,
        )
        return dof_pos_error
