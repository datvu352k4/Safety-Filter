from legged_gym import *
import numpy as np
import torch
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.math_utils import torch_rand_float
from legged_gym.utils.math_utils import *
from collections import deque

class G1Mimic(LeggedRobot):
    
    def compute_observations(self):
        key_body_pos_relative_to_base = self.simulator.key_body_pos - \
                self.simulator.base_pos.unsqueeze(1)

        # get key body point position relative to base in the body frame
        key_body_pos_b = quat_rotate_inverse(
            self.simulator.base_quat.unsqueeze(1).repeat(1, self.simulator.key_body_pos.shape[1], 1),
            key_body_pos_relative_to_base 
        )
        ref_motion_obs = self._get_ref_motion_obs()
        
        obs_buf = torch.cat((
            # proprioceptive features
            self.simulator.base_ang_vel * self.obs_scales.ang_vel,
            self.simulator.base_quat,
            (self.simulator.dof_pos - 
             self.simulator.default_dof_pos) * self.obs_scales.dof_pos,
            self.simulator.dof_vel * self.obs_scales.dof_vel,
            key_body_pos_b.flatten(start_dim=1),
            self.actions,
            ref_motion_obs,
        ), dim=-1)
        
        # domain randomization params
        domain_params = torch.cat((
            self.simulator.dr_friction_values - self.friction_value_offset,
            self.simulator.dr_added_base_mass,
            self.simulator.dr_base_com_bias,
            self.simulator.dr_rand_push_vels,
        ), dim=-1)
        
        single_critic_obs = torch.cat((
            (self.simulator.base_pos - self.simulator.env_origins),
            self.simulator.base_lin_vel * self.obs_scales.lin_vel,
            self.ref_root_pos[self.frame_idx],
            obs_buf,
            domain_params,
        ), dim=-1)
        
        self.critic_obs_deque.append(single_critic_obs)
        self.privileged_obs_buf = torch.cat(
            [self.critic_obs_deque[i]
                for i in range(self.critic_obs_deque.maxlen)],
            dim=-1,
        )
        
        if self.add_noise:
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec

        # push obs_buf to obs_history
        self.obs_history_deque.append(obs_buf)
        self.obs_buf = torch.cat(
            [self.obs_history_deque[i]
                for i in range(self.obs_history_deque.maxlen)],
            dim=-1,
        )
        
    def post_physics_step(self):
        self.frame_idx += 1
        # find out which envs have exceeded the motion length, and mark them as time out
        # resample frame_idx for the time out envs and reset root states and dofs of them seperately
        time_out_buf = self.frame_idx >= self.motion_length
        self.ref_time_out_env_ids = time_out_buf.nonzero(as_tuple=False).flatten()
        if len(self.ref_time_out_env_ids) > 0:
            self.frame_idx[:] = self.frame_idx % self.motion_length
            # BUG: IsaacGym requires 1 step after resetting to get the correct rigid body states
            # The rigid body state does not update after this reset, which causes the termination abnormally
            # The dof state and root state is reset correctly, but the rigid body state is not updated
            self._reset_root_states(self.ref_time_out_env_ids)
            self._reset_dofs(self.ref_time_out_env_ids)
        super().post_physics_step()
        if self.debug:
            ref_key_body_pos = self.ref_key_body_pos_relative_to_base[self.frame_idx] \
                + self.ref_root_pos[self.frame_idx].unsqueeze(1) \
                + self.simulator.env_origins.unsqueeze(1)
            self.simulator.draw_debug_vis(ref_key_body_pos)
            
    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        # randomly sample a starting frame index for each environment
        self.frame_idx[env_ids] = torch.randint(0, self.motion_length, (len(env_ids),), device=self.device)
        super().reset_idx(env_ids)
        # clear obs history for the envs that are reset
        for i in range(self.obs_history_deque.maxlen):
            self.obs_history_deque[i][env_ids] *= 0
        for i in range(self.critic_obs_deque.maxlen):
            self.critic_obs_deque[i][env_ids] *= 0
    
    def _reset_dofs(self, env_ids):
        # reset dofs to match the reference motion at the current frame index
        cur_frame_idx = self.frame_idx[env_ids]
        dof_pos = self.ref_dof_pos[cur_frame_idx]
        dof_vel = self.ref_dof_vel[cur_frame_idx]
        self.simulator.reset_dofs(env_ids, 
                                  dof_pos, 
                                  dof_vel)

    def _reset_root_states(self, env_ids):
        # reset root states to match the reference motion at the current frame index
        cur_frame_idx = self.frame_idx[env_ids]
        root_pos = self.ref_root_pos[cur_frame_idx] + self.simulator.env_origins[env_ids]
        root_pos[:, 2] += 0.05 # add a small vertical offset to avoid initial penetration
        root_rot = self.ref_root_rot[cur_frame_idx]
        root_lin_vel = self.ref_root_lin_vel[cur_frame_idx]
        root_ang_vel = self.ref_root_ang_vel[cur_frame_idx]
        self.simulator.reset_root_states(env_ids, 
                                         root_pos, 
                                         root_rot, 
                                         root_lin_vel, 
                                         root_ang_vel)
    
    def _get_noise_scale_vec(self):
        noise_vec = torch.zeros(self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        # noise_vec[:3] = 0.
        # noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel # ang_vel
        # noise_vec[6:9] = noise_scales.gravity * noise_level # projected gravity
        # noise_vec[9:9 + self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        # noise_vec[9 + self.num_actions:9 + 2 * self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        # noise_vec[9 + 2 * self.num_actions:24 + 2*self.num_actions] = 0.  # key body pos relative to base and actions
        # noise_vec[24 + 2*self.num_actions:24 + 3 * self.num_actions] = 0.  # previous actions
        # noise_vec[24 + 3 * self.num_actions:25 + 3 * self.num_actions] = 0. # frame_portion
        # noise_vec[25 + 3 * self.num_actions:28 + 3 * self.num_actions] = 0.  # ref_root_lin_vel
        # noise_vec[28 + 3 * self.num_actions:31 + 3 * self.num_actions] = 0.  # ref_root_ang_vel
        # noise_vec[31 + 3 * self.num_actions:34 + 3 * self.num_actions] = 0.  # ref_projected_gravity
        # noise_vec[34 + 3 * self.num_actions:34 + 4 * self.num_actions] = 0.  # ref_dof_pos
        # noise_vec[34 + 4 * self.num_actions:34 + 5 * self.num_actions] = 0.  # ref_dof_vel
        # noise_vec[34 + 5 * self.num_actions:49 + 5 * self.num_actions] = 0.  # ref_key_body_pos_relative_to_base
        return noise_vec
    
    def _init_buffers(self):
        super()._init_buffers()
        # load reference motion data from the motion file specified in the config
        import pickle
        # Compatibility shim: pickle files saved with NumPy >= 2.0 reference
        # numpy._core (e.g. numpy._core.multiarray), which does not exist in
        # NumPy < 2.0.  Register aliases so unpickling works transparently.
        if not hasattr(np, '_core'):
            import numpy.core as _np_core
            for _attr in dir(_np_core):
                _mod_name = f"numpy._core.{_attr}"
                _submod = getattr(_np_core, _attr, None)
                if isinstance(_submod, type(sys)):
                    sys.modules.setdefault(_mod_name, _submod)
            sys.modules.setdefault("numpy._core", _np_core)
            sys.modules.setdefault("numpy._core.multiarray", _np_core.multiarray)
            del _np_core, _attr, _mod_name, _submod
        
        # open the .pkl file and load the motion data
        # the motion data should be a dictionary with the following keys:
        # motion_data = {
        #     "fps": aligned_fps,
        #     "root_pos": root_pos.cpu().numpy(), # root link position in world frame
        #     "root_lin_vel": root_lin_vel.cpu().numpy(), # root link linear velocity in world frame
        #     "root_rot": root_rot.cpu().numpy(), # root link orientation as quaternion in world frame
        #     "root_euler": root_euler.cpu().numpy(), # root link orientation as euler angles in world frame
        #     "root_ang_vel": root_ang_vel.cpu().numpy(), # root link angular velocity in world frame
        #     "dof_pos": dof_pos.cpu().numpy(), # joint angles matching dof_names order
        #     "dof_vel": dof_vel.cpu().numpy(), # joint velocities matching dof_names order
        #     "key_body_pos_relative_to_base": key_body_pos_relative_to_base.cpu().numpy(),
        #     # key body point positions relative to the base in the world frame, shape [motion_length, num_key_bodies, 3]
        # }
        motion_file_dir = LEGGED_GYM_ROOT_DIR + "/resources/reference_motion/"
        motion_file_path = motion_file_dir + self.cfg.env.motion_file
        with open(motion_file_path, "rb") as f:
            motion_data = pickle.load(f)
        root_pos = motion_data["root_pos"]
        self.ref_root_pos = torch.from_numpy(root_pos).to(self.device).float()
        # judge if root_lin_vel is provided in the motion data, for visualing the reference motion from GMR
        if "root_lin_vel" in motion_data:
            root_lin_vel = motion_data["root_lin_vel"]
            self.ref_root_lin_vel = torch.from_numpy(root_lin_vel).to(self.device).float()
        else:
            self.ref_root_lin_vel = torch.zeros_like(self.ref_root_pos)
        root_rot = motion_data["root_rot"]
        self.ref_root_rot = torch.from_numpy(root_rot).to(self.device).float()
        if "root_ang_vel" in motion_data:
            root_ang_vel = motion_data["root_ang_vel"]
            self.ref_root_ang_vel = torch.from_numpy(root_ang_vel).to(self.device).float()
        else:
            self.ref_root_ang_vel = torch.zeros_like(self.ref_root_pos)
        dof_pos = motion_data["dof_pos"]
        self.ref_dof_pos = torch.from_numpy(dof_pos).to(self.device).float()
        if "dof_vel" in motion_data:
            dof_vel = motion_data["dof_vel"]
            self.ref_dof_vel = torch.from_numpy(dof_vel).to(self.device).float()
        else:
            self.ref_dof_vel = torch.zeros_like(self.ref_dof_pos)
        if "key_body_pos_relative_to_base" in motion_data:
            key_body_pos_relative_to_base = motion_data["key_body_pos_relative_to_base"]
            self.ref_key_body_pos_relative_to_base = torch.from_numpy(key_body_pos_relative_to_base).to(self.device).float()
        else:
            self.ref_key_body_pos_relative_to_base = torch.zeros(
                self.ref_root_pos.shape[0],          # motion_length frames
                self.simulator.key_body_pos.shape[1], # num_key_bodies
                3,
                dtype=torch.float, device=self.device
            )
        self.motion_length = self.ref_root_pos.shape[0]
        assert self.ref_root_pos.shape[0] == self.ref_root_rot.shape[0] == self.ref_root_lin_vel.shape[0] == self.ref_root_ang_vel.shape[0] == self.ref_dof_pos.shape[0] == self.ref_dof_vel.shape[0], "Reference motion data length mismatch among different features"
        print(f"Loaded reference motion from {self.cfg.env.motion_file}, motion length: {self.motion_length} frames")
        # create a buffer to store the current frame index for each environment
        self.frame_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # flag set each step: True for envs whose frame_idx would exceed motion_length
        self._timed_out = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # obs_history
        self.obs_history_deque = deque(maxlen=self.cfg.env.frame_stack)
        for _ in range(self.cfg.env.frame_stack):
            self.obs_history_deque.append(
                torch.zeros(
                    self.num_envs,
                    self.cfg.env.num_single_obs,
                    dtype=torch.float,
                    device=self.device,
                )
            )
        # critic observation buffer
        self.critic_obs_deque = deque(maxlen=self.cfg.env.c_frame_stack)
        for _ in range(self.cfg.env.c_frame_stack):
            self.critic_obs_deque.append(
                torch.zeros(
                    self.num_envs,
                    self.cfg.env.num_single_critic_obs,
                    dtype=torch.float,
                    device=self.device,
                )
            )
    
    def _get_ref_motion_obs(self):
        """Get the reference motion features for the current frame index.
        """
        ref_motion_obs = []
        for i in range(self.cfg.env.ref_motion_frame_stack):
            frame_idx = (self.frame_idx + i + 1) % self.motion_length
            ref_motion_obs.append(self._get_single_frame_ref_motion_obs(frame_idx))
        ref_motion_obs = torch.cat(ref_motion_obs, dim=-1)
        return ref_motion_obs
    
    def _get_single_frame_ref_motion_obs(self, frame_idx):
        """Get the reference motion features for the given frame index.
        """
        key_body_pos_relative_to_base = self.ref_key_body_pos_relative_to_base[frame_idx]
        key_body_pos_b = quat_rotate_inverse(
            # repeat the quaternion for each key body point, [N,4]->[N,num_key_bodies,4]
            self.ref_root_rot[frame_idx].unsqueeze(1).repeat(1, self.ref_key_body_pos_relative_to_base.shape[1], 1),
            key_body_pos_relative_to_base # [N,num_key_bodies,3] 
        )
        ref_root_lin_vel_b = quat_rotate_inverse(
            self.ref_root_rot[frame_idx],
            self.ref_root_lin_vel[frame_idx]
        )
        ref_root_ang_vel_b = quat_rotate_inverse(
            self.ref_root_rot[frame_idx],
            self.ref_root_ang_vel[frame_idx]
        )
        ref_motion_obs = torch.cat((
            ref_root_lin_vel_b * self.obs_scales.lin_vel,
            ref_root_ang_vel_b * self.obs_scales.ang_vel,
            self.ref_root_rot[frame_idx],
            (self.ref_dof_pos[frame_idx] - 
             self.simulator.default_dof_pos) * self.obs_scales.dof_pos,
            self.ref_dof_vel[frame_idx] * self.obs_scales.dof_vel,
            key_body_pos_b.flatten(start_dim=1),
        ), dim=-1)
        return ref_motion_obs
        
    def _reward_tracking_ref_dof_pos(self):
        """Reward term for imitating the reference motion's dof positions.
        """
        dof_pos_error = torch.sum(torch.square(
            self.simulator.dof_pos - 
            self.ref_dof_pos[self.frame_idx]), dim=-1)
        
        return torch.exp(-dof_pos_error 
                         / self.cfg.rewards.tracking_dof_pos_sigma)
    
    def _reward_tracking_ref_dof_vel(self):
        """Reward term for imitating the reference motion's dof velocities.
        """
        dof_vel_error = torch.sum(torch.abs(
            self.simulator.dof_vel - 
            self.ref_dof_vel[self.frame_idx]), dim=-1)
        
        return torch.exp(-dof_vel_error 
                         / self.cfg.rewards.tracking_dof_vel_sigma)
        
    def _reward_tracking_ref_root_pose(self):
        """Reward term for imitating the reference motion's root position and orientaion.
        """
        root_pos_error = torch.sum(torch.square(
            self.simulator.base_pos - 
            (self.ref_root_pos[self.frame_idx] + 
             self.simulator.env_origins)), dim=-1)
        
        root_rot_error = torch.sum(torch.square(
            self.simulator.base_quat - 
            self.ref_root_rot[self.frame_idx]), dim=-1)
        
        return torch.exp(-(root_pos_error + 0.1 * root_rot_error) /
                         self.cfg.rewards.tracking_ref_root_pose_sigma)
    
    def _reward_tracking_ref_root_vel(self):
        """Reward term for imitating the reference motion's root linear velocity and root angular velocity.
        """
        ref_root_lin_vel_b = quat_rotate_inverse(
            self.ref_root_rot[self.frame_idx],
            self.ref_root_lin_vel[self.frame_idx]
        )
        root_lin_vel_error = torch.sum(torch.square(
            self.simulator.base_lin_vel - 
            ref_root_lin_vel_b), dim=-1)
        
        ref_root_ang_vel_b = quat_rotate_inverse(
            self.ref_root_rot[self.frame_idx],
            self.ref_root_ang_vel[self.frame_idx]
        )
        root_ang_vel_error = torch.sum(torch.square(
            self.simulator.base_ang_vel - 
            ref_root_ang_vel_b), dim=-1)
        
        return torch.exp(-(root_lin_vel_error + 0.1 * root_ang_vel_error) /
                         self.cfg.rewards.tracking_ref_root_vel_sigma)
    
    def _reward_tracking_ref_key_pos(self):
        """Reward term for imitating the reference motion's key body position relative to base.
        """
        key_body_pos_relative_to_base = self.simulator.key_body_pos - \
                self.simulator.base_pos.unsqueeze(1)
        key_body_pos_error = torch.sum(torch.square(
            key_body_pos_relative_to_base - 
            self.ref_key_body_pos_relative_to_base[self.frame_idx]), dim=[1,2])
        
        return torch.exp(-key_body_pos_error / 
                         self.cfg.rewards.tracking_ref_key_pos_sigma)