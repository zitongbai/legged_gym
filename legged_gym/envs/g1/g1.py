from legged_gym.envs import LeggedRobot
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import numpy as np
import torch
import os
from legged_gym import LEGGED_GYM_ROOT_DIR

@torch.jit.script
def copysign(a:float, b:torch.Tensor):
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)
def get_euler_xyz(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * \
                q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(
        torch.abs(sinp) >= 1, copysign(np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
                q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack((roll, pitch, yaw), dim=-1)

class G1(LeggedRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        
    def check_termination(self):
        super().check_termination()
        self.reset_buf |= torch.logical_or(torch.abs(self.rpy[:,1])>1.0, torch.abs(self.rpy[:,0])>0.8)
        
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.update_upper_dof_pos_limit_curriculum(env_ids)

    def _post_physics_step_callback(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.feet_vel = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]
        
        self.rpy[:] = get_euler_xyz(self.base_quat[:])

        period = 0.8    # TODO: move to config file
        offset = 0.5
        self.phase[:] = (self.episode_length_buf[:]*self.dt) % period / period 
        # self.phase_left[:] = self.phase[:]
        # self.phase_right[:] = (self.phase[:] + offset) % 1.0
        self.leg_phase[:,0] = self.phase[:]
        self.leg_phase[:,1] = (self.phase[:] + offset) % 1.0
        
        super()._post_physics_step_callback()

    def compute_observations(self):
        sin_phase = torch.sin(2*np.pi*self.phase).unsqueeze(1)
        cos_phase = torch.cos(2*np.pi*self.phase).unsqueeze(1)
        
        # upper_coeff = torch.ones(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        # upper_coeff[self.upper_dof_indices] = 0.25
        
        self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_phase,
                                    cos_phase
                                    ),dim=-1)
        self.privileged_obs_buf = torch.cat((  
                                    self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_phase,
                                    cos_phase
                                    ),dim=-1)
        
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            # heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            # self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
            raise NotImplementedError("Height measurements not implemented yet")
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0. # commands
        noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0. # previous actions
        noise_vec[9+3*self.num_actions:9+3*self.num_actions+2] = 0. # sin/cos phase
        
        if self.cfg.terrain.measure_heights:
            # TODO: support for height measurements
            raise NotImplementedError("Noise for height measurements not implemented yet")
        
        return noise_vec

    def _init_buffers(self):
        super()._init_buffers()
        
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.feet_vel = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]

        self.rpy = get_euler_xyz(self.base_quat)
        
        self.phase = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.leg_phase = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.long)
        
    def _create_envs(self):
        super()._create_envs()
        
        # find all upper dofs, obeying order in self.dof_names
        self.upper_dof_indices = []
        for i in range(len(self.dof_names)):
            if any([s in self.dof_names[i] for s in self.cfg.asset.upper_dof_name]):
                self.upper_dof_indices.append(i)
        self.upper_dof_indices = torch.tensor(self.upper_dof_indices, dtype=torch.long, device=self.device, requires_grad=False)
        
        # find all hip dofs, obeying order in self.dof_names
        self.hip_dof_indices = []
        for i in range(len(self.dof_names)):
            if any([s in self.dof_names[i] for s in self.cfg.asset.hip_dof_name]):
                self.hip_dof_indices.append(i)
        self.hip_dof_indices = torch.tensor(self.hip_dof_indices, dtype=torch.long, device=self.device, requires_grad=False)
        
        # self.leg_dof_indices = []
        # for i in range(len(self.dof_names)):
        #     if any([s in self.dof_names[i] for s in self.cfg.asset.leg_dof_name]):
        #         self.leg_dof_indices.append(i)
        # self.leg_dof_indices = torch.tensor(self.leg_dof_indices, dtype=torch.long, device=self.device, requires_grad=False)
        
        # dof pos limits of upper dof is 0 at beginning.
        
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity
        
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        self.dof_limits_upper = self.dof_props_asset["upper"][:]
        self.dof_limits_lower = self.dof_props_asset["lower"][:]
        
        self._change_upper_dof_pos_limits(0.0)
        

    def update_upper_dof_pos_limit_curriculum(self, env_ids):
        
        track_rew_percent =  torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length 
        # print(track_rew_percent)
        if track_rew_percent > 0.5 * self.reward_scales["tracking_lin_vel"]:
            self._change_upper_dof_pos_limits(0.1)
            # self.reward_scales["upper_dof"] = 0.1
        elif track_rew_percent > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self._change_upper_dof_pos_limits(0.5)
        elif track_rew_percent > 0.9 * self.reward_scales["tracking_lin_vel"]:
            self._change_upper_dof_pos_limits(1.0)
        

    def _change_upper_dof_pos_limits(self, scale):
        for i, env_handle in enumerate(self.envs):
            actor_handle = self.actor_handles[i]
            for idx in self.upper_dof_indices:
                self.dof_props_asset["lower"][idx] = (1-scale) * self.default_dof_pos[idx] + scale * self.dof_limits_lower[idx]
                self.dof_props_asset["upper"][idx] = (1-scale) * self.default_dof_pos[idx] + scale * self.dof_limits_upper[idx]
            self.gym.set_actor_dof_properties(env_handle, actor_handle, self.dof_props_asset)

        for idx in self.upper_dof_indices:
            self.dof_pos_limits[idx, 0] = (1-scale) * self.default_dof_pos[idx] + scale * self.dof_limits_lower[idx]
            self.dof_pos_limits[idx, 1] = (1-scale) * self.default_dof_pos[idx] + scale * self.dof_limits_upper[idx]

    # ------------ reward functions----------------
    # def _reward_torques(self):
    #     # Penalize torques
    #     return torch.sum(torch.square(self.torques[:, self.leg_dof_indices]), dim=1)
    
    # def _reward_dof_vel(self):
    #     # Penalize dof velocities
    #     return torch.sum(torch.square(self.dof_vel[:, self.leg_dof_indices]), dim=1)
    
    # def _reward_dof_acc(self):
    #     # Penalize dof accelerations
    #     return torch.sum(torch.square((self.last_dof_vel[:, self.leg_dof_indices] - self.dof_vel[:, self.leg_dof_indices]) / self.dt), dim=1)
    
    # def _reward_action_rate(self):
    #     # Penalize changes in actions
    #     return torch.sum(torch.square(self.last_actions[:, self.leg_dof_indices] - self.actions[:, self.leg_dof_indices]), dim=1)
    
    def _reward_contact(self):
        """ Reward for contact when in stance phase
        """
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(2):
            is_stance = self.leg_phase[:,i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            res += ~(contact ^ is_stance)
        return res

    def _reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        pos_error = torch.square(self.feet_pos[:, :, 2] - self.cfg.rewards.feet_swing_height) * ~contact
        return torch.sum(pos_error, dim=(1))

    def _reward_alive(self):
        # Reward for staying alive
        return 1.0
    
    def _reward_contact_no_vel(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1,2))
    
    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:, self.hip_dof_indices]), dim=1)
    
    # def _reward_upper_dof(self):
    #     # upper dof
    #     upper_dof_err = torch.sum(torch.abs(self.dof_pos[:, self.upper_dof_indices] - self.default_dof_pos[:, self.upper_dof_indices]), dim=1)
    #     return torch.exp(-upper_dof_err/self.cfg.rewards.tracking_sigma)
    
    def _reward_foot_clearance(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
            footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])
        
        height_error = torch.square(footpos_in_body_frame[:, :, 2] - self.cfg.rewards.clearance_height_target).view(self.num_envs, -1)
        foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)
        return torch.sum(height_error * foot_leteral_vel, dim=1)
    