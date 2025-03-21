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
        # self.update_upper_dof_pos_limit_curriculum(env_ids)
        # self.update_upper_dof_action_clip_curriculum(env_ids)

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
        
        self.obs_buf = torch.cat((  
                                    self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    # sin_phase,
                                    # cos_phase
                                    ),dim=-1)
        self.privileged_obs_buf = torch.cat((  
                                    self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    # sin_phase,
                                    # cos_phase
                                    ),dim=-1)
        
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            # heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            # self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
            raise NotImplementedError("Height measurements not implemented yet")
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    # def _get_noise_scale_vec(self, cfg):
    #     """ Sets a vector used to scale the noise added to the observations.
    #         [NOTE]: Must be adapted when changing the observations structure

    #     Args:
    #         cfg (Dict): Environment config file

    #     Returns:
    #         [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
    #     """
    #     noise_vec = torch.zeros_like(self.obs_buf[0])
    #     self.add_noise = self.cfg.noise.add_noise
    #     noise_scales = self.cfg.noise.noise_scales
    #     noise_level = self.cfg.noise.noise_level
    #     noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
    #     noise_vec[3:6] = noise_scales.gravity * noise_level
    #     noise_vec[6:9] = 0. # commands
    #     noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
    #     noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
    #     noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0. # previous actions
    #     noise_vec[9+3*self.num_actions:9+3*self.num_actions+2] = 0. # sin/cos phase
        
    #     if self.cfg.terrain.measure_heights:
    #         # TODO: support for height measurements
    #         raise NotImplementedError("Noise for height measurements not implemented yet")
        
    #     return noise_vec
    
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
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12:12+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[12+self.num_actions:12+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[12+2*self.num_actions:12+3*self.num_actions] = 0. # previous actions
        noise_vec[12+3*self.num_actions:12+3*self.num_actions+2] = 0. # sin/cos phase
        
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
        
        # self._change_upper_dof_pos_limits(0.1)
        
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
        
        self.arm_dof_indices = []
        for i in range(len(self.dof_names)):
            if any([s in self.dof_names[i] for s in self.cfg.asset.arm_dof_name]):
                self.arm_dof_indices.append(i)
        self.arm_dof_indices = torch.tensor(self.arm_dof_indices, dtype=torch.long, device=self.device, requires_grad=False)
        
        self.waist_dof_indices = []
        for i in range(len(self.dof_names)):
            if any([s in self.dof_names[i] for s in self.cfg.asset.waist_dof_name]):
                self.waist_dof_indices.append(i)
        self.waist_dof_indices = torch.tensor(self.waist_dof_indices, dtype=torch.long, device=self.device, requires_grad=False)
        
        self.leg_dof_indices = []
        for i in range(len(self.dof_names)):
            if any([s in self.dof_names[i] for s in self.cfg.asset.leg_dof_name]):
                self.leg_dof_indices.append(i)
        self.leg_dof_indices = torch.tensor(self.leg_dof_indices, dtype=torch.long, device=self.device, requires_grad=False)

        self.ankle_dof_indices = []
        for i in range(len(self.dof_names)):
            if any([s in self.dof_names[i] for s in self.cfg.asset.ankle_dof_name]):
                self.ankle_dof_indices.append(i)
        self.ankle_dof_indices = torch.tensor(self.ankle_dof_indices, dtype=torch.long, device=self.device, requires_grad=False)

        self.hip_knee_dof_indices = []
        for i in range(len(self.dof_names)):
            if any([s in self.dof_names[i] for s in self.cfg.asset.hip_knee_dof_name]):
                self.hip_knee_dof_indices.append(i)
        self.hip_knee_dof_indices = torch.tensor(self.hip_knee_dof_indices, dtype=torch.long, device=self.device, requires_grad=False)
        
    # def update_upper_dof_action_clip_curriculum(self, env_ids):
    #     track_rew_percent =  torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length
    #     if track_rew_percent > 0.4 * self.reward_scales["tracking_lin_vel"]:
    #         self.cfg.normalization.clip_upper_dof_actions_scale = 0.1
    #     elif track_rew_percent > 0.6 * self.reward_scales["tracking_lin_vel"]:
    #         self.cfg.normalization.clip_upper_dof_actions_scale = 0.5
    #     elif track_rew_percent > 0.8 * self.reward_scales["tracking_lin_vel"]:
    #         self.cfg.normalization.clip_upper_dof_actions_scale = 1.0
    #     else:
    #         self.cfg.normalization.clip_upper_dof_actions_scale = 0.0
        
        
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.5 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)
            self.command_ranges["lin_vel_y"][0] = np.clip(self.command_ranges["lin_vel_y"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_y"][1] = np.clip(self.command_ranges["lin_vel_y"][1] + 0.5, 0., self.cfg.commands.max_curriculum)
            self.command_ranges["ang_vel_yaw"][0] = np.clip(self.command_ranges["ang_vel_yaw"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["ang_vel_yaw"][1] = np.clip(self.command_ranges["ang_vel_yaw"][1] + 0.5, 0., self.cfg.commands.max_curriculum)

        
    #------------ reward functions----------------
    
    # ----------------------------------------------------------------
    # Body
    # ----------------------------------------------------------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    # ----------------------------------------------------------------
    # Joint (Dof)
    # ----------------------------------------------------------------
    
    def _reward_dof_torques(self):
        # Penalize torques
        weighted_torque = self.torques / self.p_gains
        return torch.sum(torch.square(weighted_torque), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_dof_power(self):
        # Penalize dof power
        numerator = torch.sum(torch.abs(self.torques * self.dof_vel), dim=1)
        denominator = torch.sum(torch.square(self.base_lin_vel), dim=1) + 0.2 * torch.sum(torch.square(self.base_ang_vel), dim=1)
        return numerator / denominator

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)


    def _reward_arm_dof_deviation(self):
        arm_dof_err = torch.sum(torch.abs(self.dof_pos[:, self.arm_dof_indices] - self.default_dof_pos[:, self.arm_dof_indices]), dim=1)
        return arm_dof_err
    
    def _reward_waist_dof_deviation(self):
        waist_dof_err = torch.sum(torch.abs(self.dof_pos[:, self.waist_dof_indices] - self.default_dof_pos[:, self.waist_dof_indices]), dim=1)
        return waist_dof_err
    
    def _reward_hip_dof_deviation(self):
        hip_dof_err = torch.sum(torch.abs(self.dof_pos[:, self.hip_dof_indices] - self.default_dof_pos[:, self.hip_dof_indices]), dim=1)
        return hip_dof_err
    
    def _reward_ankle_action(self):
        return torch.sum(torch.square(self.actions[:, self.ankle_dof_indices]), dim=1)


    def _reward_ankle_dof_pos_limits(self):
        out_of_limits = -(self.dof_pos[:, self.ankle_dof_indices] - self.dof_pos_limits[self.ankle_dof_indices, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos[:, self.ankle_dof_indices] - self.dof_pos_limits[self.ankle_dof_indices, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_hip_knee_dof_acc(self):
        return torch.sum(torch.square((self.last_dof_vel[:, self.hip_knee_dof_indices] - self.dof_vel[:, self.hip_knee_dof_indices]) / self.dt), dim=1)

    def _reward_hip_knee_dof_torques(self):
        return torch.sum(torch.square(self.torques[:, self.hip_knee_dof_indices]), dim=1)

    # ----------------------------------------------------------------
    # Feet
    # ----------------------------------------------------------------
    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.0) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime = torch.clamp(rew_airTime, min=0., max=0.4)
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_fly(self):
        # penalize double fly feet
        no_contacts = self.contact_forces[:, self.feet_indices, 2] < 5.0
        return 1.0 * (torch.sum(no_contacts, dim=1) > 1)

    def _reward_feet_slip(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.0
        # contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        # penalize = torch.square(contact_feet_vel[:, :, :2])
        # return torch.sum(penalize, dim=(1,2))
        contact_vel_xy = torch.norm(self.feet_vel[:, :, :2], dim=2) * contact
        return torch.sum(contact_vel_xy, dim=1)
        
    def _reward_feet_distance(self):
        # penalize feet distance getting too close or too far away
        feet_dist = torch.norm(self.feet_pos[:, 0] - self.feet_pos[:, 1], dim=1)
        d_min = torch.clamp(feet_dist - self.cfg.rewards.feet_dist_min, min=-0.5, max=0.0)
        d_max = torch.clamp(feet_dist - self.cfg.rewards.feet_dist_max, min=0.0, max=0.5)
        return (torch.exp(-torch.abs(d_min)*100) + torch.exp(-torch.abs(d_max)*100))/2.0

    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)

    def _reward_feet_contact_gait(self):
        """ Reward for contact when in stance phase
        """
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(2):
            is_stance = self.leg_phase[:,i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            res += ~(contact ^ is_stance)
        return res
    
    def _reward_feet_clearance(self):
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
    
    # def _reward_feet_swing_height(self):
    #     contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
    #     pos_error = torch.square(self.feet_pos[:, :, 2] - self.cfg.rewards.feet_swing_height) * ~contact
    #     return torch.sum(pos_error, dim=(1))

    def _reward_alive(self):
        # Reward for staying alive
        return 1.0

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
            
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
        # diff = self.dof_pos - self.default_dof_pos
        # r =  torch.exp(-2*torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(min=0, max=0.5)
        # return r * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    
    def _reward_feet_contact(self):
        # stance phase: cmd < 0.05
        # moving: cmd > 0.05
        contact = self.contact_forces[:, self.feet_indices, 2] > 1. # indicate contact or not
        contact_num = torch.sum(contact, dim=1)
        
        stance_rew = (torch.norm(self.commands[:, :2], dim=1) < 0.1) * (contact_num > 0.5)
        walk_rew = (torch.norm(self.commands[:, :2], dim=1) >= 0.1) * torch.logical_and(contact_num > 0.5, contact_num < 1.5)
        return (stance_rew + walk_rew)*1.0