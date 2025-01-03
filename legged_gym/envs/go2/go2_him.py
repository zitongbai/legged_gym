from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch

from legged_gym.envs.base.him_robot import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR

from .go2_him_config import Go2HimRoughCfg

class Go2HimRough(LeggedRobot):
    cfg : Go2HimRoughCfg
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        if self.cfg.control.use_actuator_network:
            actuator_network_path = self.cfg.control.actuator_net_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            self.actuator_network = torch.jit.load(actuator_network_path).to(self.device)
        
    def _init_buffers(self):
        super()._init_buffers()
        self.actuator_net_input = torch.zeros(self.num_envs*self.num_actions, 6, device=self.device, requires_grad=False)
        self.joint_pos_err_last = torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False)
        self.joint_vel_err_last = torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False)
        self.joint_pos_err_last_last = torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False)
        self.joint_vel_err_last_last = torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False)
    
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.joint_pos_err_last[env_ids] = 0.
        self.joint_pos_err_last_last[env_ids] = 0.
        self.joint_vel_err_last[env_ids] = 0.
        self.joint_vel_err_last_last[env_ids] = 0.
    
    def _compute_torques(self, actions):
        if self.cfg.control.use_actuator_network:
            action_scaled = actions * self.cfg.control.action_scale
            joint_pos_des = self.default_dof_pos + action_scaled
            joint_pos_err = self.dof_pos - joint_pos_des    # shape: (num_envs, num_actions)
            joint_vel_err = self.dof_vel    # shape: (num_envs, num_actions)
            with torch.inference_mode():
                self.actuator_net_input = torch.cat((
                    joint_pos_err.unsqueeze(-1),
                    self.joint_pos_err_last.unsqueeze(-1),
                    self.joint_pos_err_last_last.unsqueeze(-1),
                    joint_vel_err.unsqueeze(-1),
                    self.joint_vel_err_last.unsqueeze(-1),
                    self.joint_vel_err_last_last.unsqueeze(-1)
                ), dim=2).view(-1, 6)   # shape: (num_envs*num_actions, 6)
                torques = self.actuator_network(self.actuator_net_input).view(self.num_envs, self.num_actions)
                
            self.joint_pos_err_last_last = torch.clone(self.joint_pos_err_last)
            self.joint_vel_err_last_last = torch.clone(self.joint_vel_err_last)
            self.joint_pos_err_last = torch.clone(joint_pos_err)
            self.joint_vel_err_last = torch.clone(joint_vel_err)
            
            return torques
            
        else:
            return super()._compute_torques(actions)