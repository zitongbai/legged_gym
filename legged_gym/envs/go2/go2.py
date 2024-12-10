from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch

from legged_gym.envs import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR

from .go2 import GO2RoughCfg

class Go2(LeggedRobot):
    cfg : GO2RoughCfg
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)

    def _init_buffers(self):
        super()._init_buffers()

    def _compute_torques(self, actions):
        return super()._compute_torques(actions)