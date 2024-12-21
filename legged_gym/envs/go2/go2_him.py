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
        
        pass