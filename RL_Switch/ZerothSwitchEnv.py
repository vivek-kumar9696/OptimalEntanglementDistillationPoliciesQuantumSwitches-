from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from gymnasium.spaces.space import MaskNDArray, Space
import gymnasium as gym
from gymnasium import spaces
from zeroth_iteration_switchModel import *
import random
        
        
        
    
class ZerothSwitchEnv(gym.Env):
    
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    
    def __init__(self, switch_config_file, switch_name, fid_thresh, distillFlag, x_value = None):
        
        
        self.distillFlag = distillFlag
        self.fid_thresh = fid_thresh
        self.obs_dim = 0
        self.switch_name = switch_name
        self.switch_config_file = switch_config_file
        self.x_value = x_value
        
        super(ZerothSwitchEnv, self).__init__()
        
        
        self.switchModel = Switch(self.switch_config_file,self.switch_name, self.fid_thresh, self.distillFlag, self.x_value)
        self.action_array = [self.switchModel.no_users+1] * (self.switchModel.no_users * self.switchModel.queue_length)
        

        # Define action space
        # They must be gym.spaces objects
        self.possible_actions_dict = self.switchModel.possible_actions_dict
        self.action_space = spaces.Discrete(len(self.possible_actions_dict))
        
        
        # Define observation space
        # They must be gym.spaces objects
        self.observation_space = spaces.Box(low=0, high=1,shape=(self.switchModel.no_users, self.switchModel.queue_length), dtype="float32")
        #self.observation_space = spaces.Box(low=0, high=1,shape=(1,self.no_users * self.queue_length), dtype="float32")

        #monkey patching existing methods to override functionality
        self.action_space.sample = self.sample
        
        
    def action_masks(self):
            return self.switchModel.action_mask()
        
    
        
    def sample(self, mask: MaskNDArray | None = None) -> NDArray[np.int8]:
        r"""Generates a single random sample inside the Box.

        In creating a sample of the box, each coordinate is sampled (independently) from a distribution
        that is chosen according to the form of the interval:

        * :math:`[a, b]` : uniform distribution
        * :math:`[a, \infty)` : shifted exponential distribution
        * :math:`(-\infty, b]` : shifted negative exponential distribution
        * :math:`(-\infty, \infty)` : normal distribution

        Args:
            mask: A mask for sampling values from the Box space, currently unsupported.

        Returns:
            A sampled value from the Box
        """
        if mask is not None:
            raise gym.error.Error(
                f"Box.sample cannot be provided a mask, actual value: {mask}"
            )

        if self.distillFlag == True:
            available_actions = self.switchModel.allActions()
        else:
            available_actions = self.switchModel.allActions_onlySwap()
            
        sampleAction = random.choice(available_actions)
        
        return sampleAction
    

    def step(self, action):
        
        info = {}
        terminated = False
        truncated = False
        switchAction = self.possible_actions_dict[int(action.item())]
        observation, reward = self.switchModel.takeAction(switchAction)
        
        return observation, reward, terminated, truncated, info
    
    def reset(self, seed = None, options = 0):
        info = {}
        self.switchModel.switchReset()
        return self.switchModel.getSwitchState() , info
    
    def getGreedyAction(self, observation):
        pass

    
