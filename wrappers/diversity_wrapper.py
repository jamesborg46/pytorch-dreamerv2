"""Wrapper for diversity is all you need implementation for gym.Env"""
import gym
from gym import spaces
import numpy as np
from gym.spaces import Dict, Discrete

from garage.torch import global_device
import torch


class DiversityWrapper(gym.Wrapper):

    def __init__(self, env,  number_skills=10, skill_mode='random'):
        super().__init__(env)

        self.number_skills = number_skills
        self.original_obs_space = env.observation_space
        self.observation_space = Dict({
            'state': self.original_obs_space,
            'skill': Discrete(number_skills)
        })
        self.skill = None
        self.skill_mode = skill_mode

    def set_skill_mode(self, skill_mode):
        if skill_mode not in ['random', 'consecutive', 'constant']:
            raise ValueError("skill_mode must be 'random', 'consecutive' or"
                             "constant")
        self.skill_mode = skill_mode

    def set_skill(self, skill):
        if skill is not None:
            self.skill = skill
            self.metadata['skill'] = skill

    def reset(self):
        if self.skill_mode not in ['random', 'consecutive', 'constant']:
            raise ValueError("skill_mode must be 'random', 'consecutive' or"
                             "constant")

        if self.skill_mode == 'random':
            self.set_skill(
                np.random.randint(low=0, high=self.number_skills)
            )
        elif self.skill_mode == 'consecutive':
            self.set_skill(
                (self.skill + 1) % self.number_skills
            )
        elif self.skill_mode == 'constant':
            pass
        else:
            raise Exception()

        obs = self.env.reset()

        return {'state': obs, 'skill': self.skill}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # info['gt_reward'] = reward
        # new_obs = np.concatenate([self.skill_one_hot, obs])
        new_obs = {'state': obs, 'skill': self.skill}
        return new_obs, reward, done, info

