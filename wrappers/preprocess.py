"""Resize wrapper for gym.Env."""
from copy import deepcopy
from typing import Optional

import gym
import gym.spaces
import numpy as np
import cv2
from skimage import color

from models import World


class Preprocess(gym.Wrapper):
    """gym.Env wrapper for preprocessing observations.

    Example:

    Args:

    Raises:

    """

    def __init__(self,
                 env,
                 height: Optional[int] = 64,
                 width: Optional[int] = 64,
                 grayscale: bool = False,
                 world_type: World = World.DIAMOND):

        super().__init__(env)
        obs_space = env.observation_space
        self._world_type = world_type
        self._height = height
        self._width = width
        self._grayscale = grayscale

        self._color_channels = 1 if grayscale else 3

        if world_type == World.ATARI:
            assert isinstance(obs_space, gym.spaces.Box)
            assert (obs_space.low == 0).all() and (obs_space.high == 255).all()
            assert obs_space.shape == (210, 160, 3)

            self._observation_space = gym.spaces.Dict({
                'pov': gym.spaces.Box(
                    low=-0.5,
                    high=0.5,
                    shape=(self._color_channels, height, width),
                    dtype=np.float32
                ),
            })

        elif world_type in [World.DIAMOND, World.BASALT]:
            assert isinstance(obs_space, gym.spaces.Dict)
            assert isinstance(obs_space['pov'], gym.spaces.Box)
            assert (obs_space['pov'].low == 0).all() and \
                (obs_space['pov'].high == 255).all()
            assert obs_space['pov'].shape == (64, 64, 3)

            obs_space = deepcopy(obs_space)
            obs_space.spaces['pov'] = gym.spaces.Box(
                low=-0.5,
                high=0.5,
                shape=(self._color_channels, height, width),
                dtype=np.float32
            )
            self._observation_space = obs_space

        else:
            raise ValueError()

    @property
    def observation_space(self):
        """gym.Env observation space."""
        return self._observation_space

    @observation_space.setter
    def observation_space(self, observation_space):
        self._observation_space = observation_space

    def _preprocess_img(self, img):
        if self._grayscale:
            img = self._apply_grayscale(img)
        img = self._resize_img(img)
        img = self._tranpose_img_channels(img)
        img = self._scale_img(img)
        return img

    def _apply_grayscale(self, img):
        return color.rgb2gray((img))

    def _resize_img(self, img):
        height, width, _ = img.shape
        if (height, width) == (self._height, self._width):
            return img  # already at the correct size
        img = cv2.resize(  # type: ignore
            img, (self._width, self._height),
            interpolation=cv2.INTER_AREA)  # type: ignore
        return img

    def _tranpose_img_channels(self, img):
        """Coverts from HWC to CHW"""
        return np.transpose(img, (2, 0, 1))

    def _scale_img(self, img):
        return img / 255 - 0.5

    def _observation(self, obs):
        if self._world_type == World.ATARI:
            obs = {'pov': self._preprocess_img(obs)}
        elif self._world_type == World.DIAMOND or self._world_type == World.BASALT:
            obs['pov'] = self._preprocess_img(obs['pov'])
        else:
            raise ValueError()
        return obs

    def reset(self):
        """gym.Env reset function."""
        return self._observation(self.env.reset())

    def step(self, action):
        """gym.Env step function."""
        obs, reward, done, info = self.env.step(action)
        return self._observation(obs), reward, done, info
