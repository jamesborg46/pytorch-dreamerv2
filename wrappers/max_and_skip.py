"""Max and Skip wrapper for gym.Env."""
import gym
import numpy as np


class MaxAndSkip(gym.Wrapper):
    """Max and skip wrapper for gym.Env.

    It returns only every `skip`-th frame. Action are repeated and rewards are
    sum for the skipped frames.

    It also takes element-wise maximum over the last two consecutive frames,
    which helps algorithm deal with the problem of how certain Atari games only
    render their sprites every other game frame.

    Args:
        env (gym.Env): The environment to be wrapped.
        skip (int): The environment only returns `skip`-th frame.

    """

    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last two observations.

        Args:
            action (int): action to take in the atari environment.

        Returns:
            np.ndarray: observation of shape :math:`(O*,)` representating
                the max values over the last two oservations.
            float: Reward for this step
            bool: Termination signal
            dict: Extra information from the environment.

        """
        total_reward = 0.0
        done = None
        last = None
        second_last = None

        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            second_last = last
            last = obs
            total_reward += reward
            if done:
                break
        max_frame = last if second_last is None else np.maximum(last, second_last)
        return max_frame, total_reward, done, info

    # pylint: disable=arguments-differ
    def reset(self):
        """gym.Env reset.

        Returns:
            np.ndarray: observaion of shape :math:`(O*,)`.
        """
        return self.env.reset()
