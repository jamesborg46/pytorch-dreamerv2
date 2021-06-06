"""Reward mask wrapper for gym.Env"""
import gym


class RewardMasker(gym.Wrapper):

    """Reward Mask Wrapper for masking out reward values

    Using this wrapper will set the returned reward to None, instead providing
    this reward value in info['gt_reward']. This is useful for algorithms
    which should use a different reward for example a learnt reward in the case
    of certain IRL algorithms or an intrinsic reward for unsupervised/curiosity
    style algorithms.

    """

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info['gt_reward'] = reward
        masked_reward = None
        return obs, masked_reward, done, info

