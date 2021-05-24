
from garage.torch.policies.stochastic_policy import StochasticPolicy
from garage import EnvSpec
import gym
import numpy as np
import torch



class RandomPolicy(StochasticPolicy):

    def __init__(self, env_spec: EnvSpec):

        self.env_spec = env_spec
        self.action_space = env_spec.action_space

    def _get_rand_distribution(self, action_space):
        if isinstance(action_space, gym.spaces.Discrete):
            dist = torch.distribution.Categorical(
                probs=np.ones(action_space.n) / action_space.n
            )
        elif isinstance(action_space, gym.spaces.Box):
            raise NotImplementedError()
        elif isinstance(action_space, gym.spaces.Dict):
            raise NotImplementedError()
        else:
            raise NotImplementedError()

        return dist

    def forward(self, observations):
        """Compute the action distributions from the observations.

        Args:
            observations (torch.Tensor): Batch of observations on default
                torch device.

        Returns:
            torch.distributions.Distribution: Batch distribution of actions.
            dict[str, torch.Tensor]: Additional agent_info, as torch Tensors.
                Do not need to be detached, and can be on any device.
        """

        dist = self._get_rand_distribution(self.action_space)
        dist.expand(observations.shape[0])
        info = dict()
        return dist, info

def preprocess_img(img):
    return img / 255.
