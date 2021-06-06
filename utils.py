from garage.torch.policies.stochastic_policy import StochasticPolicy
from garage import EnvSpec
from garage.torch import global_device
import gym
import json
import torch
import os

# import safety_gym
# from wrappers import SafetyEnvStateAppender


from ruamel.yaml import YAML
from dotmap import DotMap
yaml = YAML()
with open('./config.yaml', 'r') as f:
    CONFIG = DotMap(yaml.load(f))


def segs_to_batch(segs, env_spec):
    device = global_device()
    obs = torch.tensor(
        [seg.next_observations for seg in segs]).type(torch.float).to(device)
    if CONFIG.image.color_channels == 1:
        obs = obs.unsqueeze(2)
    obs = obs / 255 - 0.5

    actions = torch.tensor(
        [env_spec.action_space.flatten_n(seg.actions) for seg in segs]
    ).type(torch.float).to(device)

    rewards = torch.tensor(
        [seg.rewards for seg in segs]).type(torch.float).to(device)

    discounts = (
        1 - torch.tensor([seg.terminals for seg in segs]).type(torch.float)
    ).to(device)

    return obs, actions, rewards, discounts


class RandomPolicy(StochasticPolicy):

    def __init__(self, env_spec: EnvSpec):
        super().__init__(env_spec=env_spec, name="RandomPolicy")

    def _get_rand_distribution(self, action_space):
        if isinstance(action_space, gym.spaces.Discrete):
            dist = torch.distributions.Categorical(
                probs=torch.ones(action_space.n) / action_space.n
            )
        elif isinstance(action_space, gym.spaces.Box):
            raise NotImplementedError()
        elif isinstance(action_space, gym.spaces.Dict):
            raise NotImplementedError()
        else:
            raise NotImplementedError()

        return dist

    @property
    def env_spec(self):
        return self._env_spec

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
        dist = dist.expand((observations.shape[0],))
        info = dict()
        return dist, info


def preprocess_img(img):
    return img / 255.


# def make_env(env_id, snapshot_dir):
#     env = gym.make(env_id)
#     is_safety_gym_env = isinstance(env, safety_gym.envs.engine.Engine)

#     if is_safety_gym_env:
#         config = env.config

#         with open(os.path.join(snapshot_dir, 'config.json'), 'w') \
#                 as outfile:
#             json.dump(config, outfile)

#         env = SafetyEnvStateAppender(env)

#     env.metadata['render.modes'] = ['rgb_array']
