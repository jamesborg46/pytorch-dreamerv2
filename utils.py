from garage.torch.policies.stochastic_policy import StochasticPolicy
from garage import EnvSpec
from garage.torch import global_device
from garage.sampler.env_update import EnvUpdate
import collections.abc
import gym
import json
import math
import torch
import os
import os.path as osp
import ray
import wandb
import numpy as np

from ruamel.yaml import YAML
from dotmap import DotMap
from video import export_video

def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def set_config(config_type='defaults'):
    """Set GPU mode and device ID.

    Args:
        mode (bool): Whether or not to use GPU
        gpu_id (int): GPU ID

    """
    # pylint: disable=global-statement
    global _CONFIG

    yaml = YAML()
    with open('./config.yaml', 'r') as f:
        base = DotMap(yaml.load(f))
        if config_type == 'defaults':
            _CONFIG = base.defaults
        else:
            assert config_type in base.keys()
            _CONFIG = update(base.defaults, base[config_type])

def get_config():
    global _CONFIG
    return _CONFIG


class EnvConfigUpdate(EnvUpdate):

    def __init__(self,
                 enable_render=False,
                 file_prefix=""):

        self.enable_render = enable_render
        self.file_prefix = file_prefix

    def __call__(self, old_env):
        old_env.enable_rendering(self.enable_render,
                                 file_prefix=self.file_prefix)
        return old_env


class CloseRenderer(EnvUpdate):

    def __call__(self, old_env):
        old_env.close_renderer()
        return old_env


def log_episodes(itr,
                 snapshot_dir,
                 sampler,
                 policy,
                 number_eps=None,
                 enable_render=False,
                 ):

    if hasattr(sampler, '_worker_factory'):
        n_workers = sampler._worker_factory.n_workers
    else:
        n_workers = sampler._factory.n_workers

    n_eps_per_worker = (
        1 if number_eps is None else math.ceil(number_eps / n_workers)
    )

    env_updates = []

    for i in range(n_workers):
        env_updates.append(EnvConfigUpdate(
            enable_render=enable_render,
            file_prefix=f"epoch_{itr:04}_worker_{i:02}"
        ))

    sampler._update_workers(
        env_update=env_updates,
        agent_update={k: v.cpu() for k, v in policy.state_dict().items()}
    )

    episodes = sampler.obtain_exact_episodes(
        n_eps_per_worker=n_eps_per_worker,
        agent_update={k: v.cpu() for k, v in policy.state_dict().items()}
    )

    if enable_render:
        env_updates = [CloseRenderer() for _ in range(n_workers)]

        updates = sampler._update_workers(
            env_update=env_updates,
            agent_update={k: v.cpu() for k, v in policy.state_dict().items()}
        )

        while updates:
            ready, updates = ray.wait(updates)

    if enable_render:
        for episode in episodes.split():
            video_file = episode.env_infos['video_filename'][0]
            assert '.mp4' in video_file
            wandb.log({
                os.path.basename(video_file): wandb.Video(video_file),
            }, step=itr)

    return episodes


def log_imagined_rollouts(eps,
                          env_spec,
                          world_model,
                          itr,
                          path):

    with torch.no_grad():
        for i, ep in enumerate(eps.split()):
            obs = (torch.tensor(eps.observations).type(torch.float)
                   .unsqueeze(1)).to(global_device()) / 255 - 0.5
            actions = (torch.tensor(env_spec.action_space.flatten_n(eps.actions))
                       .type(torch.float)).to(global_device())
            steps, channels, height, width = obs.shape
            embedded_observations = world_model.image_encoder(obs[:5])

            # Run first five steps with observations
            out = world_model.observe(embedded_observations[:5].unsqueeze(0),
                                      actions[:5].unsqueeze(0))
            recon_latent_states = out['latent_states'].reshape(
                5, world_model.latent_state_size)

            initial_stoch = out['posterior_samples'][:1, -1]
            inital_deter = out['deters'][:1, -1]
            _, imagined_latent_states, _, _ = (
                world_model.imagine(initial_stoch, inital_deter,
                                    actions=actions[5:].unsqueeze(1))
            )
            imagined_latent_states = imagined_latent_states.reshape(
                -1, world_model.latent_state_size)
            latent_states = torch.cat(
                [recon_latent_states, imagined_latent_states], dim=0)

            image_recon = world_model.image_decoder(latent_states).reshape(
                steps, channels, height, width).cpu().numpy()

            image_recon = np.transpose(image_recon, (0, 2, 3, 1))
            original_obs = np.transpose(obs.cpu().numpy(), (0, 2, 3, 1))

            if original_obs.shape[-1] == 1:
                image_recon = np.tile(image_recon, (1, 1, 1, 3))
                original_obs = np.tile(original_obs, (1, 1, 1, 3))

            original_obs = (original_obs + 0.5) * 255
            image_recon = np.clip((image_recon + 0.5) * 255, 0, 255)
            original_obs = original_obs.astype(np.uint8)
            image_recon = image_recon.astype(np.uint8)

            side_by_side = np.concatenate([original_obs, image_recon], axis=2)

            fname = osp.join(path, f'imagined_{itr}_{i}.mp4')
            export_video(
                frames=side_by_side[:, ::-1],
                fname=fname,
                fps=10
            )

            wandb.log({
                os.path.basename(fname): wandb.Video(fname),
            }, step=itr)


def log_reconstructions(eps,
                        env_spec,
                        world_model,
                        itr,
                        path):

    with torch.no_grad():
        lengths = eps.lengths
        obs = (torch.tensor(eps.observations).type(torch.float)
               .unsqueeze(1)).to(global_device()) / 255 - 0.5
        actions = (torch.tensor(env_spec.action_space.flatten_n(eps.actions))
                   .type(torch.float)).to(global_device())

        image_recon = world_model.reconstruct(obs, actions).cpu().numpy()
        image_recon = np.transpose(image_recon, (0, 2, 3, 1))
        original_obs = np.transpose(obs.cpu().numpy(), (0, 2, 3, 1))

        if original_obs.shape[-1] == 1:
            image_recon = np.tile(image_recon, (1, 1, 1, 3))
            original_obs = np.tile(original_obs, (1, 1, 1, 3))

        original_obs = (original_obs + 0.5) * 255
        image_recon = np.clip((image_recon + 0.5) * 255, 0, 255)
        original_obs = original_obs.astype(np.uint8)
        image_recon = image_recon.astype(np.uint8)

        side_by_side = np.concatenate([original_obs, image_recon], axis=2)
        start = 0
        for i, length in enumerate(lengths):
            fname = osp.join(path, f'reconstructed_{itr}_{i}.mp4')
            export_video(
                frames=side_by_side[start:start+length, ::-1],
                fname=fname,
                fps=10
            )
            start += length

            wandb.log({
                os.path.basename(fname): wandb.Video(fname),
            }, step=itr)


def segs_to_batch(segs, env_spec):

    device = global_device()
    obs = []
    actions = []
    rewards = []
    discounts = []

    for seg in segs:
        obs.append(seg.next_observations)
        actions.append(env_spec.action_space.flatten_n(seg.actions))
        rewards.append(seg.rewards)
        discounts.append(1 - seg.terminals)

    obs = torch.tensor(np.array(obs), device=device, dtype=torch.float)
    obs = obs.unsqueeze(2)
    obs = obs / 255 - 0.5

    actions = torch.tensor(
        np.array(actions), device=device, dtype=torch.float)

    rewards = torch.tensor(
        np.array(rewards), device=device, dtype=torch.float)

    discounts = torch.tensor(
        np.array(discounts), device=device, dtype=torch.float)

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

