import collections.abc
import os
import os.path as osp
from collections import OrderedDict
import pickle

import akro
from dowel import logger
import numpy as np
import torch
from dotmap import DotMap
from garage import EnvSpec, EpisodeBatch, StepType
from garage.torch import global_device
from garage.torch.policies import Policy
import minerl
import minerl.data
from ruamel.yaml import YAML
import plotly.graph_objects as go
from video import export_video

import wandb
from replay_buffer import ReplayBuffer


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


def log_eps_video(eps, log_dir, itr):
    obs = np.array([o['pov'] for o in eps.observations])
    obs = np.transpose(obs, (0, 2, 3, 1))

    if not osp.exists(log_dir):
        os.makedirs(log_dir)

    fname = osp.join(log_dir, f'eps_{itr}.mp4')

    export_video(
        frames=obs[:, ::-1],
        fname=fname,
        fps=20
    )

    wandb.log({
        os.path.basename(fname): wandb.Video(fname),
    }, step=itr)


def log_reconstructions(obs, wm_out, log_dir, itr, n=3):

    if not osp.exists(log_dir):
        os.makedirs(log_dir)

    org_img = np.transpose(
        obs['pov'].cpu().numpy().astype(np.uint8),
        (0, 1, 3, 4, 2)
    )

    recon_img = np.transpose(
        np.clip(
            unscale_img(wm_out['recon_obs']['image_recon_dist'].mean)
            .cpu().numpy(), 0, 255
        ).astype(np.uint8),
        (0, 1, 3, 4, 2)
    )

    side_by_side = np.concatenate([org_img, recon_img], axis=3)

    for i in np.random.choice(range(len(side_by_side)), size=n, replace=False):
        fname = osp.join(log_dir, f'reconstructed_{itr}_{i}.mp4')

        export_video(
            frames=side_by_side[i, :, ::-1],
            fname=fname,
            fps=10
        )

        wandb.log({
            os.path.basename(fname): wandb.Video(fname),
        }, step=itr)


def log_action_dist_plots(human_actions, policy_actions, itr):
    for i in range(8):
        fig = go.Figure()  # type: ignore

        fig.add_trace(
            go.Violin(  # type: ignore
                x=np.repeat(np.arange(i*8, i*8+8).reshape(-1, 1),
                            human_actions.shape[1],
                            axis=1).flatten(),
                y=human_actions[i*8: i*8+8].flatten(),
                legendgroup='Human',
                scalegroup='Human',
                name='Human',
                side='negative',
                line_color='blue',
            )
        )

        fig.add_trace(
            go.Violin(  # type: ignore
                x=np.repeat(np.arange(i*8, i*8+8).reshape(-1, 1),
                            policy_actions.shape[1],
                            axis=1).flatten(),
                y=policy_actions[i*8: i*8+8].flatten(),
                legendgroup='Policy',
                scalegroup='Policy',
                name='Policy',
                side='positive',
                line_color='orange',
            )
        )

        fig.update_traces(meanline_visible=True)
        fig.update_layout(violingap=0, violinmode='overlay', title='blah')

        wandb.log({
            f"action_dists_{itr}_{i}": fig,
        }, step=itr)


def flatten_segs_dicts(segs, attr, dtype=torch.float):
    """flattens dicts in segs

    Parameters
    ----------
    segs : TOD
    attr : TODO

    Returns
    -------
    TODO

    """
    keys = getattr(segs[0], attr)[0].keys()
    device = global_device()
    dl = {
        k: torch.tensor(
            # must wrap in np or else very slow
            np.array([[dic[k] for dic in getattr(seg, attr)] for seg in segs]),
            device=device,
            dtype=dtype
        )
        for k in keys
    }
    return dl


def segs_to_batch(segs, env_spec):

    device = global_device()
    rewards = []
    discounts = []

    for seg in segs:
        rewards.append(seg.rewards)
        discounts.append(1 - seg.terminals)

    if type(env_spec.action_space) == akro.Dict:
        actions = flatten_segs_dicts(segs, attr='actions')
    elif type(env_spec.action_space) == akro.Discrete:
        actions = torch.tensor(
                np.array([env_spec.action_space.flatten_n(seg.actions)
                          for seg in segs]),
                device=device,
                dtype=torch.float
            )
    else:
        raise ValueError()

    obs = flatten_segs_dicts(segs, attr='next_observations')

    rewards = torch.tensor(
        np.array(rewards), device=device, dtype=torch.float)

    discounts = torch.tensor(
        np.array(discounts), device=device, dtype=torch.float)

    return obs, actions, rewards, discounts


def unflatten(x):
    k = next(iter(x))
    return np.array([OrderedDict({k: v[i] for k, v in x.items()})
                     for i in range(len(x[k]))])


# MineRL Seq to garage episode
def minerl_seq_to_ep(seq, spec):
    obs, action, rew, next_obs, done = seq
    obs['pov'] = np.transpose(obs['pov'], (0, 3, 1, 2)).copy()
    next_obs['pov'] = np.transpose(next_obs['pov'], (0, 3, 1, 2)).copy()
    length = len(done)
    step_types = np.array([StepType.FIRST] +
                          ([StepType.MID] * (length-2)) +
                          [StepType.TERMINAL], dtype=StepType)

    human_reward_bias = get_config().buffer.human_reward_bias
    ep = EpisodeBatch(
        env_spec=spec,
        episode_infos={},
        observations=unflatten(obs),
        last_observations=np.array([unflatten(next_obs)[-1]]),
        actions=unflatten(action),
        rewards=rew + human_reward_bias,
        env_infos={},
        agent_infos={},
        step_types=step_types,
        lengths=np.array([length])
    )
    return ep


def load_human_data_buffer(env_name: str, spec: EnvSpec):
    data_root = os.environ.get('MINERL_DATA_ROOT')
    buffer_path = osp.join(data_root, env_name, "buffer.pkl")  # type: ignore
    human_reward_bias = get_config().buffer.human_reward_bias

    # if osp.exists(buffer_path) and human_reward_bias == 0:
    #     logger.log('Loading existing human dataset buffer')
    #     with open(buffer_path, 'rb') as f:
    #         buf = pickle.load(f)
    #     return buf
    # else:
    #     logger.log('No human dataset buffer found')

    logger.log('Creating human dataset buffer...')
    data = minerl.data.make(env_name)
    buf = ReplayBuffer(spec, segment_length=get_config().training.seg_length)
    trajectories = data.get_trajectory_names()

    for traj in trajectories:
        path = osp.join(data_root, env_name, traj)  # type: ignore
        seq = data._load_data_pyfunc(path, -1, None)
        buf.collect(minerl_seq_to_ep(seq, spec))

    # logger.log('Saving human dataset buffer...')
    # with open(buffer_path, 'wb') as f:
    #     pickle.dump(buf, f)

    return buf


def get_human_actions(human_buffer: ReplayBuffer, n=10000):
    human_actions = np.array(
        [act['vector'] for act in human_buffer.episodes.actions])
    i = np.random.choice(len(human_actions), size=n, replace=False, p=None)
    human_actions = human_actions[i].T
    return human_actions


def get_actions_mean_and_std(human_buffer: ReplayBuffer):
    human_actions = np.array(
        [act['vector'] for act in human_buffer.episodes.actions])
    mean = human_actions.mean(axis=0)
    std = human_actions.std(axis=0)
    return mean, std


def get_policy_actions(eps: EpisodeBatch):
    return np.array([act['vector'] for act in eps.actions]).T

class RandomPolicy(Policy):

    def __init__(self, env_spec: EnvSpec):
        super().__init__(env_spec=env_spec, name="RandomPolicy")

    @property
    def env_spec(self):
        return self._env_spec

    def get_action(self, observation):
        return self.action_space.sample(), {}

    def get_actions(self, observations):
        pass


def scale_img(img):
    # return img / 255. - 0.5
    return (2*img) / 255. - 1


def unscale_img(img):
    # return (img + 0.5) * 255
    return (img + 1.) * (255/2)
