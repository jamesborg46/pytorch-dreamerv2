#!/usr/bin/env python3
"""
TODO
"""
import argparse
import os
import pickle
import socket
import time

import dowel
import gym
import gym.envs.atari
import torch
import torch.nn.functional as F  # type: ignore  # noqa: F401
from dowel import logger
from garage import wrap_experiment
from garage.envs import GymEnv
from garage.envs.wrappers import EpisodicLife, Grayscale, Noop
from garage.experiment.deterministic import set_seed
from garage.sampler import DefaultWorker, LocalSampler, RaySampler, VecWorker
from garage.torch import set_gpu_mode
from garage.torch.optimizers import OptimizerWrapper  # noqa: F401
from garage.trainer import Trainer

from dreamer import Dreamer
from models import ActorCritic, WorldModel, World
from replay_buffer import ReplayBuffer  # noqa: F401
from utils import RandomPolicy, get_config, set_config
from wrappers import MaxAndSkip, Renderer, Resize, Preprocess


def dreamer(ctxt, gpu_id=0):

    snapshot_dir = ctxt.snapshot_dir

    CONFIG = get_config()
    max_episode_length = (
        108000 / 4 if CONFIG.training.max_episode_length is None
        else CONFIG.training.max_episode_length
    )

    env = gym.envs.atari.AtariEnv(
        CONFIG.env.name, obs_type='image', frameskip=1,
        repeat_action_probability=0.25, full_action_space=True)
    env = Noop(env, noop_max=30)
    env = MaxAndSkip(env, skip=4)
    # env = EpisodicLife(env)
    # if CONFIG.image.color_channels == 1:
    #     env = Grayscale(env)
    # env = Resize(env, CONFIG.image.height, CONFIG.image.height)
    env = Preprocess(env, world_type=World.ATARI, grayscale=True)
    env = Renderer(env, directory=os.path.join(snapshot_dir, 'videos'))
    env = GymEnv(env, max_episode_length=max_episode_length, is_image=True)

    set_seed(CONFIG.training.seed)

    with open(os.path.join(snapshot_dir, 'env.pkl'), 'wb') as outfile:
        pickle.dump(env, outfile)

    trainer = Trainer(ctxt)

    buf = ReplayBuffer(env.spec,
                       segment_length=CONFIG.training.seg_length)
    world_model = WorldModel(env.spec,
                             config=CONFIG,
                             world_type=World.ATARI)
    agent = ActorCritic(env.spec,
                        world_model,
                        config=CONFIG,
                        world_type=World.ATARI)

    if CONFIG.training.sampler == "ray":
        Sampler = RaySampler
    elif CONFIG.training.sampler == "local":
        Sampler = LocalSampler

    sampler = Sampler(  # noqa: F841
        agents=agent,
        envs=env,
        max_episode_length=max_episode_length,
        n_workers=CONFIG.samplers.agent.n)

    log_sampler = Sampler(  # noqa: F841
        agents=agent,
        envs=env,
        max_episode_length=max_episode_length,
        n_workers=CONFIG.samplers.log.n)

    if gpu_id < 0:
        set_gpu_mode(False)
        mixed_prec = False
    else:
        set_gpu_mode(True, gpu_id=gpu_id)
        mixed_prec = True

    algo = Dreamer(
        env.spec,
        sampler=sampler,
        log_sampler=log_sampler,
        world_model=world_model,
        agent=agent,
        buf=buf,
        mixed_prec=mixed_prec,
    )

    trainer.setup(
        algo=algo,
        env=env,
    )

    trainer.train(n_epochs=CONFIG.training.n_epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='preference reward learning')
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--config', type=str, default='defaults')

    torch.set_num_threads(4)

    kwargs = vars(parser.parse_args())
    args = {k: v for k, v in kwargs.items() if v is not None}

    kwargs['name'] = (
        kwargs['name'] + '_' + time.ctime().replace(' ', '_')
    )

    set_config(kwargs['config'])

    experiment_dir = os.getenv('EXPERIMENT_LOGS',
                               default=os.path.join(os.getcwd(), 'experiment'))

    log_dir = os.path.join(experiment_dir,
                           kwargs['name'] + time.ctime().replace(' ', '_'))
    config = get_config()
    hostname = socket.gethostname()
    config['hostname'] = hostname

    logger.add_output(
        dowel.WandbOutput(
            project='dreamer',
            name=args['name'],
            config=config,
        )
    )

    dreamer = wrap_experiment(
        dreamer,
        name=kwargs['name'],
        snapshot_gap=100,
        snapshot_mode='gap_overwrite',
        log_dir=log_dir,
    )

    dreamer(gpu_id=kwargs['gpu'])

