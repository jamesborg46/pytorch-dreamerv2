#!/usr/bin/env python3
"""
TODO
"""
import torch
import torch.nn.functional as F  # noqa: F401

from garage import wrap_experiment
from garage.envs import GymEnv
from garage.envs.wrappers import (
    ClipReward, EpisodicLife,  FireReset, Grayscale,  MaxAndSkip, Noop,
    Resize, StackFrames)
from garage.experiment.deterministic import set_seed
from garage.torch import set_gpu_mode
from garage.sampler import LocalSampler, RaySampler, DefaultWorker
from garage.trainer import Trainer
from garage.torch.optimizers import OptimizerWrapper  # noqa: F401

import gym
import gym.envs.atari
from replay_buffer import ReplayBuffer # noqa: F401
from dreamer import Dreamer
from models import WorldModel
from utils import CONFIG, RandomPolicy

import time
import os
import argparse
import json
import pickle
import dowel
from dowel import logger


def dreamer(ctxt):

    snapshot_dir = ctxt.snapshot_dir

    env = gym.envs.atari.AtariEnv(
        CONFIG.env.name, obs_type='image', frameskip=1,
        repeat_action_probability=0.25, full_action_space=False)
    env = Noop(env, noop_max=30)
    env = MaxAndSkip(env, skip=4)
    # env = EpisodicLife(env)
    if CONFIG.image.color_channels == 1:
        env = Grayscale(env)
    env = Resize(env, CONFIG.image.height, CONFIG.image.height)
    max_episode_length = 108000 / 4
    env = GymEnv(env, max_episode_length=max_episode_length, is_image=True)

    set_seed(CONFIG.training.seed)

    with open(os.path.join(snapshot_dir, 'env.pkl'), 'wb') as outfile:
        pickle.dump(env, outfile)

    trainer = Trainer(ctxt)

    buf = ReplayBuffer(env.spec)
    policy = RandomPolicy(env.spec)
    world_model = WorldModel(env.spec)

    if CONFIG.training.sampler == "ray":
        Sampler = RaySampler
    elif CONFIG.training.sampler == "local":
        Sampler = LocalSampler

    sampler = Sampler(agents=policy,  # noqa: F841
                      envs=env,
                      max_episode_length=max_episode_length,
                      n_workers=4)

#     log_sampler = Sampler(agents=policy,  # noqa: F841
#                           envs=env,
#                           max_episode_length=kwargs['max_episode_length'],
#                           n_workers=kwargs['n_workers'])

    set_gpu_mode(True)

    algo = Dreamer(
        env.spec,
        sampler=sampler,
        world_model=world_model,
        agent=policy,
        buf=buf,
    )

    trainer.setup(
        algo=algo,
        env=env,
    )

    trainer.train(n_epochs=CONFIG.training.n_epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='preference reward learning')
    parser.add_argument('--name', type=str, required=True)

    torch.set_num_threads(4)

    kwargs = vars(parser.parse_args())
    args = {k: v for k, v in kwargs.items() if v is not None}

    kwargs['name'] = (
        kwargs['name'] + '_' + time.ctime().replace(' ', '_')
    )

    experiment_dir = os.getenv('EXPERIMENT_LOGS',
                               default=os.path.join(os.getcwd(), 'experiment'))

    log_dir = os.path.join(experiment_dir,
                           kwargs['name'] + time.ctime().replace(' ', '_'))

    logger.add_output(
        dowel.WandbOutput(
            project='dreamer',
            name=args['name'],
            config=CONFIG,
        )
    )

    dreamer = wrap_experiment(
        dreamer,
        name=kwargs['name'],
        snapshot_gap=100,
        snapshot_mode='gap_overwrite',
        log_dir=log_dir,
    )
    dreamer()