import argparse
import dowel
from dowel import logger, tabular
import gym
from garage.torch import set_gpu_mode, global_device
import minerl
import torch
import wandb

import os
import os.path as osp
import time
import pickle
import socket

from utils import set_config, get_config, sample_minerl_segments
from models import WorldModel, ActorCritic
from video import export_video

import torch
from torch import optim
import numpy as np

scaler = torch.cuda.amp.GradScaler(enabled=True)


MINERL_DATASET = '/data/umihebi0/users/james/MineRL'
ENV_SPEC = minerl.data.make('MineRLTreechop-v0', data_dir=MINERL_DATASET)._env_spec


def main(log_dir):

    print('LOADING DATA')
    with open('./minerldata.pickle', 'rb') as f:
        episodes = pickle.load(f)
    print('LOADING DATA COMPLETE')

    world_model = WorldModel(ENV_SPEC, config=get_config()).to(global_device())

    world_optimizer = optim.Adam(
        world_model.parameters(),
        lr=get_config().world.lr,
        eps=get_config().world.eps,
        weight_decay=get_config().world.wd)

    epochs = 50000
    steps_per_epoch = 100
    for i in range(epochs * steps_per_epoch):

        obs, actions, rewards, discounts = sample_minerl_segments(episodes,
                                                                  n=50)

        wm_out = train_world_model_once(
            world_model, world_optimizer, obs, actions, rewards, discounts
        )

        logger.log(tabular)
        logger.dump_all(i)
        tabular.clear()
        i += 1

        if i and i % (get_config().logging.log_vid_freq * steps_per_epoch) == 0:
            with torch.no_grad():
                segs = sample_minerl_segments(
                    episodes,
                    n=3,
                    segment_length=400)
                log_reconstructions(segs, ENV_SPEC, world_model,
                                    i, osp.join(log_dir, 'videos'))

                segs = sample_minerl_segments(
                    episodes,
                    n=3,
                    segment_length=200)

                log_imagined_rollouts(segs, ENV_SPEC, world_model,
                                    i, osp.join(log_dir, 'videos'))


def train_world_model_once(world_model, world_optimizer,
                           obs, actions, rewards, discounts):

    world_optimizer.zero_grad()

    with torch.cuda.amp.autocast():
        out = world_model(obs, actions)
        loss, loss_info = world_model.loss(out, obs, rewards, discounts)

    scaler.scale(loss).backward(retain_graph=True)
    scaler.unscale_(world_optimizer)

    torch.nn.utils.clip_grad_norm_(
        world_model.parameters(), get_config().training.grad_clip)
    scaler.step(world_optimizer)
    scaler.update()

    with tabular.prefix('world_model_'):
        for k, v in loss_info.items():
            tabular.record(k, v.cpu().item())

    print(loss)

    return out


def log_reconstructions(segs, env_spec, world_model, itr, path, n=3):
    obs, actions, rewards, discounts = segs

    n, seq, channels, height, width = obs.shape
    embedded = world_model.image_encoder(obs.reshape(n*seq, channels, height, width))
    out = world_model.observe(embedded.reshape(n, seq, -1), actions)
    image_recon = world_model.image_decoder(
        out['latent_states'].reshape(n*seq, -1),
    ).reshape(n, seq, channels, height, width).cpu().numpy()

    image_recon = np.clip(
        (np.transpose(image_recon, (0, 1, 3, 4, 2)) + 0.5) * 255,
        0, 255).astype(np.uint8)
    original_obs = ((np.transpose(obs.cpu().numpy(), (0, 1, 3, 4, 2)) + 0.5) * 255).astype(np.uint8)
    side_by_side = np.concatenate([original_obs, image_recon], axis=3)

    for i, img in enumerate(side_by_side):
        fname = osp.join(path, f'reconstructed_{itr}_{i}.mp4')
        export_video(
            frames=img[:, ::-1],
            fname=fname,
            fps=10
        )

        wandb.log({
            os.path.basename(fname): wandb.Video(fname),
        }, step=itr)


def log_imagined_rollouts(segs,
                          env_spec,
                          world_model,
                          itr,
                          path):

    obs, actions, rewards, discounts = segs
    n, seq, channels, height, width = obs.shape
    embedded = world_model.image_encoder(
        obs.reshape(n*seq, channels, height, width)
    ).reshape(n, seq, -1)

    out = world_model.observe(embedded[:,:5],
                              actions[:,:5])
    initial_stoch = out['posterior_samples'][:, -1]
    inital_deter = out['deters'][:, -1]
    _, imagined_latent_states, _, _ = (
        world_model.imagine(initial_stoch, inital_deter,
                            actions=torch.swapaxes(actions, 0, 1)[5:])
    )
    imagined_latent_states = torch.swapaxes(imagined_latent_states, 0, 1)
    latent_states = torch.cat([
        out['latent_states'], imagined_latent_states
    ], dim=1)
    n, seq, feat = latent_states.shape
    image_recon = world_model.image_decoder(
        latent_states.reshape(n*seq, feat)).reshape(n, seq, channels, height, width).cpu().numpy()

    image_recon = np.transpose(image_recon, (0, 1, 3, 4, 2))
    original_obs = np.transpose(obs.cpu().numpy(), (0, 1, 3, 4, 2))

    original_obs = (original_obs + 0.5) * 255
    image_recon = np.clip((image_recon + 0.5) * 255, 0, 255)
    original_obs = original_obs.astype(np.uint8)
    image_recon = image_recon.astype(np.uint8)

    side_by_side = np.concatenate([original_obs, image_recon], axis=3)

    for i, img in enumerate(side_by_side):
        fname = osp.join(path, f'imagined_{itr}_{i}.mp4')
        export_video(
            frames=img[:, ::-1],
            fname=fname,
            fps=10
        )

        wandb.log({
            os.path.basename(fname): wandb.Video(fname),
        }, step=itr)


if __name__ == "__main__":
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
    set_gpu_mode(True, kwargs['gpu'])

    logger.add_output(
        dowel.WandbOutput(
            project='dreamer',
            name=args['name'],
            config=config,
        )
    )

    os.makedirs(osp.join(log_dir, 'videos'))
    main(log_dir=log_dir)
