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
import socket

from utils import set_config, get_config, minerl_segs_to_batch
from models import WorldModel, ActorCritic
from video import export_video

import torch
from torch import optim
import numpy as np

scaler = torch.cuda.amp.GradScaler(enabled=True)


MINERL_DATASET = '/data/umihebi0/users/james/MineRL'


def main(log_dir):

    data = minerl.data.make('MineRLTreechop-v0', data_dir=MINERL_DATASET)

    world_model = WorldModel(data._env_spec, config=get_config()).to(global_device())

    world_optimizer = optim.Adam(
        world_model.parameters(),
        lr=get_config().world.lr,
        eps=get_config().world.eps,
        weight_decay=get_config().world.wd)

    i = 0
    for segs in data.batch_iter(batch_size=50, num_epochs=10, seq_len=50):

        obs, actions, rewards, discounts = minerl_segs_to_batch(
            segs, data._env_spec)

        wm_out = train_world_model_once(
            world_model, world_optimizer, obs, actions, rewards, discounts
        )

        logger.log(tabular)
        logger.dump_all(i)
        tabular.clear()
        i += 1

        if i and i % get_config().logging.log_vid_freq == 0:
            log_reconstructions(data, data._env_spec, world_model,
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


def log_reconstructions(data, env_spec, world_model, itr, path, n=3):
    segs = next(data.batch_iter(batch_size=n, num_epochs=1, seq_len=200))
    obs, actions, rewards, discounts = minerl_segs_to_batch(segs, data._env_spec)

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

    main(log_dir=log_dir)

