import os.path as osp
import time
from contextlib import nullcontext
from typing import Union

import models
import numpy as np
import replay_buffer
import torch
from dowel import logger, tabular
from garage import EnvSpec
from garage.np.algos import RLAlgorithm
from garage.sampler import LocalSampler, RaySampler
from garage.torch import global_device
from torch import optim
from torch.cuda.amp import autocast
from tqdm import tqdm
from utils import (RandomPolicy, get_config, log_episodes,
                   log_imagined_rollouts, log_reconstructions, segs_to_batch)

scaler = torch.cuda.amp.GradScaler(enabled=True)


class Dreamer(RLAlgorithm):

    def __init__(self,
                 env_spec: EnvSpec,
                 sampler: Union[LocalSampler, RaySampler],
                 log_sampler: Union[LocalSampler, RaySampler],
                 world_model: models.WorldModel,
                 agent: models.ActorCritic,
                 buf: replay_buffer.ReplayBuffer,
                 mixed_prec: bool,
                 ):

        device = global_device()
        self.env_spec = env_spec
        self._sampler = sampler
        self._log_sampler = log_sampler
        self.world_model = world_model.to(device)
        self.agent = agent.to(device)
        self.buffer = buf
        self.mixed_prec = mixed_prec

        self.log_vid_freq = get_config().logging.log_vid_freq
        self._num_segs_per_batch = get_config().training.num_segs_per_batch
        self._num_training_steps = get_config().training.num_training_steps

        self.world_optimizer = optim.Adam(
            self.world_model.parameters(),
            lr=get_config().world.lr,
            eps=get_config().world.eps,
            weight_decay=get_config().world.wd)

        self.actor_optimizer = optim.Adam(
            self.agent.actor.parameters(),
            lr=get_config().actor.lr,
            eps=get_config().actor.eps,
            weight_decay=get_config().actor.wd)

        self.critic_optimizer = optim.Adam(
            self.agent.critic.parameters(),
            lr=get_config().critic.lr,
            eps=get_config().critic.eps,
            weight_decay=get_config().critic.wd)

    def agent_update(self):
        sampler_type = type(self._sampler)
        if sampler_type == RaySampler:
            return {k: v.cpu() for k, v in self.agent.state_dict().items()}
        elif sampler_type == LocalSampler:
            return {k: v for k, v in self.agent.state_dict().items()}

    def train(self, trainer):
        """Obtain samplers and start actual training for each epoch.

        Args:
            trainer (Trainer): Gives the algorithm the access to
                :method:`~Trainer.step_epochs()`, which provides services
                such as snapshotting and sampler control.

        Returns:
            float: The average return in last epoch cycle.

        """

        logger.log('INITIALIZING')
        self._initialize_dataset(trainer,)

        for i in trainer.step_epochs():
            logger.log('COLLECTING')
            start = time.time()
            eps = self._sampler.obtain_exact_episodes(
                n_eps_per_worker=get_config().samplers.agent.eps,
                agent_update=self.agent_update()
            )
            self.buffer.collect(eps)
            self.buffer.log_stats(trainer.step_itr)
            print(time.time() - start)

            with tabular.prefix('rollouts_'):
                tabular.record(
                    'average_return',
                    np.mean([ep.rewards.sum() for ep in eps.split()]))

            logger.log('TRAINING')
            for j in tqdm(range(self._num_training_steps)):
                segs = self.buffer.sample_segments(
                    n=self._num_segs_per_batch)
                obs, actions, rewards, discounts = segs_to_batch(
                    segs, self.env_spec)
                start = time.time()
                wm_out = self.train_world_model_once(
                    obs, actions, rewards, discounts, self.mixed_prec
                )
                print("train time:", time.time() - start)

                if i >= get_config().world.pretrain:
                    self.train_actor_critic_once(wm_out, self.mixed_prec)

                logger.log(tabular)
                logger.dump_all(trainer.step_itr)
                tabular.clear()
                trainer.step_itr += 1

            self.agent.update_target_critic()

            # if i and i % self.log_vid_freq == 0:

            #     eps = log_episodes(
            #         trainer.step_itr,
            #         trainer._snapshotter.snapshot_dir,
            #         sampler=self._log_sampler,
            #         agent_update=self.agent_update(),
            #         policy=self.agent,
            #         enable_render=True,
            #     )

                # log_reconstructions(
                #     eps,
                #     self.env_spec,
                #     self.world_model,
                #     trainer.step_itr,
                #     path=osp.join(trainer._snapshotter.snapshot_dir, 'videos')
                # )

                # log_imagined_rollouts(
                #     eps,
                #     self.env_spec,
                #     self.world_model,
                #     trainer.step_itr,
                #     path=osp.join(trainer._snapshotter.snapshot_dir, 'videos')
                # )

    def train_actor_critic_once(self, wm_out, mixed_prec=True):

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        context = autocast() if mixed_prec else nullcontext()
        with context:

            initial_stoch = wm_out['posterior_samples'].reshape(
                -1, get_config().rssm.stoch_state_size,
                get_config().rssm.stoch_state_classes).detach()
            initial_deter = wm_out['deters'].reshape(
                -1, get_config().rssm.det_state_size).detach()

            act_out = self.agent(initial_stoch, initial_deter)

            actor_loss, actor_entropy = self.agent.actor_loss(
                act_out['latents'],
                act_out['values'],
                act_out['actions'],
                act_out['targets'],
                act_out['weights'])

            critic_loss = self.agent.critic_loss(
                act_out['latents'], act_out['targets'], act_out['weights']
            )

        if mixed_prec:
            scaler.scale(actor_loss).backward(retain_graph=True)
            scaler.scale(critic_loss).backward()
            scaler.unscale_(self.actor_optimizer)
            scaler.unscale_(self.critic_optimizer)
        else:
            actor_loss.backward(retain_graph=True)
            critic_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.agent.actor.parameters(), get_config().training.grad_clip)
        torch.nn.utils.clip_grad_norm_(
            self.agent.critic.parameters(), get_config().training.grad_clip)

        if mixed_prec:
            scaler.step(self.actor_optimizer)
            scaler.step(self.critic_optimizer)
            scaler.update()
        else:
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        with tabular.prefix('actor_critic_'):
            tabular.record('actor_loss', actor_loss.cpu().item())
            tabular.record('actor_entropy', actor_entropy.mean().cpu().item())
            tabular.record('critic_loss', critic_loss.cpu().item())
            tabular.record_misc_stat(
                'pred_discount',
                act_out['discounts'].cpu().detach().flatten().tolist())
            tabular.record_misc_stat(
                'pred_reward',
                act_out['rewards'].cpu().detach().flatten().tolist())
            tabular.record_misc_stat(
                'pred_values',
                act_out['values'].cpu().detach().flatten().tolist())
            tabular.record_misc_stat(
                'pred_target_values',
                act_out['values_t'].cpu().detach().flatten().tolist())

    def train_world_model_once(
            self,
            obs,
            actions,
            rewards,
            discounts,
            mixed_prec=True):

        self.world_optimizer.zero_grad()

        context = autocast() if mixed_prec else nullcontext()
        with context:
            out = self.world_model(obs, actions)
            loss, loss_info = self.world_model.loss(out,
                                                    obs,
                                                    rewards,
                                                    discounts)

        if mixed_prec:
            scaler.scale(loss).backward(retain_graph=True)
            scaler.unscale_(self.world_optimizer)
        else:
            loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.world_model.parameters(), get_config().training.grad_clip)

        if mixed_prec:
            scaler.step(self.world_optimizer)
            scaler.update()
        else:
            self.world_optimizer.step()

        with tabular.prefix('world_model_'):
            for k, v in loss_info.items():
                tabular.record(k, v.cpu().item())

        return out

    def _initialize_dataset(self, trainer):
        config = get_config()
        random_policy = RandomPolicy(self.env_spec)
        sampler_type = type(self._sampler)
        random_sampler = sampler_type(
            agents=random_policy,
            envs=trainer._env,
            max_episode_length=self.env_spec.max_episode_length,
            n_workers=config.samplers.init.n)
        initial_episodes = random_sampler.obtain_exact_episodes(
            n_eps_per_worker=config.samplers.init.eps,
            agent_update=random_policy)
        self.buffer.collect(initial_episodes)

        if sampler_type == RaySampler:
            for worker in random_sampler._all_workers.values():
                worker.shutdown.remote()

        del random_sampler

    # def _dynamics_learning(self):
    #     pass

    # def _behaviour_learning(self):
    #     pass

    # Initialize Dataset D with S random seed episodes.

    # Initialize neural network and their params

    # While not converged:
    #   For update step range(C):
    #       == Dynamics Learning ==
    #       Draw B data sequences of length L from Dataset D
    #       Compute next states using model and update theta using rep learning
    #
    #       == Behavior Learning ==
    #       Imagine trajectories from each state of length H
    #       Predict rewards and values
    #       Compute value estimates
    #       Update policy params
    #       Update critic params
    #
    #   Reset Env
    #   for timestep t = 1..T:
    #       model next state from history
    #       compute action from policy
    #       add exploration noise
    #       take step and record r and o
    #   Add experience to dataset
    #
