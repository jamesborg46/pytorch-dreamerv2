import os.path as osp
import time
from contextlib import nullcontext
from typing import Union, Optional

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
import utils
from utils import (RandomPolicy, get_config, log_action_dist_plots, segs_to_batch, log_eps_video,
                   log_reconstructions)

scaler = torch.cuda.amp.GradScaler(enabled=True)


class Dreamer(RLAlgorithm):

    def __init__(self,
                 env_spec: EnvSpec,
                 sampler: Union[LocalSampler, RaySampler],
                 log_sampler: Union[LocalSampler, RaySampler],
                 world_model: models.WorldModel,
                 agent: models.ActorCritic,
                 buf: replay_buffer.ReplayBuffer,
                 human_buf: Optional[replay_buffer.ReplayBuffer] = None,
                 mixed_prec: bool = True,
                 ):

        device = global_device()
        self.env_spec = env_spec
        self._sampler = sampler
        self._log_sampler = log_sampler
        self.world_model = world_model.to(device)
        self.agent = agent.to(device)
        self.buffer = buf
        self.human_buf = human_buf
        self.mixed_prec = mixed_prec
        self.human_portion = get_config().buffer.human_portion

        self.log_freq = get_config().logging.log_freq
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

        if human_buf is not None:
            self.example_human_actions = utils.get_human_actions(human_buf)

    def agent_update(self):
        sampler_type = type(self._sampler)
        if sampler_type == RaySampler:
            return {k: v.cpu() for k, v in self.agent.state_dict().items()}
        elif sampler_type == LocalSampler:
            return {k: v for k, v in self.agent.state_dict().items()}

    def sample_segments(self):
        if get_config().buffer.shared_buffer:
            n = int(self._num_segs_per_batch * self.human_portion)
            k = self._num_segs_per_batch - n
            human_segs = self.human_buf.sample_segments(n=n)
            policy_segs = self.buffer.sample_segments(n=k)
            segs = human_segs + policy_segs
        else:
            segs = self.buffer.sample_segments(n=self._num_segs_per_batch)
        return segs

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

        self._initialize_dataset(trainer)
        self._pretrain()

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
                segs = self.sample_segments()
                obs, actions, rewards, discounts = segs_to_batch(
                    segs, self.env_spec)
                start = time.time()
                wm_out = self.train_world_model_once(
                    obs, actions, rewards, discounts, self.mixed_prec
                )
                print("train time:", time.time() - start)

                self.train_actor_critic_once(wm_out, self.mixed_prec)

                logger.log(tabular)
                logger.dump_all(trainer.step_itr)
                tabular.clear()
                trainer.step_itr += 1

                if (j+1) % get_config().training.target_update_freq == 0:
                    self.agent.update_target_critic()

            if i and i % self.log_freq == 0:
                with torch.no_grad():
                    video_dir = osp.join(trainer._snapshotter.snapshot_dir,
                                         'videos')
                    logger.log('LOGGING')
                    start = time.time()

#                     log_eps_video(
#                         eps=eps,
#                         log_dir=video_dir,
#                         itr=trainer.step_itr
#                     )

#                     log_reconstructions(obs,
#                                         wm_out,
#                                         log_dir=video_dir,
#                                         itr=trainer.step_itr)

                    if self.human_buf is not None:
                        policy_actions = utils.get_policy_actions(eps)
                        log_action_dist_plots(self.example_human_actions,
                                              policy_actions,
                                              itr=trainer.step_itr)

                    logger.log(f"LOG TIME: {time.time() - start}")


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

        with tabular.prefix('actor_critic/'):
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

        with tabular.prefix('world_model/'):
            for k, v in loss_info.items():
                tabular.record(k, v.cpu().item())

        return out

    def _initialize_dataset(self, trainer):
        initial_episodes = self._sampler.obtain_exact_episodes(
            n_eps_per_worker=get_config().samplers.init.eps,
            agent_update=self.agent_update()
        )

        # config = get_config()
        # random_policy = RandomPolicy(self.env_spec)
        # sampler_type = type(self._sampler)
        # random_sampler = sampler_type(
        #     agents=random_policy,
        #     envs=trainer._env,
        #     max_episode_length=self.env_spec.max_episode_length,
        #     n_workers=config.samplers.init.n)
        #

        self.buffer.collect(initial_episodes)

    def _pretrain(self):
        logger.log('PRE TRAINING')
        for j in tqdm(range(get_config().world.pretrain_steps)):
            segs = self.sample_segments()
            obs, actions, rewards, discounts = segs_to_batch(
                segs, self.env_spec)
            self.train_world_model_once(
                obs, actions, rewards, discounts, self.mixed_prec
            )

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
