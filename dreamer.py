from dowel import tabular, logger
from garage import EnvSpec
from garage.np.algos import RLAlgorithm
from utils import RandomPolicy
from math import ceil
from garage.sampler import RaySampler, LocalSampler
from utils import get_config, segs_to_batch, log_episodes, log_reconstructions
import torch
from torch import optim
from garage.torch import global_device
from tqdm import tqdm
import os.path as osp
import time
import numpy as np

scaler = torch.cuda.amp.GradScaler(enabled=True)

class Dreamer(RLAlgorithm):

    def __init__(self,
                 env_spec: EnvSpec,
                 sampler,
                 log_sampler,
                 world_model,
                 agent,
                 buf,
                 ):

        device = global_device()
        self.env_spec = env_spec
        self._sampler = sampler
        self._log_sampler = log_sampler
        self.world_model = world_model.to(device)
        self.agent = agent.to(device)
        self.buffer = buf

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
        self._initialize_dataset(
            trainer,
            seed_episodes=get_config().training.seed_episodes)

        for i in trainer.step_epochs():
            logger.log('COLLECTING')
            start = time.time()
            eps = self._sampler.obtain_exact_episodes(
                n_eps_per_worker=1,
                agent_update={k: v.cpu() for k, v in self.agent.state_dict().items()}
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
                wm_out = self.train_world_model_once(
                    obs, actions, rewards, discounts
                )

                if i >= get_config().world.pretrain:
                    self.train_actor_critic_once(wm_out)

                logger.log(tabular)
                logger.dump_all(trainer.step_itr)
                tabular.clear()
                trainer.step_itr += 1

            self.agent.update_target_critic()

            if i and i % self.log_vid_freq == 0:

                eps = log_episodes(
                    trainer.step_itr,
                    trainer._snapshotter.snapshot_dir,
                    sampler=self._log_sampler,
                    policy=self.agent,
                    enable_render=True,
                )

                log_reconstructions(
                    eps,
                    self.env_spec,
                    self.world_model,
                    trainer.step_itr,
                    path=osp.join(trainer._snapshotter.snapshot_dir, 'videos')
                )

    def train_actor_critic_once(self, wm_out):

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        with torch.cuda.amp.autocast():

            initial_stoch = wm_out['posterior_samples'].reshape(
                -1, get_config().rssm.stoch_state_size,
                get_config().rssm.stoch_state_classes)
            initial_deter = wm_out['deters'].reshape(
                -1, get_config().rssm.det_state_size)

            act_out = self.agent(initial_stoch, initial_deter)

            actor_loss = self.agent.actor_loss(
                act_out['latents'],
                act_out['values'],
                act_out['actions'],
                act_out['targets'],
                act_out['weights'])

            critic_loss = self.agent.critic_loss(
                act_out['latents'], act_out['targets'], act_out['weights']
            )

        scaler.scale(actor_loss).backward(retain_graph=True)
        scaler.scale(critic_loss).backward()
        scaler.unscale_(self.actor_optimizer)
        scaler.unscale_(self.critic_optimizer)

        torch.nn.utils.clip_grad_norm_(
            self.agent.actor.parameters(), get_config().training.grad_clip)
        torch.nn.utils.clip_grad_norm_(
            self.agent.critic.parameters(), get_config().training.grad_clip)
        scaler.step(self.actor_optimizer)
        scaler.step(self.critic_optimizer)
        scaler.update()

        with tabular.prefix('actor_critic_'):
            tabular.record('actor_loss', actor_loss.cpu().item())
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

    def train_world_model_once(self, obs, actions, rewards, discounts):

        self.world_optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            out = self.world_model(obs, actions)
            loss, loss_info = self.world_model.loss(out, obs, rewards, discounts)

        scaler.scale(loss).backward(retain_graph=True)
        scaler.unscale_(self.world_optimizer)

        torch.nn.utils.clip_grad_norm_(
            self.world_model.parameters(), get_config().training.grad_clip)
        scaler.step(self.world_optimizer)
        scaler.update()

        with tabular.prefix('world_model_'):
            for k, v in loss_info.items():
                tabular.record(k, v.cpu().item())

        return out

    def _initialize_dataset(self, trainer, seed_episodes=5):
        random_policy = RandomPolicy(self.env_spec)
        random_sampler = RaySampler(
            agents=random_policy,
            envs=trainer._env,
            max_episode_length=self.env_spec.max_episode_length,
            n_workers=8)
        initial_episodes = random_sampler.obtain_exact_episodes(
            n_eps_per_worker=seed_episodes,
            agent_update=random_policy)
        self.buffer.collect(initial_episodes)

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

