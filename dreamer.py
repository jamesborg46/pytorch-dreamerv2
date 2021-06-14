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
        self.agent = agent
        self.buffer = buf

        self.log_vid_freq = get_config().logging.log_vid_freq
        self._num_segs_per_batch = get_config().training.num_segs_per_batch
        self._num_training_steps = get_config().training.num_training_steps
        self.world_optimizer = optim.Adam(self.world_model.parameters(),
                                          lr=get_config().training.lr)

        # self.critic_optimizer = optim.Adam()
        # self.actor_optimizer = optim.Adam()

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

            logger.log('TRAINING')
            for j in tqdm(range(self._num_training_steps)):
                segs = self.buffer.sample_segments(
                    n=self._num_segs_per_batch)
                obs, actions, rewards, discounts = segs_to_batch(
                    segs, self.env_spec)
                self.train_world_model_once(
                    obs, actions, rewards, discounts
                )

                logger.log(tabular)
                logger.dump_all(trainer.step_itr)
                tabular.clear()
                trainer.step_itr += 1

                # Each should be SEGS x STEPS x ... in shape

                # Model learning
                # Update rssm_model params
                #
                # Behaviour learning
                #

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

    def train_world_model_once(self, obs, actions, rewards, discounts):

        self.world_optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            out = self.world_model(obs, actions)
            loss, loss_info = self.world_model.loss(out, obs, rewards, discounts)

        scaler.scale(loss).backward()
        scaler.unscale_(self.world_optimizer)

        # loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.world_model.parameters(), get_config().training.grad_clip)
        scaler.step(self.world_optimizer)
        scaler.update()
        # self.world_optimizer.step()

        with tabular.prefix('world_model'):
            for k, v in loss_info.items():
                tabular.record(k, v)

    def _initialize_dataset(self, trainer, seed_episodes=5):
        random_policy = RandomPolicy(self.env_spec)
        random_sampler = LocalSampler(
            agents=random_policy,
            envs=trainer._env,
            max_episode_length=self.env_spec.max_episode_length,
            n_workers=1)
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

