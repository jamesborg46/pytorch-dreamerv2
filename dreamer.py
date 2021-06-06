from dowel import tabular, logger
from garage import EnvSpec
from garage.np.algos import RLAlgorithm
from utils import RandomPolicy
from math import ceil
from garage.sampler import LocalSampler
from utils import CONFIG, segs_to_batch
from torch import optim
from garage.torch import global_device
from tqdm import tqdm


class Dreamer(RLAlgorithm):

    def __init__(self,
                 env_spec: EnvSpec,
                 sampler,
                 world_model,
                 agent,
                 buf,
                 ):

        device = global_device()
        self.env_spec = env_spec
        self._sampler = sampler
        self.world_model = world_model.to(device)
        self.agent = agent
        self.buffer = buf

        self._num_segs_per_batch = CONFIG.training.num_segs_per_batch
        self._num_training_steps = CONFIG.training.num_training_steps
        self.optimizer = optim.Adam(self.world_model.parameters(),
                                    lr=CONFIG.training.lr)

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

        for i in trainer.step_epochs():
            logger.log('COLLECTING')
            eps = self._sampler.obtain_exact_episodes(
                n_eps_per_worker=1,
                agent_update=self.agent,
            )
            self.buffer.collect(eps)

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
                logger.dump_all(i*self._num_training_steps+j)
                tabular.clear()

                # Each should be SEGS x STEPS x ... in shape

                # Model learning
                # Update rssm_model params
                #
                # Behaviour learning

    def train_world_model_once(self, obs, actions, rewards, discounts):

        self.optimizer.zero_grad()
        out = self.world_model(obs, actions)
        loss, loss_info = self.world_model.loss(out, obs, rewards, discounts)
        loss.backward()
        self.optimizer.step()

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

