from garage.np.algos import RLAlgorithm


class Dreamer(RLAlgorithm):


    def __init__(self,
                 rssm_model,
                 actor,
                 critic,
                 buf):

        self.rssm_model = rssm_model
        self.actor = actor
        self.critic = critic
        self.buffer = buf

    def train(self, trainer):
        """Obtain samplers and start actual training for each epoch.

        Args:
            trainer (Trainer): Gives the algorithm the access to
                :method:`~Trainer.step_epochs()`, which provides services
                such as snapshotting and sampler control.

        Returns:
            float: The average return in last epoch cycle.

        """

    def _initialize_dataset(self, trainer):
        initial_episodes = trainer.obtain_episodes(
            trainer.step_itr, batch_size)
        self.buffer.add_episodes(initial_episodes)

    def _dynamics_learning(self):
        pass

    def _behaviour_learning(self):
        pass

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

