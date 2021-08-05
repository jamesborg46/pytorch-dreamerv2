import abc
from garage import EpisodeBatch, TimeStepBatch
from garage.np import slice_nested_dict
import numpy as np
from dowel import tabular


def slice_from_episode(episode, start, end):
    assert len(episode.lengths) == 1
    assert len(episode.last_observations) == 1

    # If the start,end range includes the final episode step we need to manually
    # append the last observation to next_observations
    next_observations = episode.observations[start+1:end+1]
    if len(next_observations) != (end - start):
        next_observations = np.concatenate(
            [next_observations, episode.last_observations],
            axis=0)

    try:
        sliced = TimeStepBatch(
            env_spec=episode.env_spec,
            episode_infos=slice_nested_dict(episode.agent_infos, 0, 1),
            observations=episode.observations[start:end],
            actions=episode.actions[start:end],
            rewards=episode.rewards[start:end],
            next_observations=next_observations,
            env_infos=slice_nested_dict(episode.env_infos, start, end),
            agent_infos=slice_nested_dict(episode.agent_infos, start, end),
            step_types=episode.step_types[start:end],
        )
    except AttributeError as e:
        print(e)
        breakpoint()

    return sliced


class ReplayBuffer(abc.ABC):

    def __init__(self,
                 env_spec,
                 segment_length,
                 max_capacity=500000,
                 flatten=True):

        self.env_spec = env_spec
        self.max_capacity = max_capacity
        self.episodes = None
        self._total_steps = 0
        self._segment_length = segment_length
        self._flatten = flatten

    def collect(self, eps):
        """
        Args:
            paths: as returned from EpisodeBatch.to_list()
        """

        self._total_steps += len(eps.observations)
        self._extend(eps)
        assert self._total_steps == len(self.episodes.observations)

        if self._total_steps > self.max_capacity:
            self._reduce()

    def _extend(self, episodes):
        if self.episodes is None:
            self.episodes = episodes
        else:
            self.episodes = EpisodeBatch.concatenate(
                self.episodes,
                episodes
            )

    def _reduce(self):
        end = len(self.episodes.observations)
        popped_length = 0
        n = len(self.episodes.lengths)
        i = 0

        while self._total_steps - popped_length > self.max_capacity:
            popped_length += self.episodes.lengths[i]
            i += 1

        popped = EpisodeBatch(
            env_spec=self.episodes.env_spec,
            episode_infos=slice_nested_dict(self.episodes.episode_infos, 0, i),  # noqa E:501
            observations=self.episodes.observations[:popped_length],
            last_observations=self.episodes.last_observations[:i],
            actions=self.episodes.actions[:popped_length],
            rewards=self.episodes.rewards[:popped_length],
            env_infos=slice_nested_dict(self.episodes.env_infos, 0, popped_length),  # noqa E:501
            agent_infos=slice_nested_dict(self.episodes.agent_infos, 0, popped_length),  # noqa E:501
            step_types=self.episodes.step_types[:popped_length],
            lengths=self.episodes.lengths[:i]
        )

        self.episodes = EpisodeBatch(
            env_spec=self.episodes.env_spec,
            episode_infos=slice_nested_dict(self.episodes.episode_infos, i, n),  # noqa E:501
            observations=self.episodes.observations[popped_length:],
            last_observations=self.episodes.last_observations[i:],
            actions=self.episodes.actions[popped_length:],
            rewards=self.episodes.rewards[popped_length:],
            env_infos=slice_nested_dict(self.episodes.env_infos, popped_length, end),  # noqa E:501
            agent_infos=slice_nested_dict(self.episodes.agent_infos, popped_length, end),  # noqa E:501
            step_types=self.episodes.step_types[popped_length:],
            lengths=self.episodes.lengths[i:],
        )

        self._total_steps -= popped_length
        assert self._total_steps == len(self.episodes.observations)

        return popped

    @property
    def buffer_full(self):
        max_episode_length = self.env_spec.max_episode_length
        if self._total_steps >= (self.max_capacity - max_episode_length):
            return True
        else:
            return False

    def sample_episodes(self, n=1, lengths_weighted=True):
        # Selecting an episode weighted by its length
        num_episodes = len(self.episodes.lengths)
        if lengths_weighted:
            weights = self.episodes.lengths / self.episodes.lengths.sum()
            idxs = np.random.choice(num_episodes, size=n, p=weights)
        # Selecting an episode at random
        else:
            idxs = np.random.choice(num_episodes, size=n)
        episodes = self.episodes.split()
        sampled_episodes = [episodes[i] for i in idxs]
        return sampled_episodes

    def sample_segments(self, n=1):
        episodes = self.sample_episodes(n=n)
        segments = []
        for ep in episodes:
            length = ep.lengths[0]
            if length < self._segment_length:
                raise Exception('Sampled epiosde shorter than required segment'
                                'length')

            # start = np.random.randint(length - self._segment_length + 1)
            # The below sampling method ensures we get enough terminal steps
            # when sampling
            start = min(np.random.randint(length),
                        length-self._segment_length)
            end = start + self._segment_length
            segments.append(slice_from_episode(ep, start, end))
        return segments

    def _get_model_input_from_segment(self, segment):
        pass

    def log_stats(self, itr):
        with tabular.prefix('Buffer/'):
            tabular.record('BufferSize', self._total_steps)
