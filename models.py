from enum import Enum, auto
from typing import Callable, Union, Optional, Dict

import akro
import gym
import torch
from dowel import logger
from garage.torch import global_device
from garage.torch.policies import Policy
import numpy as np
from torch import distributions, nn
from torch._six import inf
from torch.distributions import Independent, kl_divergence
from torch.distributions.utils import logits_to_probs
from utils import scale_img, get_dist_mode, check_tensors

EPS = 1e-9


def categorical_kl(logits_p, logits_q):
    probs_p = logits_to_probs(logits_p) + EPS
    probs_q = logits_to_probs(logits_q) + EPS
    t = probs_p * (logits_p - logits_q)
    t[(probs_q == 0).expand_as(t)] = inf
    t[(probs_p == 0).expand_as(t)] = 0
    return t.sum(dim=[-1, -2])


def kl_loss(posterior, prior, config):
    lhs = categorical_kl(posterior.logits.detach(), prior.logits)
    rhs = categorical_kl(posterior.logits, prior.logits.detach())
    kl_loss = config.rssm.alpha * lhs + (1 - config.rssm.alpha) * rhs

    if not torch.isclose(lhs, rhs).all():
        max_dif = torch.max(torch.abs(lhs - rhs)).cpu().item()
        logger._warn(f'LHS and RHS of KL differ by: {max_dif}')

    expected = kl_divergence(
        Independent(posterior, 1),
        Independent(prior, 1),)

    if not torch.isclose(lhs, expected, atol=1e-5).all():
        max_dif = torch.max(torch.abs(lhs - expected)).cpu().item()
        logger._warn(f'LHS and RHS of KL differ by: {max_dif}')

    try:
        check_tensors(posterior.logits,
                      prior.logits,
                      lhs, rhs, kl_loss)
    except Exception as e:
        print(e)
        breakpoint()

    return kl_loss


class World(Enum):
    ATARI = auto()
    DIAMOND = auto()
    BASALT = auto()


class WorldModel(torch.nn.Module):

    def __init__(self,
                 env_spec,
                 config,
                 world_type: World = World.DIAMOND):
        super().__init__()
        self.env_spec = env_spec
        self.config = config
        self.world_type = world_type

        self.latent_state_size = (
            self.config.rssm.stoch_state_classes
            * self.config.rssm.stoch_state_size
            + self.config.rssm.det_state_size
        )

        obs_space = env_spec.observation_space
        if world_type == World.ATARI:
            assert type(obs_space) == akro.Dict and \
                obs_space.spaces.keys() == set(['pov'])
        elif world_type == World.DIAMOND:
            assert type(obs_space) == akro.Dict and \
                obs_space.spaces.keys() == set(['pov', 'vector'])

            self.obs_vector_decoder = MLP(
                input_shape=self.latent_state_size,
                units=[400, 400, 400],
                out_shape=64,
                dist='gaussian')

        elif world_type == World.BASALT:
            raise NotImplementedError()
        else:
            raise ValueError()

        self.rssm = RSSM(env_spec, config=self.config, world_type=world_type)

        self.image_encoder = ImageEncoder(config=self.config)
        self.image_decoder = ImageDecoder(config=self.config)

        self.reward_predictor = MLP(
            input_shape=self.latent_state_size,
            units=self.config.reward_head.units,
            dist='mse')

        self.discount_predictor = MLP(
            input_shape=self.latent_state_size,
            units=self.config.discount_head.units,
            dist='bernoulli')

    def set_action_stats(self, mean, std):
        self.action_mean = torch.tensor(
            mean, device=global_device(), dtype=torch.float)
        self.action_std = torch.tensor(
            std, device=global_device(), dtype=torch.float)

    # def reconstruct(self, observations, actions):
    #     steps, channels, height, width = observations.shape
    #     embedded_observations = self.image_encoder(
    #         observations)
    #     out = self.observe(embedded_observations.unsqueeze(0),
    #                             actions.unsqueeze(0))
    #     latent_states = out['latent_states'].reshape(steps,
    #                                                  self.latent_state_size)
    #     image_recon = self.image_decoder(latent_states).reshape(
    #         steps, channels, height, width)
    #     return image_recon

    def forward(self, observations, actions):
        embedded_observations = self.embed_observations(observations)
        actions = self.embed_actions(actions)
        out = self.observe(embedded_observations, actions)
        out['reward_dist'] = self.reward_predictor(out['latent_states'])
        out['discount_dist'] = self.discount_predictor(out['latent_states'])
        out['recon_obs'] = self.reconstruct_observations(out['latent_states'])
        return out

    def encode_images(self, images) -> torch.Tensor:
        ndim = images.ndim
        images = scale_img(images)
        if ndim == 3:
            channels, height, width = images.shape
            image = images.reshape(1, channels, height, width)
            embedded_images = self.image_encoder(image).flatten()
        elif ndim == 5:
            segs, steps, channels, height, width = images.shape
            flattened_images = images.reshape(
                segs*steps, channels, height, width)
            embedded_images = self.image_encoder(
                flattened_images).reshape(segs, steps, -1)
        else:
            raise ValueError()

        try:
            check_tensors(images, embedded_images)
        except Exception as e:
            print(e)
            breakpoint()

        return embedded_images

    def embed_observations(self, observations):
        if self.world_type == World.ATARI:
            embedded_observations = self.encode_images(observations['pov'])

        elif self.world_type == World.DIAMOND:
            embedded_pov = self.encode_images(observations['pov'])
            embedded_observations = torch.cat(
                [embedded_pov, observations['vector']],
                dim=-1)

        elif self.world_type == World.BASALT:
            raise NotImplementedError()

        else:
            raise ValueError()

        try:
            check_tensors(*observations.values(), embedded_observations)
        except Exception as e:
            print(e)
            breakpoint()

        return embedded_observations


    def embed_actions(self, actions):
        if self.world_type == World.ATARI:
            pass
        elif self.world_type == World.DIAMOND:
            if self.config.training.scale_action_dist:
                actions = (
                    (actions['vector'] - self.action_mean) / self.action_std
                )
            else:
                actions = actions['vector']

        elif self.world_type == World.BASALT:
            raise NotImplementedError()

        else:
            raise ValueError()

        try:
            check_tensors(actions)
        except Exception as e:
            print(e)
            breakpoint()

        return actions

    def reconstruct_observations(self, latent_states: torch.Tensor) \
            -> Dict[str, torch.distributions.Distribution]:
        steps = self.config.training.seg_length
        segs = self.config.training.num_segs_per_batch
        height = self.config.image.height
        width = self.config.image.width
        channels = self.config.image.color_channels
        recon_obs = dict()

        flattened_latent_states = latent_states.reshape(segs*steps,
                                                        self.latent_state_size)
        image_mean = self.image_decoder(flattened_latent_states).reshape(
            segs, steps, channels, height, width)
        norm = distributions.Normal(loc=image_mean, scale=1)
        image_recon_dist = distributions.Independent(norm, 3)
        assert image_recon_dist.batch_shape == (segs, steps) and \
            image_recon_dist.event_shape == (channels, height, width)
        recon_obs['image_recon_dist'] = image_recon_dist

        if self.world_type == World.DIAMOND:
            vector_recon_dist = self.obs_vector_decoder(
                latent_states)
            vector_recon_dist = distributions.Independent(vector_recon_dist, 1)
            assert vector_recon_dist.batch_shape == (segs, steps) and \
                vector_recon_dist.event_shape == (64, )
            recon_obs['vector_recon_dist'] = vector_recon_dist

        try:
            check_tensors(latent_states)
        except Exception as e:
            print(e)
            breakpoint()

        return recon_obs

    def imagine(self,
                initial_stoch,
                initial_deter,
                policy=None,
                horizon=None,
                actions=None):
        """
        Rollout out imagination according to given policy OR given actions
        """

        assert (policy is None and horizon is None) or actions is None

        stoch, deter = initial_stoch, initial_deter
        latent_state = self.get_latent_state(stoch, deter)

        if policy is not None:
            actions = []
        elif actions is not None:
            horizon = len(actions)

        latents = []

        for i in range(horizon):
            latents.append(latent_state)
            if policy is not None:
                action = policy(latent_state.detach()).rsample()
                actions.append(action)
            else:
                action = actions[i]
            prior, deter = self.rssm.imagine_step(stoch, deter, action)
            stoch = prior.rsample()
            latent_state = self.get_latent_state(stoch, deter)

        if policy is not None:
            actions = torch.stack(actions)

        latents = torch.stack(latents)
        rewards = self.reward_predictor(latents).mean
        discounts = self.discount_predictor(latents).mean

        try:
            check_tensors(initial_stoch,
                          initial_deter,
                          latents, rewards, discounts, actions
                          )
        except Exception as e:
            print(e)
            breakpoint()

        return actions, latents, rewards, discounts

    def observe(self,
                embedded_observations: torch.Tensor,
                actions: torch.Tensor) -> Dict[str, Union[torch.Tensor, Dict]]:
        segs, steps, _ = embedded_observations.shape
        assert segs == actions.shape[0]
        assert steps == actions.shape[1]

        swap = lambda t: torch.swapaxes(t, 0, 1)  # noqa: E731

        # Change from SEGS x STEPS x N -> STEPS x SEGS x N
        # This facilitates
        embedded_observations = swap(embedded_observations)
        actions = swap(actions)

        initial = self.rssm.initial_state(batch_size=segs)
        stoch, deter = initial['stoch'], initial['deter']

        posterior_samples = []
        prior_samples = []
        deters = []
        # latent_states = []
        kl_losses = []

        for embed, action in zip(embedded_observations, actions):
            prior, deter = self.rssm.imagine_step(stoch, deter, action)
            posterior = self.rssm.observe_step(deter, embed)
            stoch = posterior.rsample()

            posterior_samples.append(stoch)
            prior_samples.append(prior.rsample())
            deters.append(deter)
            # latent_states.append(
            #     torch.cat([stoch.flatten(start_dim=1), deter], dim=-1))
            kl_losses.append(kl_loss(posterior, prior, self.config))

        out = {
            'posterior_samples': swap(torch.stack(posterior_samples)),
            'prior_samples': swap(torch.stack(prior_samples)),
            'deters': swap(torch.stack(deters)),
            'kl_losses': swap(torch.stack(kl_losses))
        }
        out['latent_states'] = self.get_latent_state(
            out['posterior_samples'], out['deters'])

        try:
            check_tensors(embedded_observations,
                          actions,
                          *out.values()
                          )
        except Exception as e:
            print(e)
            breakpoint()

        return out

    def get_latent_state(self, stoch, deter):
        latent_state = torch.cat(
            [stoch.flatten(start_dim=-2), deter],
            dim=-1
        )
        return latent_state

    def loss(self, out, observation_batch, reward_batch, discount_batch):
        loss_info = dict()
        kl_loss = out['kl_losses'].mean()
        reward_loss = -out['reward_dist'].log_prob(reward_batch).mean()
        discount_loss = -out['discount_dist'].log_prob(discount_batch).mean()

        recon_loss = (
            -out['recon_obs']['image_recon_dist']
            .log_prob(scale_img(observation_batch['pov'])).mean()
        )

        if self.world_type == World.DIAMOND:
            vector_scale = self.config.loss_scales.recon_vector
            image_recon_loss = recon_loss
            vector_recon_loss = (
                vector_scale * -out['recon_obs']['vector_recon_dist']
                .log_prob(observation_batch['vector']).mean()
            )
            recon_loss = image_recon_loss + vector_recon_loss

            loss_info['image_recon_loss'] = image_recon_loss
            loss_info['vector_recon_loss'] = vector_recon_loss

        reward_mae = torch.abs(out['reward_dist'].mean - reward_batch).mean()
        discount_mae = torch.abs(out['discount_dist'].mean
                                 - discount_batch).mean()

        loss = (
            self.config.loss_scales.reward * reward_loss +
            self.config.loss_scales.discount * discount_loss +
            self.config.loss_scales.recon * recon_loss +
            self.config.loss_scales.kl * kl_loss
        )

        loss_info.update({
            'kl_loss': kl_loss,
            'reward_loss': reward_loss,
            'discount_loss': discount_loss,
            'recon_loss': recon_loss,
            'total_loss': loss,
            'reward_mae': reward_mae,
            'discount_mae': discount_mae,
        })

        try:
            check_tensors(*observation_batch.values(),
                          reward_batch,
                          discount_batch,
                          *loss_info.values()
                          )
        except Exception as e:
            print(e)
            breakpoint()

        return loss, loss_info


class ActorCritic(Policy):

    def __init__(self,
                 env_spec,
                 world_model,
                 config,
                 n_envs=1,
                 random=False,
                 world_type: World = World.DIAMOND):
        super().__init__(env_spec=env_spec, name='ActorCritic')
        self.world_model = world_model
        self.config = config
        self.n_envs = n_envs
        self.random = random
        self.world_type = world_type

        self.latent_state_size = (
            self.config.rssm.stoch_state_classes
            * self.config.rssm.stoch_state_size
            + self.config.rssm.det_state_size
        )

        obs_space = env_spec.observation_space
        if world_type == World.ATARI:
            assert type(obs_space) == akro.Dict and \
                type(env_spec.action_space) == akro.Discrete
            self.action_size = env_spec.action_space.n
            actor_dist = 'onehot'
            ind_dims = None
            scale = None

        elif world_type == World.DIAMOND:
            assert type(obs_space) == akro.Dict and \
                type(env_spec.action_space) == akro.Dict
            self.action_size = env_spec.action_space.flat_dim
            actor_dist = 'gaussian'
            ind_dims = 1
            scale = None

        elif world_type == World.BASALT:
            raise NotImplementedError()
        else:
            raise ValueError()

        self.actor = MLP(
            input_shape=self.latent_state_size,
            out_shape=self.action_size,
            units=self.config.actor.units,
            dist=actor_dist,
            ind_dims=ind_dims,
            scale=scale)

        self.critic = MLP(
            input_shape=self.latent_state_size,
            units=self.config.critic.units,
            dist='mse'
        )

        self.target_critic = MLP(
            input_shape=self.latent_state_size,
            units=self.config.critic.units,
            dist='mse'
        )

        self.deter = None  # initialised in self.reset()
        self.mode = 'train'

    def eval(self):
        self.mode = 'eval'

    def train(self):
        self.mode = 'train'

    @property
    def initial_state(self):
        return self.world_model.rssm.initial_state(self.n_envs)

    def pack_action(self, action):
        if self.world_type == World.ATARI:
            return torch.argmax(action, dim=-1).item()
        elif self.world_type == World.DIAMOND:
            if self.config.training.scale_action_dist:
                action = (
                    action * self.world_model.action_std
                    + self.world_model.action_mean
                )
            return {'vector': np.clip(action.cpu().numpy(),
                                      a_min=-1.05,
                                      a_max=1.05)}
        elif self.world_type == World.BASALT:
            raise NotImplementedError()
        else:
            raise ValueError()

    def get_action(self, observation):
        # if self.random:
        #     action_space = self.action_size
        #     dist = torch.distributions.Categorical(
        #         probs=torch.ones(action_space.n) / action_space.n
        #     )
        #     return dist.sample(), {}

        with torch.no_grad():
            observation = {
                k: torch.tensor(v, device=global_device(), dtype=torch.float)
                for k, v in observation.items()
            }

            embedded_obs = (
                self.world_model.embed_observations(observation)
                    .unsqueeze(0)  # adding batch dimensions
            )

            posterior = self.world_model.rssm.observe_step(
                deter=self.deter, embed=embedded_obs)

            stoch = posterior.rsample()

            latent_state = self.world_model.get_latent_state(
                stoch, self.deter)

            if self.mode == 'train':
                action = self.actor(latent_state).rsample().unsqueeze(0)
            elif self.mode == 'eval':
                action = get_dist_mode(self.actor(latent_state)).unsqueeze(0)
            else:
                raise ValueError

            _, deter = self.world_model.rssm.imagine_step(
                stoch, self.deter, action)

            self.deter = deter

        try:
            check_tensors(*observation.values(),
                          action,
                          )
        except Exception as e:
            print(e)
            breakpoint()

        return self.pack_action(action[0]), {}

    def get_actions(self, observations):
        pass

    def reset(self):
        self.deter = self.initial_state['deter']

    def forward(self, initial_stoch, initial_deter):
        actions, latents, rewards, discounts = self.world_model.imagine(
            initial_stoch=initial_stoch,
            initial_deter=initial_deter,
            policy=self.actor,
            horizon=self.config.critic.imag_horizon
            )

        weights = torch.cumprod(
            torch.cat([torch.ones_like(discounts[:1]), discounts[1:]]),
            dim=0).detach()

        critic_dist = self.critic(latents)
        values = critic_dist.mean
        values_t = self.target_critic(latents).mean
        targets = self.lambda_return(
            values_t, rewards, discounts, lam=self.config.critic.lam)

        out = {
            'actions': actions,
            'latents': latents,
            'rewards': rewards,
            'discounts': discounts,
            'critic_dist': critic_dist,
            'values': values,
            'values_t': values_t,
            'targets': targets,
            'weights': weights,
        }

        try:
            check_tensors(initial_stoch,
                          initial_deter,
                          actions,
                          latents,
                          rewards,
                          discounts,
                          values,
                          values_t,
                          targets,
                          weights)
        except Exception as e:
            print(e)
            breakpoint()

        return out

    def lambda_return(self, values_t, rewards, discounts, lam):
        horizon, rollouts = values_t.shape
        assert rewards.shape == (horizon, rollouts)
        assert discounts.shape == (horizon, rollouts)

        H = horizon
        prev_return = values_t[H-1]
        returns_reversed = [prev_return]

        for i in range(1, horizon):
            lambda_return = (
                rewards[H-i-1] +
                discounts[H-i-1] * (
                    (1-lam)*values_t[H-i] + lam * prev_return
                )
            )
            returns_reversed.append(lambda_return)
            prev_return = lambda_return

        lambda_returns = torch.flip(torch.stack(returns_reversed), dims=[0])

        try:
            check_tensors(values_t,
                          rewards,
                          discounts,
                          lambda_returns)
        except Exception as e:
            print(e)
            breakpoint()

        return lambda_returns

    def supervised_actor_loss(self, latents, actions):
        policy = self.actor(latents[:, :-1].detach())
        supervised_actor_loss = -policy.log_prob(actions[:, 1:].detach()).mean()

        try:
            check_tensors(latents,
                          actions,
                          supervised_actor_loss)
        except Exception as e:
            print(e)
            breakpoint()

        return supervised_actor_loss

    def actor_loss(self, latents, values, action, target, weights):
        policy = self.actor(latents.detach())

        if self.config.actor.grad == 'reinforce':
            baseline = values
            advantage = (target - baseline).detach()
            objective = policy.log_prob(action) * advantage
        elif self.config.actor.grad == 'dynamics':
            objective = target
        else:
            raise ValueError()

        ent_scale = self.config.actor.ent_scale
        objective += ent_scale * policy.entropy()
        actor_loss = -(objective * weights)[:-1].mean()

        try:
            check_tensors(latents,
                          values,
                          action,
                          target,
                          weights,
                          actor_loss)
        except Exception as e:
            print(e)
            breakpoint()

        return actor_loss, policy.entropy()

    def critic_loss(self, latents, targets, weights):
        dist = self.critic(latents.detach())
        critic_loss = -(dist.log_prob(targets.detach())
                        * weights.detach())[:-1].mean()

        try:
            check_tensors(latents,
                          targets,
                          weights,
                          critic_loss)
        except Exception as e:
            print(e)
            breakpoint()

        return critic_loss

    def update_target_critic(self):
        self.target_critic.load_state_dict(self.critic.state_dict())


class RSSM(torch.nn.Module):

    def __init__(self,
                 env_spec,
                 config,
                 world_type: World = World.DIAMOND):
        super().__init__()
        self._env_spec = env_spec
        self.config = config

        if type(env_spec.action_space) == akro.Dict:
            self.action_size = env_spec.action_space.flat_dim
        elif type(env_spec.action_space) == akro.Discrete:
            self.action_size = env_spec.action_space.n
        else:
            raise ValueError()

        obs_space = env_spec.observation_space
        if world_type == World.ATARI:
            self.embed_size = self.config.image_encoder.N * 32
        elif world_type == World.DIAMOND:
            self.embed_size = self.config.image_encoder.N * 32 + 64
        elif world_type == World.BASALT:
            raise NotImplementedError()
        else:
            raise ValueError()

        self.stoch_state_classes = self.config.rssm.stoch_state_classes
        self.stoch_state_size = self.config.rssm.stoch_state_size
        self.det_state_size = self.config.rssm.det_state_size
        self.act = eval(self.config.rssm.act)

        self.register_parameter(
            name='cell_initial_state',
            param=nn.Parameter(torch.zeros(self.det_state_size,
                                           device=global_device()))
        )

        self.cell = nn.GRUCell(input_size=self.det_state_size,
                               hidden_size=self.det_state_size)

        self._initialize_imagination_layers()
        self._initialize_observation_layers()

    def _initialize_imagination_layers(self):
        self.embed_stoch_state_and_action = nn.Linear(
            self.action_size + self.stoch_state_size * self.stoch_state_classes,
            self.det_state_size)

        self.imagine_out_1 = nn.Linear(self.det_state_size, self.det_state_size)
        self.imagine_out_2 = nn.Linear(
            self.det_state_size,
            self.stoch_state_classes*self.stoch_state_size)

    def _initialize_observation_layers(self):
        self.observe_out_1 = nn.Linear(
            self.det_state_size + self.embed_size,
            self.det_state_size)
        self.observe_out_2 = nn.Linear(
            self.det_state_size,
            self.stoch_state_classes*self.stoch_state_size)

    def initial_state(self, batch_size):
        state = {
            'logits': torch.zeros(batch_size,
                                  self.stoch_state_size,
                                  self.stoch_state_classes,
                                  device=global_device()),
            'stoch': torch.zeros(batch_size,
                                 self.stoch_state_size,
                                 self.stoch_state_classes,
                                 device=global_device()),
            'deter': self.cell_initial_state.repeat([batch_size, 1])
        }
        return state

    def step(self, prev_stoch, prev_deter, prev_action):
        x = torch.cat((prev_stoch.flatten(start_dim=1), prev_action), dim=-1)
        x = self.act(self.embed_stoch_state_and_action(x))
        deter = self.cell(x, prev_deter)
        return deter

    def imagine_step(self, prev_stoch, prev_deter, prev_action):
        deter = self.step(prev_stoch, prev_deter, prev_action)
        x = self.act(self.imagine_out_1(deter))
        x = self.imagine_out_2(x)
        logits = x.reshape(
            *x.shape[:-1],
            self.stoch_state_size,
            self.stoch_state_classes
        )
        prior = torch.distributions.OneHotCategoricalStraightThrough(
            logits=logits)
        return prior, deter

    def observe_step(self, deter, embed):
        x = torch.cat([deter, embed], dim=-1)
        x = self.act(self.observe_out_1(x))
        x = self.observe_out_2(x)
        logits = x.reshape(
            *x.shape[:-1],
            self.stoch_state_size,
            self.stoch_state_classes
        )
        posterior = torch.distributions.OneHotCategoricalStraightThrough(
            logits=logits)
        return posterior


class ImageEncoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        Activation = eval(self.config.image_encoder.Activation)
        self.N = N = self.config.image_encoder.N
        color_channels = self.config.image.color_channels

        self.model = nn.Sequential(
            nn.Conv2d(color_channels, N*1, 4, 2),
            Activation(),
            nn.Conv2d(N*1, N*2, 4, 2),
            Activation(),
            nn.Conv2d(N*2, N*4, 4, 2),
            Activation(),
            nn.Conv2d(N*4, N*8, 4, 2),
            Activation(),
        )

    def forward(self, img):
        x = self.model(img)
        return torch.flatten(x, start_dim=1)


class ImageDecoder(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        Activation = eval(self.config.image_decoder.Activation)
        self.N = N = self.config.image_decoder.N
        self.shape = [
            self.config.image.height,
            self.config.image.width,
            self.config.image.color_channels
        ]
        latent_state_shape = (
            self.config.rssm.stoch_state_classes * self.config.rssm.stoch_state_size
            + self.config.rssm.det_state_size)

        self.dense = nn.Linear(latent_state_shape, N*32)
        self.deconvolve = nn.Sequential(
            nn.ConvTranspose2d(N*32, N*4, 5, 2),
            Activation(),
            nn.ConvTranspose2d(N*4, N*2, 5, 2),
            Activation(),
            nn.ConvTranspose2d(N*2, N, 6, 2),
            Activation(),
            nn.ConvTranspose2d(N, self.shape[-1], 6, 2),
            # nn.Sigmoid(),  # TODO: Check this
        )

    def forward(self, embed):
        x = self.dense(embed).reshape(-1, self.N*32, 1, 1)
        x = self.deconvolve(x)
        return x


class MLP(torch.nn.Module):

    def __init__(
            self,
            input_shape,
            units,
            out_shape=1,
            dist: Union[str, Callable] = 'mse',
            scale: Optional[float] = 1.,
            ind_dims: Optional[int] = None,
            Activation=torch.nn.ELU):
        super().__init__()
        self.dist = dist
        self.scale = scale
        self.ind_dims = ind_dims
        self.out_shape = out_shape

        self.net = nn.Sequential()
        for i, unit in enumerate(units):
            self.net.add_module(f"linear_{i}", nn.Linear(input_shape, unit))
            self.net.add_module(f"activation_{i}", Activation())
            input_shape = unit

        if dist == 'gaussian':
            out_shape *= 2

        self.net.add_module("out_layer", nn.Linear(input_shape, out_shape))

    def forward(self, features):
        logits = self.net(features).squeeze()
        if self.dist == 'mse':
            dist = torch.distributions.Normal(loc=logits, scale=self.scale)
        elif self.dist == 'bernoulli':
            dist = torch.distributions.Bernoulli(logits=logits)
        elif self.dist == 'onehot':
            dist = torch.distributions.OneHotCategoricalStraightThrough(
                logits=logits)
        elif self.dist == 'gaussian':
            mean = logits[..., :self.out_shape]
            std = torch.log(1 + torch.exp(logits[..., self.out_shape:]))
            try:
                dist = torch.distributions.Normal(loc=mean, scale=std+0.01)
            except Exception as e:
                print(e)
                print(std)
                print(std.max())
                breakpoint()
        else:
            raise ValueError()

        if self.ind_dims is not None:
            dist = torch.distributions.Independent(
                dist, reinterpreted_batch_ndims=self.ind_dims)

        return dist

    def initialize_mean_std(self, mean, std):
        rho = torch.log(torch.exp(torch.tensor(std, dtype=torch.float32)) - 1)
        mean = torch.tensor(mean, dtype=torch.float32)
        init = torch.cat([mean, rho])
        with torch.no_grad():
            self.net.out_layer.bias.data = torch.nn.Parameter(init)
