import torch
import torch.nn.functional as F
from torch import nn
from torch import distributions
from torch.distributions import kl_divergence, Independent
from torch.distributions.utils import logits_to_probs
from torch._six import inf
import torch.nn.functional as F
from garage.torch import global_device
from garage.torch.policies import Policy
from utils import get_config
from dowel import logger

# def categorical_kl(probs_a, probs_b):
#     return torch.sum(probs_a * torch.log(probs_a / probs_b), dim=[-1, -2])


def categorical_kl(logits_p, logits_q):
    probs_p = logits_to_probs(logits_p)
    probs_q = logits_to_probs(logits_q)
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

    return kl_loss


class StraightThroughOneHotDist(torch.distributions.OneHotCategorical):

    def rsample(self, sample_shape=torch.Size()):
        sample = self.sample(sample_shape).type(torch.float)
        sample += self.probs - self.probs.detach()  # Straight through gradients trick
        return sample


class WorldModel(torch.nn.Module):

    def __init__(self, env_spec, config):
        super().__init__()
        self.env_spec = env_spec
        self.config = config

        self.rssm = RSSM(env_spec, config=self.config)
        self.image_encoder = ImageEncoder(config=self.config)
        self.image_decoder = ImageDecoder(config=self.config)

        self.latent_state_size = (
            self.config.rssm.stoch_state_classes * self.config.rssm.stoch_state_size
            + self.config.rssm.det_state_size
        )

        self.reward_predictor = MLP(
            input_shape=self.latent_state_size,
            units=self.config.reward_head.units,
            dist='mse')

        self.discount_predictor = MLP(
            input_shape=self.latent_state_size,
            units=self.config.discount_head.units,
            dist='bernoulli')

    def reconstruct(self, observations, actions):
        steps, channels, height, width = observations.shape
        embedded_observations = self.image_encoder(
            observations)
        out = self.observe(embedded_observations.unsqueeze(0),
                                actions.unsqueeze(0))
        latent_states = out['latent_states'].reshape(steps,
                                                     self.latent_state_size)
        image_recon = self.image_decoder(latent_states).reshape(
            steps, channels, height, width)
        return image_recon

    def forward(self, observations, actions):
        segs, steps, channels, height, width = observations.shape
        flattened_observations = observations.reshape(
            segs*steps, channels, height, width)
        embedded_observations = self.image_encoder(
            flattened_observations).reshape(segs, steps, -1)
        out = self.observe(embedded_observations, actions)
        out['reward_dist'] = self.reward_predictor(out['latent_states'])
        out['discount_dist'] = self.discount_predictor(out['latent_states'])
        flattened_latent_states = (
            out['latent_states'].reshape(segs*steps, self.latent_state_size))
        mean = self.image_decoder(flattened_latent_states).reshape(
            segs, steps, channels, height, width)
        norm = distributions.Normal(loc=mean, scale=1)
        image_recon_dist = distributions.Independent(norm, 3)
        assert image_recon_dist.batch_shape == (segs, steps)
        out['image_recon_dist'] = image_recon_dist
        return out

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

        return actions, latents, rewards, discounts

    def observe(self, embedded_observations, actions):
        segs, steps, embedding_size = embedded_observations.shape
        assert segs == actions.shape[0]
        assert steps == actions.shape[1]

        swap = lambda t: torch.swapaxes(t, 0, 1)

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

        return out

    def get_latent_state(self, stoch, deter):
        latent_state = torch.cat(
            [stoch.flatten(start_dim=-2), deter],
            dim=-1
        )
        return latent_state

    def loss(self, out, observation_batch, reward_batch, discount_batch):
        kl_loss = out['kl_losses'].mean()
        reward_loss = -out['reward_dist'].log_prob(reward_batch).mean()
        discount_loss = -out['discount_dist'].log_prob(discount_batch).mean()
        recon_loss = -out['image_recon_dist'].log_prob(observation_batch).mean()

        reward_mae = torch.abs(out['reward_dist'].mean - reward_batch).mean()
        discount_mae = torch.abs(out['discount_dist'].mean - discount_batch).mean()

        loss = (
            self.config.loss_scales.reward * reward_loss +
            self.config.loss_scales.discount * discount_loss +
            self.config.loss_scales.recon * recon_loss +
            self.config.loss_scales.kl * kl_loss
        )

        loss_info = {
            'kl_loss': kl_loss,
            'reward_loss': reward_loss,
            'discount_loss': discount_loss,
            'recon_loss': recon_loss,
            'total_loss': loss,
            'reward_mae': reward_mae,
            'discount_mae': discount_mae,
        }
        return loss, loss_info


class ActorCritic(Policy):

    def __init__(self,
                 env_spec,
                 world_model,
                 config,
                 n_envs=1,
                 random=False):
        super().__init__(env_spec=env_spec, name='ActorCritic')
        self.world_model = world_model
        self.config = config
        self.random = random

        self.latent_state_size = (
            self.config.rssm.stoch_state_classes * self.config.rssm.stoch_state_size
            + self.config.rssm.det_state_size
        )

        self.actor = MLP(
            input_shape=self.latent_state_size,
            out_shape=env_spec.action_space.n,
            units=self.config.actor.units,
            dist='onehot')

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

        self.initial_state = self.world_model.rssm.initial_state(n_envs)
        self.deter = self.initial_state['deter']

    def get_action(self, observation):
        if self.random:
            action_space = self._env_spec.action_space
            dist = torch.distributions.Categorical(
                probs=torch.ones(action_space.n) / action_space.n
            )
            return dist.sample(), {}

        with torch.no_grad():
            obs = torch.tensor(
                observation, device=global_device(), dtype=torch.float)

            obs = obs.unsqueeze(0)  # Adding batch dimension
            obs = obs / 255 - 0.5

            if self.config.image.color_channels == 1:
                obs = obs.unsqueeze(1)  # Adding color channel

            embedded_obs = self.world_model.image_encoder(obs)

            posterior = self.world_model.rssm.observe_step(
                deter=self.deter, embed=embedded_obs)

            stoch = posterior.rsample()

            latent_state = self.world_model.get_latent_state(
                stoch, self.deter)

            action = self.actor(latent_state).rsample().unsqueeze(0)

            prior, deter = self.world_model.rssm.imagine_step(
                stoch, self.deter, action)

            self.deter = deter

        return torch.argmax(action[0]).item(), {}

    def get_actions(self, observations):
        pass

    def reset(self, do_resets=None):
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
        return lambda_returns

    def actor_loss(self, latents, values, action, target, weights):
        policy = self.actor(latents.detach())

        if self.config.actor.grad == 'reinforce':
            baseline = values
            advantage = (target - baseline).detach()
            objective = policy.log_prob(action) * advantage
        elif self.config.actor.grad == 'dynamics':
            objective = target
        ent_scale = self.config.actor.ent_scale
        objective += ent_scale * policy.entropy()
        actor_loss = -(objective * weights)[:-1].mean()
        return actor_loss, policy.entropy()

    def critic_loss(self, latents, targets, weights):
        dist = self.critic(latents)
        critic_loss = -(dist.log_prob(targets) * weights)[:-1].mean()
        return critic_loss

    def update_target_critic(self):
        self.target_critic.load_state_dict(self.critic.state_dict())


class RSSM(torch.nn.Module):

    def __init__(self,
                 env_spec,
                 config,):
        super().__init__()
        self._env_spec = env_spec
        self.config = config
        self.action_size = env_spec.action_space.n

        self.embed_size = self.config.image_encoder.N * 32
        self.stoch_state_classes = self.config.rssm.stoch_state_classes
        self.stoch_state_size = self.config.rssm.stoch_state_size
        self.det_state_size = self.config.rssm.det_state_size
        self.act = eval(self.config.rssm.act)

        self.register_parameter(
            name='cell_initial_state',
            param=nn.Parameter(torch.zeros(self.det_state_size))
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
                                  self.stoch_state_classes).to(global_device()),
            'stoch': torch.zeros(batch_size,
                                 self.stoch_state_size,
                                 self.stoch_state_classes).to(global_device()),
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
        prior = StraightThroughOneHotDist(logits=logits)
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
        posterior = StraightThroughOneHotDist(logits=logits)
        return posterior


class ImageEncoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        Activation = eval(self.config.image_encoder.Activation)
        self.N = N = self.config.image_encoder.N

        self.model = nn.Sequential(
            nn.Conv2d(1, N*1, 4, 2),
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
        batch_shape = embed.shape[0]
        x = self.dense(embed).reshape(-1, self.N*32, 1, 1)
        x = self.deconvolve(x)
        return x


class MLP(torch.nn.Module):

    def __init__(self, input_shape, units, out_shape=1,
                 dist='mse', Activation=torch.nn.ELU):
        super().__init__()
        self.dist = dist

        self.net = nn.Sequential()
        for i, unit in enumerate(units):
            self.net.add_module(f"linear_{i}", nn.Linear(input_shape, unit))
            self.net.add_module(f"activation_{i}", Activation())
            input_shape = unit
        self.net.add_module("out_layer", nn.Linear(input_shape, out_shape))

    def forward(self, features):
        logits = self.net(features).squeeze()
        if self.dist == 'mse':
            return torch.distributions.Normal(loc=logits, scale=1)
        elif self.dist == 'bernoulli':
            return torch.distributions.Bernoulli(logits=logits)
        elif self.dist == 'onehot':
            return StraightThroughOneHotDist(logits=logits)

