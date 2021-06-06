import torch
import torch.nn.functional as F
from torch import nn
from torch import distributions
from torch.distributions import kl_divergence, Independent
import torch.nn.functional as F
from garage.torch import global_device
from utils import CONFIG

def categorical_kl(probs_a, probs_b):
    return torch.sum(probs_a * torch.log(probs_a / probs_b), dim=[-1, -2])


def kl_loss(posterior, prior):
    lhs = categorical_kl(posterior.probs.detach(), prior.probs)
    rhs = categorical_kl(posterior.probs, prior.probs.detach())
    kl_loss = CONFIG.rssm.alpha * lhs + (1 - CONFIG.rssm.alpha) * rhs

    assert torch.isclose(lhs, rhs).all()

    expected = kl_divergence(
        Independent(posterior, 1),
        Independent(prior, 1),)

    assert torch.isclose(lhs, expected, atol=1e-4).all(), lhs - expected

    return kl_loss


class WorldModel(torch.nn.Module):

    def __init__(self, env_spec):
        super().__init__()
        self.env_spec = env_spec

        self.rssm = RSSM(env_spec)
        self.image_encoder = ImageEncoder()
        self.image_decoder = ImageDecoder()

        self.feat_size = (
            CONFIG.rssm.stoch_state_classes * CONFIG.rssm.stoch_state_size
            + CONFIG.rssm.det_state_size
        )

        self.reward_predictor = MLP(
            input_shape=self.feat_size,
            units=CONFIG.reward_head.units,
            dist='mse')

        self.discount_predictor = MLP(
            input_shape=self.feat_size,
            units=CONFIG.discount_head.units,
            dist='bernoulli')

    def forward(self, observations, actions):
        segs, steps, channels, height, width = observations.shape
        flattened_observations = observations.reshape(
            segs*steps, channels, height, width)
        embedded_observations = self.image_encoder(
            flattened_observations).reshape(segs, steps, -1)
        out = self.rssm.observe(embedded_observations, actions)
        out['reward_dist'] = self.reward_predictor(out['feats'])
        out['discount_dist'] = self.discount_predictor(out['feats'])
        flattened_feats = out['feats'].reshape(segs*steps, self.feat_size)
        mean = self.image_decoder(flattened_feats).reshape(
            segs, steps, channels, height, width)
        norm = distributions.Normal(loc=mean, scale=1)
        image_recon_dist = distributions.Independent(norm, 3)
        assert image_recon_dist.batch_shape == (segs, steps)
        out['image_recon_dist'] = image_recon_dist
        return out

    def loss(self, out, observation_batch, reward_batch, discount_batch):
        kl_loss = out['kl_losses'].mean()
        reward_loss = -out['reward_dist'].log_prob(reward_batch).mean()
        discount_loss = -out['discount_dist'].log_prob(discount_batch).mean()
        recon_loss = -out['image_recon_dist'].log_prob(observation_batch).mean()
        loss = reward_loss + discount_loss + recon_loss + CONFIG.rssm.beta * kl_loss
        loss_info = {
            'kl_loss': kl_loss,
            'reward_loss': reward_loss,
            'discount_loss': discount_loss,
            'recon_loss': recon_loss,
            'total_loss': loss,
        }
        return loss, loss_info


class Actor(object):
    pass


class RSSM(torch.nn.Module):

    def __init__(self,
                 env_spec,):
        super().__init__()
        self._env_spec = env_spec
        self.action_size = env_spec.action_space.n

        self.embed_size = CONFIG.rssm.embed_size
        self.stoch_state_classes = CONFIG.rssm.stoch_state_classes
        self.stoch_state_size = CONFIG.rssm.stoch_state_size
        self.det_state_size = CONFIG.rssm.det_state_size
        self.act = eval(CONFIG.rssm.act)

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

    def get_stoch(self, x):
        logits = x.reshape(
            *x.shape[:-1],
            self.stoch_state_size,
            self.stoch_state_classes)
        dist = distributions.Categorical(logits=logits)
        sample = F.one_hot(dist.sample(), num_classes=self.stoch_state_classes).type(torch.float)
        sample += dist.probs - dist.probs.detach()  # Straight through gradients trick
        return sample, dist

    def imagine_step(self, prev_stoch, prev_deter, prev_action):
        deter = self.step(prev_stoch, prev_deter, prev_action)
        x = self.act(self.imagine_out_1(deter))
        x = self.imagine_out_2(x)
        sample, dist = self.get_stoch(x)
        prior = {'sample': sample, 'dist': dist}
        return prior, deter

    def observe_step(self, prev_stoch, prev_deter, prev_action, embed):
        prior, deter = self.imagine_step(prev_stoch, prev_deter, prev_action)
        x = torch.cat([deter, embed], dim=-1)
        x = self.act(self.observe_out_1(x))
        x = self.observe_out_2(x)
        sample, dist = self.get_stoch(x)
        posterior = {'sample': sample, 'dist': dist}
        return posterior, prior, deter

    def imagine(self):
        pass

    def observe(self, embedded_observations, actions):
        segs, steps, embedding_size = embedded_observations.shape
        assert segs == actions.shape[0]
        assert steps == actions.shape[1]

        # Change from SEGS x STEPS x N -> STEPS x SEGS x N
        # This facilitates 
        embedded_observations = torch.swapaxes(embedded_observations, 0, 1)
        actions = torch.swapaxes(actions, 0, 1)

        initial = self.initial_state(batch_size=segs)
        stoch, deter = initial['stoch'], initial['deter']

        posteriors = []
        priors = []
        deters = []
        feats = []
        kl_losses = []

        for embed, action in zip(embedded_observations, actions):
            posterior, prior, deter = self.observe_step(stoch, deter, action, embed)
            stoch = posterior['sample']

            posteriors.append(posterior)
            priors.append(prior)
            deters.append(deter)
            feats.append(torch.cat([stoch.flatten(start_dim=1), deter], dim=-1))
            kl_losses.append(kl_loss(posterior['dist'], prior['dist']))

        out = {
            'posteriors': posteriors,
            'priors': priors,
            'deters': torch.swapaxes(torch.stack(deters), 0, 1),
            'feats': torch.swapaxes(torch.stack(feats), 0, 1),
            'kl_losses': torch.swapaxes(torch.stack(kl_losses), 0, 1)
        }

        return out


class ImageEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        Activation = eval(CONFIG.image_encoder.Activation)
        self.N = N = CONFIG.image_encoder.N

        self.model = nn.Sequential(
            nn.Conv2d(1, N*1, 4, 2),
            Activation(),
            nn.Conv2d(32, N*2, 4, 2),
            Activation(),
            nn.Conv2d(64, N*4, 4, 2),
            Activation(),
            nn.Conv2d(128, N*8, 4, 2),
            Activation(),
        )

    def forward(self, img):
        x = self.model(img)
        return torch.flatten(x, start_dim=1)

class ImageDecoder(torch.nn.Module):

    def __init__(self):
        super().__init__()

        Activation = eval(CONFIG.image_decoder.Activation)
        self.N = N = CONFIG.image_decoder.N
        self.shape = [
            CONFIG.image.height,
            CONFIG.image.width,
            CONFIG.image.color_channels
        ]
        feat_shape = (
            CONFIG.rssm.stoch_state_classes * CONFIG.rssm.stoch_state_size
            + CONFIG.rssm.det_state_size)

        self.dense = nn.Linear(feat_shape, N*32)
        self.deconvolve = nn.Sequential(
            nn.ConvTranspose2d(N*32, N*4, 5, 2),
            Activation(),
            nn.ConvTranspose2d(N*4, N*2, 5, 2),
            Activation(),
            nn.ConvTranspose2d(N*2, N, 6, 2),
            Activation(),
            nn.ConvTranspose2d(N, self.shape[-1], 6, 2),
            nn.Sigmoid(),  # TODO: Check this
        )

    def forward(self, embed):
        batch_shape = embed.shape[0]
        x = self.dense(embed).reshape(-1, self.N*32, 1, 1)
        x = self.deconvolve(x)
        return x


class MLP(torch.nn.Module):

    def __init__(self, input_shape, units, dist='mse', Activation=torch.nn.ELU):
        super().__init__()
        self.dist = dist

        self.net = nn.Sequential()
        for i, unit in enumerate(units):
            self.net.add_module(f"linear_{i}", nn.Linear(input_shape, unit))
            self.net.add_module(f"activation_{i}", Activation())
            input_shape = unit
        self.net.add_module("out_layer", nn.Linear(input_shape, 1))

    def forward(self, features):
        logits = self.net(features).squeeze()
        if self.dist == 'mse':
            return torch.distributions.Normal(loc=logits, scale=1)
        elif self.dist == 'bernoulli':
            return torch.distributions.Bernoulli(logits=logits)

