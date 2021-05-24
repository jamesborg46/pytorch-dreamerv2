import torch
import torch.nn.functional as F
from torch import nn
from torch import distributions

class RSSM(torch.nn.Module):


    def __init__(self,
                 action_size,
                 stoch_state_size=30,
                 det_state_size=200,
                 Activation=nn.ELU):

        self.action_size = action_size
        self.stoch_state_size = stoch_state_size
        self.det_state_size = det_state_size

        self.cell = nn.GRU(input_size=det_state_size,
                           hidden_size=det_state_size)

        # Imagination layers
        self.embed_stoch_state_and_action = nn.Linear(
            action_size + stoch_state_size,
            det_state_size)

        # Observation Layers

    def observe_step(self, prev_state, prev_action, embed):
        pass

    def imagine_step(self, prev_state, prev_action):
        assert prev_state.ndim == 2 and prev_action.ndim == 2
        x = torch.cat((prev_state, prev_action), dim=1)
        x = self.embed_stoch_state_and_action(x)
        x = self.cell(x, prev_state['deter'])



class ImageEncodeor(torch.nn.Module):

    def __init__(self, N=32, Activation=nn.ReLU):
        super().__init__()
        self.N = N
        self.model = nn.Sequential(
            nn.Conv2d(3, N*1, 4, 2),
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

    def __init__(self, N=32, shape=(64, 64, 3), Activation=nn.ReLU):
        super().__init__()
        self.N = N
        self.shape = shape
        self.dense = nn.Linear(N*16, N*32)
        self.deconvolve = nn.Sequential(
            nn.ConvTranspose2d(N*32, N*4, 5, 2),
            Activation(),
            nn.ConvTranspose2d(N*4, N*2, 5, 2),
            Activation(),
            nn.ConvTranspose2d(N*2, N, 6, 2),
            Activation(),
            nn.ConvTranspose2d(N, shape[-1], 6, 2),
            nn.Sigmoid(),  # TODO: Check this
        )

    def forward(self, embed):
        batch_shape = embed.shape[0]
        x = self.dense(embed)
        x = self.deconvolve(x)
        mean = x.reshape(-1, *self.shape)
        norm = distributions.Normal(loc=mean, scale=1),
        dist = distributions.Independent(norm, len(self.shape))
        assert len(dist.batch_shape) == 1
        assert dist.batch_shape[0] == batch_shape
        return dist

