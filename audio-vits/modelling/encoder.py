
import torch
from torch import nn
from torch.nn import functional as F
from .modules import commons
from .modules.wav_net import WavNet

class Encoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WavNet(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        #x=32x1025x799, x_lengths=32x1, g=32x768x1
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)#x_mask=32x1x799
        x = self.pre(x) * x_mask#x=32x192x799
        x = self.enc(x, x_mask, g=g)#x=32x192x799
        stats = self.proj(x) * x_mask#x=32x384x799
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask #m,logs,z=32x192x799
        return z, m, logs, x_mask
    