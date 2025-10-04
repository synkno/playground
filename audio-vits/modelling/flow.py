import torch
from torch import nn
from torch.nn import functional as F
from .modules.wav_net import WavNet


class Flip(nn.Module):
    def forward(self, x, *args, reverse=False, **kwargs):
        x = torch.flip(x, [1])#x=32x192x799
        if not reverse:
            logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
            return x, logdet#logdet 32个0
        else:
            return x


class ResidualCouplingLayer(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        p_dropout=0,
        gin_channels=0,
        mean_only=False,
        wn_sharing_parameter=None,
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = (
            WavNet(
                hidden_channels,
                kernel_size,
                dilation_rate,
                n_layers,
                p_dropout=p_dropout,
                gin_channels=gin_channels,
            )
            if wn_sharing_parameter is None
            else wn_sharing_parameter
        )
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):#x=32x192x799, x_mask=32x1x799, g=32x768x1
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)#x0=32x96x799,x1=32x96x799
        h = self.pre(x0) * x_mask#h=32x192x799
        h = self.enc(h, x_mask, g=g)#h=32x192x799
        stats = self.post(h) * x_mask#stats=32x96x799
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats#m=32x96x799
            logs = torch.zeros_like(m)#logs=32x96x799

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask#logs是0，x1=m + x1*x_mask
            x = torch.cat([x0, x1], 1)#x=32x192x799
            logdet = torch.sum(logs, [1, 2])#logdet=32x1
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x


class Flow(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0,
        share_parameter=False,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()

        self.wn = (
            WavNet(
                hidden_channels,
                kernel_size,
                dilation_rate,
                n_layers,
                p_dropout=0,
                gin_channels=gin_channels,
            )
            if share_parameter
            else None
        )

        for i in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                    wn_sharing_parameter=self.wn,
                )
            )
            self.flows.append(Flip())

    def forward(
        self, x, x_mask, g=None, reverse=False
    ):  # x=32x192x799, x_mask=32x1x799, g=32x768x1, reverse=False,
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x
