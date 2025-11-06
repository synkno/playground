import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm, weight_norm
from torch.nn import functional as F

_lrelu_slope = 0.1

def _get_padding(kernel_size, dilation=1):
  return int((kernel_size*dilation - dilation)/2)
LRELU_SLOPE = 0.1

class DiscPeriod(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super().__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(_get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(_get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(_get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(_get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(_get_padding(kernel_size, 1), 0))),
        ])
        self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, _lrelu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscScale(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, _lrelu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


#Multi Period Discriminator
class MultiPeriodDisc(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        periods = [2, 3, 5, 7, 11]

        discs = [DiscScale(use_spectral_norm=use_spectral_norm)]
        discs = discs + [DiscPeriod(i, use_spectral_norm=use_spectral_norm) for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, wav_real, wav_fake):
        reals = []
        fakes = []
        fmap_reals = []
        fmap_fakes = []
        for i, d in enumerate(self.discriminators):
            real, fmap_real = d(wav_real)
            fake, fmap_fake = d(wav_fake)
            reals.append(real)
            fakes.append(fake)
            fmap_reals.append(fmap_real)
            fmap_fakes.append(fmap_fake)

        return reals, fakes, fmap_reals, fmap_fakes
    
    @staticmethod
    def discriminator_loss(net, wav:torch.Tensor, wav_hat:torch.Tensor):
        reals, fakes, _, _ = net(wav, wav_hat.detach())
        loss = 0
        for dr, df in zip(reals, fakes):
            dr = dr.float()
            df = df.float()
            r_loss = torch.mean((1 - dr) ** 2)
            f_loss = torch.mean(df**2)
            loss += r_loss + f_loss
        return loss

    @staticmethod
    def generator_loss(net, wav:torch.Tensor, wav_hat:torch.Tensor):
        _, fakes, fmap_reals, fmap_fakes = net(wav, wav_hat)
        loss = 0
        for f in fakes:
            f = f.float()
            loss += torch.mean((1 - f) ** 2)
        

        fmap_loss = 0
        for dr, df in zip(fmap_reals, fmap_fakes):
            for rl, fl in zip(dr, df):
                rl = rl.float().detach()
                fl = fl.float()
                fmap_loss += torch.mean(torch.abs(rl - fl))

        return loss, fmap_loss * 2
