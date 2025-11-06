import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import spectral_norm, weight_norm
from torch.nn.utils.parametrize import remove_parametrizations
from . import commons


def weight_norm_modules(module, name="weight", dim=0):
    return weight_norm(module, name, dim)


def remove_weight_norm_modules(module, name="weight"):
    remove_parametrizations(module, name)


class WavNet(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
        p_dropout=0,
    ):
        super().__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.kernel_size = (kernel_size,)
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels != 0:
            cond_layer = torch.nn.Conv1d(
                gin_channels, 2 * hidden_channels * n_layers, 1
            )
            self.cond_layer = weight_norm_modules(cond_layer, name="weight")

        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = nn.Conv1d(
                hidden_channels,
                2 * hidden_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
            )
            in_layer = weight_norm_modules(in_layer, name="weight")
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
            res_skip_layer = weight_norm_modules(res_skip_layer, name="weight")
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, x_mask, g=None, **kwargs):#x=32x192x799, x_mask=32x1x799, g=32x768x1
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])#self.hidden_channels=192

        if g is not None:
            g = self.cond_layer(g)#g=32x6144x1

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)#x_in=32x384x799
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :] #g_l=32x384x1
            else:
                g_l = torch.zeros_like(x_in)

            acts = commons.fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)#acts=32x192x799
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)#res_skip_acts=32x384x799
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, : self.hidden_channels, :]#res_acts=32x192x799
                x = (x + res_acts) * x_mask#x=32x192x799
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else:
                output = output + res_skip_acts
        return output * x_mask

    def remove_weight_norm(self):
        if self.gin_channels != 0:
            remove_weight_norm_modules(self.cond_layer)
        for l in self.in_layers:
            remove_weight_norm_modules(l)
        for l in self.res_skip_layers:
            remove_weight_norm_modules(l)
