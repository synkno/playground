import torch
from torch import nn

from torch.nn import functional as F


from torch.nn.utils.parametrizations import spectral_norm, weight_norm
from .modules.multi_head_attn import MultiHeadAttn
from .modules.ffn import FFN
from .modules import commons



class FFT(nn.Module):
    def __init__(
        self,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers=1,
        kernel_size=1,
        p_dropout=0.0,
        proximal_bias=False,
        proximal_init=True,
        isflow=False,
        **kwargs
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init
        if isflow:
            cond_layer = torch.nn.Conv1d(
                kwargs["gin_channels"], 2 * hidden_channels * n_layers, 1
            )
            self.cond_pre = torch.nn.Conv1d(hidden_channels, 2 * hidden_channels, 1)
            self.cond_layer = weight_norm(cond_layer)
            self.gin_channels = kwargs["gin_channels"]
        self.drop = nn.Dropout(p_dropout)
        self.self_attn_layers = nn.ModuleList()
        self.norm_layers_0 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        for i in range(self.n_layers):
            self.self_attn_layers.append(
                MultiHeadAttn(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    proximal_bias=proximal_bias,
                    proximal_init=proximal_init,
                )
            )
            self.norm_layers_0.append(commons.LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                    causal=True,
                )
            )
            self.norm_layers_1.append(commons.LayerNorm(hidden_channels))

    def forward(self, x, x_mask, g=None):#x=32x192x799, x_mask=32x1x799
        """
        x: decoder input
        h: encoder output
        """
        if g is not None:
            g = self.cond_layer(g)

        self_attn_mask = commons.subsequent_mask(x_mask.size(2)).to(
            device=x.device, dtype=x.dtype
        )#self_attn_mask=1x1x799x799
        x = x * x_mask
        for i in range(self.n_layers):
            if g is not None:
                x = self.cond_pre(x)
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
                x = commons.fused_add_tanh_sigmoid_multiply(
                    x, g_l, torch.IntTensor([self.hidden_channels])
                )
            y = self.self_attn_layers[i](x, x, self_attn_mask)#y=32x192x799
            y = self.drop(y)
            x = self.norm_layers_0[i](x + y)

            y = self.ffn_layers[i](x, x_mask)#y=32x192x799
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)
        x = x * x_mask
        return x#x=32x192x799


class F0Decoder(nn.Module):
    def __init__(
        self,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        spk_channels=0,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.spk_channels = spk_channels

        self.prenet = nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1)
        self.decoder = FFT(
            hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        self.f0_prenet = nn.Conv1d(1, hidden_channels, 3, padding=1)
        self.cond = nn.Conv1d(spk_channels, hidden_channels, 1)

    def forward(
        self, x, norm_f0, x_mask, spk_emb=None
    ):  # x=32x192x799, norm_f0=32x1x799, x_mask=32x1x799, spk_emb=32x768x1
        x = torch.detach(x)
        if spk_emb is not None:
            x = x + self.cond(spk_emb)#self.cond(spk_emb)=32x192x1
        x += self.f0_prenet(norm_f0)#x=32x192x799
        x = self.prenet(x) * x_mask#x=32x192x799
        x = self.decoder(x * x_mask, x_mask)
        x = self.proj(x) * x_mask  # x=6x1x790
        return x
