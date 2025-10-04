import torch
from torch import nn
from torch.nn import functional as F

from .modules.multi_head_attn import MultiHeadAttn
from .modules.ffn import FFN

from .modules import commons


class Encoder(nn.Module):
    def __init__(
        self,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size=1,
        p_dropout=0.0,
        window_size=4,
        **kwargs
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size

        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for i in range(self.n_layers):
            self.attn_layers.append(
                MultiHeadAttn(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    window_size=window_size,
                )
            )
            self.norm_layers_1.append(commons.LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_2.append(commons.LayerNorm(hidden_channels))

    def forward(self, x, x_mask):#x=32x192x799, x_mask=32x1x799
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)#32x1x799x799
        #x_mask.unsqueeze(2) → [B, L, 1], x_mask.unsqueeze(-1) → [B, 1, L]
        #attn_mask → [B, L, L], 
        #只有当 query 位置 i 和 key 位置 j 都是有效 token 时, attn_mask[b, i, j] = 1. 否则都是0
        x = x * x_mask#x=32x192x799
        for i in range(self.n_layers):
            y = self.attn_layers[i](x, x, attn_mask)#y=32x192x799
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        return x#x=32x192x799


class TextEncoder(nn.Module):
    def __init__(
        self,
        out_channels,
        hidden_channels,
        kernel_size,
        n_layers,
        gin_channels=0,
        filter_channels=None,
        n_heads=None,
        p_dropout=None,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
        self.f0_emb = nn.Embedding(256, hidden_channels)

        self.enc_ = Encoder(
            hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
        )

    def forward(
        self, x, x_mask, f0=None, noice_scale=1
    ):  # x=6x192x790, x_mask=6x1x790, f0=6x790
        x = x + self.f0_emb(f0).transpose(1, 2)  # self.f0_emb(f0)=6x790x192
        x = self.enc_(x * x_mask, x_mask)  # x=6x192x790
        stats = self.proj(x) * x_mask  # stats=6x384x790
        m, logs = torch.split(
            stats, self.out_channels, dim=1
        )  # m,logs=6x192x790, self.out_channels=192
        z = (m + torch.randn_like(m) * torch.exp(logs) * noice_scale) * x_mask
        # z=6x192x790
        return z, m, logs, x_mask
