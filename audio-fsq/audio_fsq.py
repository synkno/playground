import math
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from fsq import FSQ
from public.toolkit.nn import init_weights

def make_group_norm(num_channels:int, max_groups:int = 8) -> nn.GroupNorm:
    for g in range(min(max_groups, num_channels), 0, -1):
        #从大到小找一个 可以分组的值
        if num_channels % g == 0:
            return nn.GroupNorm(g, num_channels)
    return nn.GroupNorm(1, num_channels)

#在每个通道上进行norm
class ChannelwiseNorm1d(nn.Module):
    def __init__(self, 
        num_channels:int, 
        eps:float = 1e-5, 
        affine:bool = True
    ):
        super().__init__()
        self.affine = affine
        self.eps = eps
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, num_channels, 1))
            self.beta = nn.Parameter(torch.zeros(1, num_channels, 1))
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x_hat = (x - mean)/torch.sqrt(var + self.eps)
        if self.affine:
            x_hat = x_hat * self.gamma + self.beta
        return x_hat

#残差 + Dilated Conv1d  + GroupNorm + SiLU
class ResDilatedBlock1D(nn.Module):
    def __init__(self, 
        channels:int, 
        dilation1: int = 1, 
        dilation2: int = 2, 
        kernel_size: int = 3
    ):
        super().__init__()
        pad1 = (kernel_size - 1)//2 * dilation1
        pad2 = (kernel_size - 1) //2 * dilation2
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size, 
            padding=pad1, dilation=dilation1
        )
        self.norm1 = make_group_norm(channels)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size, 
            padding=pad2, dilation=dilation2
        )
        self.norm2 = make_group_norm(channels)
        self.act2 = nn.SiLU()
    
    def forward(self, x:torch.Tensor):
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x  = x + residual
        x = self.act2(x)
        return x

class Encoder1D(nn.Module):
    def __init__(self, 
        in_channels:int = 1,
        base_channels:int = 128,
        latent_channels:int = 128,
        strides:List[int] = [5, 4, 4, 3, 2],
        num_res_per_stage:int = 1
    ):
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, c, kernel_size=7, padding=3),
            make_group_norm(c),
            nn.SiLU()
        )

        stages = []
        for s in strides:
            blocks = []
            for _ in range(num_res_per_stage):
                blocks.append(
                    ResDilatedBlock1D(
                        c, dilation1=1, dilation2=2, kernel_size=3
                    )
                )
            blocks.append(nn.Conv1d(c, c, kernel_size=s, stride=s, padding=0))
            #下采样，这个卷积减少时间步数
            blocks.append(make_group_norm(c))
            blocks.append(nn.SiLU())
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)
        self.pre_vq_norm = ChannelwiseNorm1d(c, affine=True)
        self.to_latent = nn.Conv1d(c, latent_channels, kernel_size=1)
    
    def forward(self, x:torch.Tensor):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x= self.pre_vq_norm(x)
        z_e = self.to_latent(x)
        return z_e
    

class Decoder1D(nn.Module):
    def __init__(self, 
        latent_channels: int = 128,
        base_channels: int = 128,
        strides: List[int] = [5, 4, 4, 3, 2],
        num_res_per_stage: int = 1,
        out_channels: int = 1,
    ):
        super().__init__()
        c = base_channels
        self.from_latent = nn.Sequential(
            nn.Conv1d(latent_channels, c, kernel_size=1),
            make_group_norm(c),
            nn.SiLU()
        )
        stages = []
        for s in reversed(strides):
            blocks = []
            blocks.append(nn.Upsample(scale_factor=s))
            #blocks.append(nn.ConvTranspose1d(c, c, kernel_size=s + 2, stride=s, padding=1))
            #上采样，增加时间步数
            blocks.append(nn.Conv1d(c, c, kernel_size=3, padding=1))
            blocks.append(make_group_norm(c))
            blocks.append(nn.SiLU())
            for _ in range(num_res_per_stage):
                blocks.append(ResDilatedBlock1D(c, dilation1=1, dilation2=2, kernel_size=3))
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)
        self.head = nn.Sequential(
            nn.Conv1d(c, out_channels, kernel_size=7, padding=3),
            nn.Tanh()
        )
    
    def forward(self, z_q:torch.Tensor):
        x = self.from_latent(z_q)
        for stage in self.stages:
            x = stage(x)
        x = self.head(x)
        return x


class AudioFSQ(nn.Module):
    def __init__(self, 
        encoder:Encoder1D, fsq:FSQ, decoder:Decoder1D
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.fsq = fsq
    
    def forward(self, x:torch.Tensor):
        z_e = self.encoder(x)
        z_e = self.fsq(z_e)
        x_rec = self.decoder(z_e)
        return {"recons" : x_rec}
    
def build_model(
    input_dim: int = 128,       
    levels:List[int] = [4, 4, 4, 4, 4, 4],
    num_codebooks:int = 12,
    base_channels: int = 128,
    strides: List[int] = [5, 4, 2, 1],
    num_res_per_stage: int = 1,
    model_file:str = None
):
    vq = FSQ(
        dim=input_dim,
        levels = levels,
        num_codebooks = num_codebooks,
    )

    enc = Encoder1D(
        in_channels=1,
        base_channels=base_channels,
        latent_channels=input_dim,
        strides=strides,
        num_res_per_stage=num_res_per_stage,
    )

    dec = Decoder1D(
        latent_channels=input_dim,
        base_channels=base_channels,
        strides=strides,
        num_res_per_stage=num_res_per_stage,
        out_channels=1,
    )

    model = AudioFSQ(enc, vq, dec)
    model.apply(lambda m : init_weights(m, (nn.Linear, nn.Conv1d)))
    if model_file:
        state_dict = torch.load(model_file)
        model.load_state_dict(state_dict)
        
    return model