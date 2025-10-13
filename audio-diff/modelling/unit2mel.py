import os

import numpy as np
import torch
import torch.nn as nn
import yaml

from .diffusion import GaussianDiffusion
from .wavenet import WaveNet


class Unit2Mel(nn.Module):
    def __init__(
            self,
            input_channel,
            n_spk,
            out_dims=128,
            n_layers=20, 
            n_chans=384, 
            n_hidden=256,
            timesteps=1000,
            k_step_max=1000
            ):
        super().__init__()
        self.unit_embed = nn.Linear(input_channel, n_hidden)
        self.f0_embed = nn.Linear(1, n_hidden)
        self.volume_embed = nn.Linear(1, n_hidden)
        self.spk_embed = nn.Linear(192, n_hidden)
        self.aug_shift_embed = nn.Linear(1, n_hidden, bias=False)
        self.n_spk = n_spk
        
        
        self.timesteps = timesteps if timesteps is not None else 1000
        self.k_step_max = k_step_max if k_step_max is not None and k_step_max>0 and k_step_max<self.timesteps else self.timesteps

        self.n_hidden = n_hidden
        # diffusion
        self.decoder = GaussianDiffusion(
            WaveNet(out_dims, n_layers, n_chans, n_hidden),
            timesteps=self.timesteps,k_step=self.k_step_max, out_dims=out_dims
        )
        self.input_channel = input_channel
    
    def forward(self, units, f0, volume, spk_emb = None, aug_shift = None, gt_spec=None, infer=True, k_step=300):
        #units(content vec)=8x172x768, f0=8x172x1, volume=8x172x1, spk_emb=8x1x192, aug_shift=8x1x1, gt_spec=8x172x128
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''

        if not self.training and gt_spec is not None and k_step>self.k_step_max:
            raise Exception("The shallow diffusion k_step is greater than the maximum diffusion k_step(k_step_max)!")

        if not self.training and gt_spec is None and self.k_step_max!=self.timesteps:
            raise Exception("This model can only be used for shallow diffusion and can not infer alone!")

        x = self.unit_embed(units) + self.f0_embed((1+ f0 / 700).log()) + self.volume_embed(volume)#x=8x172x256
        spk_emb = self.spk_embed(spk_emb)#8x1x256
        x = x + spk_emb
        if aug_shift is not None:
            x = x + self.aug_shift_embed(aug_shift / 5) #self.aug_shift_embed(aug_shift / 5) = 8x1x256
        x = self.decoder(x, gt_spec=gt_spec, infer=infer,  k_step=k_step)
    
        return x

