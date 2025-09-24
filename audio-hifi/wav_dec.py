import math
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
import torch.nn.functional as F
from public.toolkit.nn import size_of_model



class ResBlock(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 =  nn.ModuleList([
            weight_norm( 
                nn.Conv1d( 
                    channels,  
                    channels,
                    kernel_size, 
                    1, 
                    dilation=d, 
                    padding=int((kernel_size * d - d) / 2)
                )
            ) for d in dilation
        ])
        self.convs2 = nn.ModuleList([
            weight_norm( 
                nn.Conv1d( 
                    channels,  
                    channels,
                    kernel_size, 
                    1, 
                    dilation=1, 
                    padding=int((kernel_size  - 1) / 2)
                )
            ) for d in dilation
        ])

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x
    
class WavDec(nn.Module):
    def __init__(self, 
        in_channels: int = 128,
        strides: List[int] = [5, 4, 4, 3, 2],
        #resblock_kernel_sizes=[3, 5, 7, 9, 11, 13],
        resblock_kernel_sizes=[3, 5, 7],
        resblock_dilation_sizes=[(1, 3, 5),
        (1, 4, 6),
        (1, 6, 9),
        #(1, 2, 4),
        #(1, 5, 7),
        #(1, 8, 12)
        ],
        out_channels: int = 1,
    ):
        super().__init__()
        c = in_channels 
        self.in_proj = nn.Conv1d( in_channels, c, 7, 1, padding=3)

        stages = []
        for i, s in enumerate(strides):
            
            #blocks.append(nn.Upsample(scale_factor=s, mode="linear"))
            ic = c // (2 ** i)
            oc = c // (2 ** (i + 1))
            up = (weight_norm(
                nn.ConvTranspose1d(
                    ic, oc, kernel_size=s*2 - (s%2), stride=s, padding=(s*2 - s)//2
                )
            ))
            blocks = []
            for k, d in  zip(resblock_kernel_sizes, resblock_dilation_sizes):
                blocks.append(ResBlock(oc, k, d))
            stages.append(
                nn.ModuleList([up, nn.ModuleList(blocks)])
            )
        self.stages = nn.ModuleList(stages)
        self.head = nn.Sequential(
            nn.Conv1d(oc, out_channels, kernel_size=7, padding=3),
            nn.Tanh()
        )
    
    def forward(self, content_vec:torch.Tensor):
        x = self.in_proj(content_vec)
        for stage in self.stages:
            x = F.leaky_relu(x, 0.1)
            x = stage[0](x)
            xs = None
            for b in stage[1]:
                xs = b(x) if xs is None else xs + b(x)
            x = xs / len(stage[1])
        
        x = F.leaky_relu(x)
        x = self.head(x)
        return x