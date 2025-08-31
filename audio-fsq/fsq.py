from typing import List, Optional
import torch
import torch.nn as nn
from einops import rearrange



class FSQ(nn.Module):
    def __init__(self, 
        dim:int,
        levels:List[int],
        num_codebooks = 1,
        
    ):
        super().__init__()
        self.register_buffer("levels",
            torch.tensor(levels, dtype=torch.int32), persistent=False 
        )
        self.register_buffer("basis",
            torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=torch.int32),
            persistent=False
        )

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim =  codebook_dim * num_codebooks
        self.codebook_size = self.levels.prod().item()

        self.inproj = nn.Linear(dim, self.effective_codebook_dim)
        self.outproj = nn.Linear(self.effective_codebook_dim, dim)
    
    def forward(self, z:torch.Tensor):
        #z = Bacthsize, Dim, Timesteps
        z = self.inproj(z.transpose(1, 2))

        z = rearrange(z, "b t (c d) -> b t c d", c=self.num_codebooks)

        levels:torch.Tensor = self.levels
        basis:torch.Tensor = self.basis
        eps = 1e-3

        #这段量化代码让量化可微 可导
        half_l = (levels - 1) * (1 - eps) / 2
        offset = torch.where(levels % 2 == 0, 0.5, 0)
        shift = (offset / half_l).tan()
        quantized = (z + shift).tanh() * half_l - offset
        qround = quantized.round()
        quantized = quantized + (qround - quantized).detach()
        """
        当levels是奇数如=3时  half_l=2,   offset=0,    shift=0, quantized=z.tanh() * half_l
        当levels是偶数如=4时  half_l=2.5, offset=0.5,  shift=0.2, quantized=tanh( z + tan(0.5/2.5) ) * 2.5 - 0.5
        """
        half_width = levels // 2
        codes = quantized/half_width

        indices= (
            (codes - half_width) / half_width * basis
        ).sum(dim=-1).to(torch.int32)
        codes = rearrange(codes, 'b n c d -> b n (c d)')

        out:torch.Tensor = self.outproj(codes)
    

        return out.transpose(1, 2)








