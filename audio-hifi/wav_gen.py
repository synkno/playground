import torch
import torch.nn as nn
from wav_dec import WavDec
from wav_dec1 import WavDec1

from torchaudio.transforms import MelSpectrogram
from public.toolkit.nn import set_requires_grad, size_of_model
from config import config

class WavGen(nn.Module):
    def __init__(self, sample_rate:int):
        super().__init__()
        latent_dim = config["latent_dim"]
        content_dim = 128

        strides = [5, 2, 2, 4, 2, 2]
        if config["hop_length"] == 160:
            strides = [5, 2, 2, 2, 2, 2]
        if config["hop_length"] == 80:
            strides = [5, 2, 2, 2, 2]
        if config["hop_length"] == 40:
            strides = [5, 2, 2, 2]

        self.content_proj = nn.Conv1d(content_dim, latent_dim, kernel_size=7, stride=1, padding=3)
        if config["decoder"] == "v1":
            self.decoder = WavDec1(
                in_channels=latent_dim, strides=strides,
                resblock_kernel_sizes=config["resblock_kernel_sizes"],
                resblock_dilation_sizes=config["resblock_dilation_sizes"]
            ) 
        else:
            self.decoder = WavDec(
                in_channels=latent_dim, strides=strides, 
                resblock_kernel_sizes=config["resblock_kernel_sizes"],
                resblock_dilation_sizes=config["resblock_dilation_sizes"]
            ) 
            
    

    def forward(self, batch:dict):
        icv = self.content_proj(batch["mel"])
        x = self.decoder(icv)
        return {
            "wav" : batch["wav"],
            "hat_wav" : x,
        }
