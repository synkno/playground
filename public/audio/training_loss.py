import torch
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram
from typing import List

class MultiMelSpecLoss(nn.Module):
    def __init__(
        self,
        sample_rate:int,
        n_mels:List[int],
        window_lengths:List[int],
        clamp_eps:float,
    ):
        super().__init__()
        self.mel_transforms = nn.ModuleList([
            MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=window_length,
                hop_length=window_length//4,
                n_mels=n_mel,
                power=1.0,
                center=True,
                norm="slaney",
                mel_scale="slaney"
            )
            for n_mel, window_length in zip(n_mels, window_lengths)
        ])
        self.n_mels = n_mels
        self.loss_fn = nn.L1Loss()
        self.clamp_eps = clamp_eps
    def forward(self, x:torch.Tensor, y:torch.Tensor)->torch.Tensor:
        loss = 0.0
        for mel_transform in self.mel_transforms:
            x_mel:torch.Tensor = mel_transform(x)
            y_mel:torch.Tensor = mel_transform(y)

            log_x_mel = x_mel.clamp(self.clamp_eps).log10()
            log_y_mel = y_mel.clamp(self.clamp_eps).log10()
            #clamp(self.clamp_eps) 避免数据为0
            #MelSpectrogram 默认是 power=1.0（即幅度谱），但有时候我们希望对其进行非线性变换，比如 pow=2.0 表示能量谱。
            #log10： 将频谱转换为对数尺度，更贴近人类听觉系统。，人耳对声音的感知是对数感知的（比如分贝 dB 就是 log10 的单位），而不是线性感知。

            loss += self.loss_fn(log_x_mel, log_y_mel)
        return loss
    
def stft_loss(x_rec: torch.Tensor, x_gt: torch.Tensor, configs = [(1024,256), (512,128), (256,64)]) -> torch.Tensor:
    x_rec = x_rec.squeeze(1)
    x_gt  = x_gt.squeeze(1)
    loss = 0.0
    for win, hop in configs:
        window = torch.hann_window(win, device=x_rec.device)
        Sr = torch.stft(x_rec, n_fft=win, hop_length=hop, win_length=win, window=window, return_complex=True)
        Sg = torch.stft(x_gt,  n_fft=win, hop_length=hop, win_length=win, window=window, return_complex=True)
        loss = loss + (Sr.abs() - Sg.abs()).abs().mean()
    return loss / len(configs)