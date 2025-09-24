import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple, Dict, Any, List
from audiotools import AudioSignal

def stft( x: torch.Tensor, fft_size: int, hop_size: int,  win_length: int, window: str) -> torch.Tensor:
    x_stft = torch.stft(
        x, fft_size, hop_size, win_length, window.to(x.device), return_complex=True
    )
    res = torch.cat([x_stft.real.unsqueeze(1), x_stft.imag.unsqueeze(1)], dim=1)
    return res.transpose(2, 3)  # [B, 2, T, F]


class MPD(nn.Module):#Multi-Period Discriminator
    def __init__(self, period):#period是每个周期的长度.
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(1, 16, (5, 1), (3, 1), padding=(2, 0)),
                nn.Conv2d(16, 64, (5, 1), (3, 1), padding=(2, 0)),
                nn.Conv2d(64, 128, (5, 1), (3, 1), padding=(2, 0)),
                nn.Conv2d(128, 256, (5, 1), (3, 1), padding=(2, 0)),
                nn.Conv2d(256, 1, (5, 1), 1, padding=(2, 0)),
            ]
        )
        self.conv_post = nn.Conv2d(
            1, 1, kernel_size=(3, 1), padding=(1, 0), 
        )

    def pad_to_period(self, x):#此函数是为了能让x被self.period整除
        t = x.shape[-1]#t=38400
        x = F.pad(x, (0, self.period - t % self.period), mode="reflect")#self.period=2, x=8x1x38402
        return x

    def forward(self, x):#x=8x1x38400
        fmap = []

        x = self.pad_to_period(x)#x=8x1x38402
        x = rearrange(x, "b c (l p) -> b c l p", p=self.period)#x=x=8x1x19201x2

        for layer in self.convs:
            x = layer(x)#x=8x32x6401x2, 8x128x2134x2, 8x512x712x2, 8x1024x238x2, 8x1024x238x2
            fmap.append(x)

        x = self.conv_post(x)#8x1x238x2
        fmap.append(x)

        return fmap

BANDS = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]


class MRD(nn.Module):#Multi-Resolution Discriminator
    def __init__(
        self,
        fft_size: int,
        win_length: int,
        hop_size: int,
        window: str,
        sample_rate: int = 44100,
        bands: list = BANDS,
    ):
        super().__init__()

        self.fft_size = fft_size
        self.win_length = win_length
        self.hop_size = hop_size
        self.sample_rate = sample_rate
        self.window = getattr(torch, window)(win_length)

        bin_size = self.fft_size // 2 + 1
        self.bands = [(int(b[0] * bin_size), int(b[1] * bin_size)) for b in bands]

        ch = 32
        convs = lambda: nn.ModuleList(
            [
                 nn.Conv2d(2, ch, (3, 9), (1, 1), padding=(1, 4)),
                 nn.Conv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                 nn.Conv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                 nn.Conv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                 nn.Conv2d(ch, ch, (3, 3), (1, 1), padding=(1, 1)),
            ]
        )
        self.band_convs = nn.ModuleList([convs() for _ in range(len(self.bands))])
        self.conv_post =  nn.Conv2d(ch, 1, (3, 3), (1, 1), padding=(1, 1), )

    def spectrogram(self, x:torch.Tensor):
        if len(x.shape) == 3:
            x = x.squeeze(1)
        t = x.dtype
        x = stft(x, self.fft_size, self.hop_size, self.win_length, self.window).to(t)
        x_bands = [x[..., b[0] : b[1]] for b in self.bands]
        return x_bands 

    def forward(self, x):
        x_bands = self.spectrogram(x)
        fmap = []

        x = []
        for band, stack in zip(x_bands, self.band_convs):
            for layer in stack:
                band = layer(band)
                fmap.append(band)
            x.append(band)
        x = torch.cat(x, dim=-1)
        x = self.conv_post(x) 
        fmap.append(x)
        return fmap


class WaveDisc(nn.Module):
    def __init__(
        self,
        periods: list,
        sample_rate: int,
        bands: list,
        stft_params: Dict[str, Any],
    ):
        super().__init__()
        
        discs = []
        discs += [MPD(p) for p in periods]
        discs += [MRD(
            stft_params["fft_sizes"][i],
            stft_params["hop_sizes"][i],
            stft_params["win_lengths"][i],
            stft_params["window"],
            sample_rate=sample_rate,
            bands=bands) for i in range(len(stft_params["fft_sizes"]))]
        self.discriminators = nn.ModuleList(discs)
        self.sample_rate = sample_rate

    def __preprocess(self, y):
        # Remove DC offset
        y = y - y.mean(dim=-1, keepdims=True)
        # Peak normalize the volume of input audio
        y = 0.8 * y / (y.abs().max(dim=-1, keepdim=True)[0] + 1e-9)
        return y

    def __forward(self, x):
        x = self.__preprocess(x)
        fmaps = [d(x) for d in self.discriminators]
        return fmaps#[[tensor, ...], [tensor, ...], ...]
    

    def forward( self, fake: AudioSignal, real: AudioSignal ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        d_fake = self.__forward(fake.audio_data)
        d_real = self.__forward(real.audio_data)
        return d_fake, d_real
    

    def adversarial_loss(self, raw_wav, rec_wav) -> Dict[str, Any]:
        """
        Get adversarial loss
        """
        signal = AudioSignal(raw_wav.clone(), self.sample_rate)
        recons = AudioSignal(rec_wav.clone(), signal.sample_rate)

        d_fake, d_real = self.forward(recons, signal)

        adv_loss = 0
        for x_fake in d_fake:
            adv_loss += F.mse_loss(x_fake[-1], torch.ones_like(x_fake[-1]))
            #生成器希望判别器输出接近 1（即“真实”）的结果。

        feature_map_loss = 0
        for i in range(len(d_fake)):
            for j in range(len(d_fake[i]) - 1):
                feature_map_loss += F.l1_loss(d_fake[i][j], d_real[i][j].detach())
                #中间层的 L1 差异, 生成器在特征层面也模仿真实音频的判别器响应。

        return {"adv_loss": adv_loss, "feature_map_loss": feature_map_loss}

    def discriminative_loss(self, raw_wav, rec_wav) -> torch.Tensor:
        """
        用来训练辨别器，希望假的声音接近0，真的声音接近1
        fake.clone().detach() 这样就不会训练生成器了。
        """
        real = AudioSignal(raw_wav, self.sample_rate) #inputs["audios"] 8x1x38400
        fake = AudioSignal(rec_wav, real.sample_rate) #inputs["recons"] 8x1x38400
        d_fake, d_real = self.forward(fake.clone().detach(), real)
        loss_d = 0
        for x_fake, x_real in zip(d_fake, d_real):
            loss_d += torch.mean(x_fake[-1] ** 2) #x_fake[-1]=8x1x238x2
            loss_d += torch.mean((1 - x_real[-1]) ** 2)#x_real[-1]=8x1x238x2

        return loss_d