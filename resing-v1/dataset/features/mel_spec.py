import torch
import torch.nn as nn
import torchaudio.transforms as T


class MelSpectrogram:
    def __init__(self,
        sample_rate=None,
        n_fft=None,
        hop_length=None,
        center=False,
        device = "cpu"
    ):
        super().__init__()
        win_length=2048
        n_mels=80
        f_min=0.0
        f_max=22050
        center=False

        self.pad = (n_fft-hop_length)//2
        self.spectrogram = T.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            power=2.0,        
            center=center,
            pad=0,
            pad_mode="reflect"
        ).to(device)

        self.mel_scale = T.MelScale(
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=f_min,
            f_max=f_max,
            n_stft=n_fft // 2 + 1,
            mel_scale="slaney",
            norm="slaney"        
        ).to(device)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.get_mel(waveform)
    
    def get_spec(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = torch.nn.functional.pad(
            waveform.unsqueeze(1),
            (self.pad,  self.pad),
            mode='reflect'
        ).squeeze(1)
        return torch.sqrt(self.spectrogram(waveform)  + 1e-6)
    
    def get_mel_from_spec(self, spec: torch.Tensor) -> torch.Tensor:
        mel_spec = self.mel_scale(spec)   
        C=1 
        clip_val=1e-5
        return torch.log(torch.clamp(mel_spec, min=clip_val) * C)    
        #如果复现论文/开源模型（如 HiFi-GAN、Tacotron2） → 须用 torch.log(torch.clamp(mel_spec, min=clip_val) * C)  
    
    
    def get_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        spec = self.get_spec(waveform)
        return self.get_mel_from_spec(spec)




