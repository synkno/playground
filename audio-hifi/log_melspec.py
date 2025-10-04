import torch
import torchaudio

class LogMelSpec(torch.nn.Module):
    def __init__(self,
                 sample_rate=16000,
                 n_fft=1024,
                 hop_length=160,
                 win_length=1024,
                 n_mels=80,
                 f_min=0.0,
                 center=True,
                 f_max=None):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            center=center,
            power=1.0
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(stype="power")

    def forward(self, wav):
        mel = self.mel_spec(wav) 
        mel_db = self.db_transform(mel) 
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)
        return mel_db
