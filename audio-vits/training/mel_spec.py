import torch
from .config import config

from librosa.filters import mel as librosa_mel_fn

MAX_WAV_VALUE = 32768.0


def __dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)
def __dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C
def __spectral_normalize_torch(magnitudes):
    output = __dynamic_range_compression_torch(magnitudes)
    return output
def __spectral_de_normalize_torch(magnitudes):
    output = __dynamic_range_decompression_torch(magnitudes)
    return output


__mel_basis = {}
__hann_window = {}


def __spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global __hann_window
    dtype_device = str(y.dtype) + '_' + str(y.device)
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if wnsize_dtype_device not in __hann_window:
        __hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)
    
    y_dtype = y.dtype
    if y.dtype == torch.bfloat16:
        y = y.to(torch.float32)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=__hann_window[wnsize_dtype_device],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    spec = torch.view_as_real(spec).to(y_dtype)

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec


    


def spec_to_mel_torch(spec):
    n_fft = config["filter_length"]
    sampling_rate = config["sampling_rate"]
    n_mel_channels = 80
    mel_fmin =  0.0
    mel_fmax =  22050
    global __mel_basis
    dtype_device = str(spec.dtype) + '_' + str(spec.device)
    fmax_dtype_device = str(mel_fmax) + '_' + dtype_device
    if fmax_dtype_device not in __mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax)
        __mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=spec.dtype, device=spec.device)
    spec = torch.matmul(__mel_basis[fmax_dtype_device], spec)
    spec = __spectral_normalize_torch(spec)
    return spec


def spectrogram_torch(y, center=False):
    n_fft = config["filter_length"]
    sampling_rate = config["sampling_rate"]
    hop_length = config["hop_length"]
    win_length = 2048
    return __spectrogram_torch(y, n_fft, sampling_rate, hop_length, win_length, center)

def mel_spectrogram_torch(y, center=False):
    spec = spectrogram_torch(y,  center)
    spec = spec_to_mel_torch(spec)
    return spec
