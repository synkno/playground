from public.toolkit.io import read_json, read_str, save_json, log
import shutil
import os
from pydub import AudioSegment
import re
import numpy as np
import soundfile
import soxr
import torch

def split_audio_by_lrc(music_dir:str, out_dir:str, is_recreate:bool = False, separated_dir:str = None, min_duration:float = 0):
    if os.path.exists(out_dir) and not is_recreate:
        return
        
    data = read_json(os.path.join(music_dir, "data.json"))
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    
    for pi, item in enumerate(data):
        lrc = read_str(os.path.join(music_dir, f"{item['id']}.lrc")).splitlines()
        if separated_dir is not None:
            singing_file = os.path.join(separated_dir, f"{item['id']}_(Vocals)_model_bs_roformer_ep_317_sdr_12.wav")
        else:
            singing_file = os.path.join(music_dir, f"{item['id']}.wav")
            
        if not os.path.exists(singing_file):
            log(f"{singing_file} not found!")
            continue
        segs = []
        for line in lrc:
            match = re.search(r'\[(\d+),(\d+)\]', line)
            if not match: continue
            timestart = int(match.group(1))
            timeend = timestart + int(match.group(2))
            text = re.sub(r'<[^>]*>', '', line)
            text = text[text.find("]") + 1 : ].strip()
            segs.append({"start" : timestart, "text" : text, "end" : timeend})

        audio = AudioSegment.from_wav(singing_file)
        new_segs = []
        index = 0
        while index < len(segs):
            duration = 0
            texts = []
            clip_start = None
            clip_end = None
            while index < len(segs):
                seg = segs[index]
                index += 1
                start,end = seg["start"], seg["end"]
                if clip_start is None: clip_start = start
                clip_end = end
                texts.append(seg["text"])
                if (clip_end - clip_start) > min_duration: break
            if (clip_end - clip_start) < min_duration: continue

            out_file = f"{out_dir}{item['id']}-{len(new_segs)}.wav"
            audio[clip_start:clip_end].export(out_file, format="wav")

            new_segs.append({
                "index" : len(new_segs), 
                "start" : clip_start, 
                "end" : clip_end,  
                "text" : " ".join(texts)
            })
        solo_sing = {
            "id" : item['id'],
            "name" : item["name"],
            "segs" : new_segs
        }
        save_json(f"{out_dir}{item['id']}.json", solo_sing)
        log(f"{(pi + 1)}/{len(data)} processed")
    log("compeleted")

def audio_volume_normalize(audio:np.ndarray, coeff:float = 0.2)->np.ndarray:
    temp = np.sort(np.abs(audio))
    if temp[-1] < 0.1:
        scaling_factor = max(temp[-1], 1e-3)
        audio = audio/scaling_factor * 0.1
        #如果音频的最大值都小于0.1，那么先把它放大到0.1

    temp = temp[temp > 0.01]
    L = temp.shape[0]
    #去掉小于0.01的值，要是样本数量小于10个，就直接返回
    if L <= 10:
        return audio
    
    volume = np.mean(temp[int(0.9 * L) : int(0.99 * L)])
    audio = audio * np.clip(coeff/volume, a_min=0.1, a_max=19)
    #volume取09-0.99的音量，用coeff/volume 缩放到 coeff 附近
    #coeff=0.2 标准语音或背景音，coeff=0.5 偏响亮但安全， 
    #coeff=0.05 非常轻柔的背景音. coeff=0.8 接近最大值，需谨慎使用.
    max_value = np.max(np.abs(audio))
    if max_value > 1:
        audio = audio/max_value
        #防止超过1 音量失真。
    return audio

def load_audio(file:str, sampling_rate:int = None, volume_normalize: bool = False)->np.ndarray:
    audio, sr = soundfile.read(file)
    if sampling_rate is not None and sr != sampling_rate:
        audio = soxr.resample(audio, sr, sampling_rate, quality="VHQ")
        sr = sampling_rate
    if volume_normalize:
        audio = audio_volume_normalize(audio)
    return audio


def stft(
    x: torch.Tensor,
    fft_size: int,
    hop_size: int,
    win_length: int,
    window: str,
    use_complex: bool = False,
) -> torch.Tensor:
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """

    x_stft = torch.stft(
        x, fft_size, hop_size, win_length, window.to(x.device), return_complex=True
    )

    # clamp is needed to avoid nan or inf
    if not use_complex:
        return torch.sqrt(
            torch.clamp(x_stft.real**2 + x_stft.imag**2, min=1e-7, max=1e3)
        ).transpose(2, 1)
    else:
        res = torch.cat([x_stft.real.unsqueeze(1), x_stft.imag.unsqueeze(1)], dim=1)
        res = res.transpose(2, 3)  # [B, 2, T, F]
        return res
