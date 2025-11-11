from config import config
import torch
import librosa
import soundfile
import soxr
import numpy as np
import os
from scipy.io import wavfile
from random import shuffle
from public.toolkit.io import log, save_json

import wave


def get_wave_duration(file:str):
    info = soundfile.info(file)
    duration = info.frames / info.samplerate
    return duration

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
    if audio.ndim == 2 and audio.shape[1] == 2:
        audio = audio.mean(axis=1)
    if sampling_rate is not None and sr != sampling_rate:
        audio = soxr.resample(audio, sr, sampling_rate, quality="VHQ")
        sr = sampling_rate
    if volume_normalize:
        audio = audio_volume_normalize(audio)
    return audio

__cache = {}
def preproc_file(file:str):
    from modelling.mel_spec import MelSpec

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sr = config["sampling_rate"]
    global __cache
    if "completed" not in __cache:
        __cache["melspec"] = MelSpec(
            sample_rate=sr, n_fft=config["n_fft"], hop_length=config["hop_length"], n_mels=config["n_mels"], padding=config["padding"]
        ).to(device)
        __cache["completed"] = True
    
    melspec = __cache["melspec"]

    wav = load_audio(file, sr, volume_normalize=True)
    wav = wav.astype(np.float32) 

    audio_norm = torch.tensor(wav, dtype=torch.float32, device=device)#215280
    audio_norm = audio_norm.unsqueeze(0)
    mel_spec = melspec.forward(audio_norm ).cpu().detach().numpy()#1x100x841

    return wav, mel_spec

def preproc_folder(in_dir:str, out_dir:str):
    os.makedirs(out_dir, exist_ok=True)
    train_file = os.path.join(out_dir, "..", "train.json")
    val_file =  os.path.join(out_dir, "..", "val.json")


    sr = config["sampling_rate"]
    files = [file for file in os.listdir(in_dir) if file.endswith(".wav")]
    log(f"start preproc folder {in_dir}")
    wav_files = []
    for i, file in enumerate(files):
        
        src_file = os.path.join(in_dir, file)
        out_file = os.path.join(out_dir, file)

        file_name = file[:-4]
        mel_file = os.path.join(out_dir, file_name + ".mel_spec.npy")
        
        recreate = next((True for file in [
            out_file, mel_file
        ] if not os.path.exists(file) ), False)

        if not recreate: 
            continue
        try:
            wav, mel_spec = preproc_file(src_file)
            np.save(mel_file, mel_spec)
            wavfile.write(out_file, sr, wav)
            wav_files.append(out_file)
            log(f"processing {(i+1)}/{len(files)} ... ")
        except Exception as e:
            print(f"error at {src_file} {e}")
        #if i > 84590: break
    shuffle(wav_files)
    save_json(train_file, wav_files[:-10])
    save_json(val_file, wav_files[-10:])
    log("preproc folder  completed!")











        



