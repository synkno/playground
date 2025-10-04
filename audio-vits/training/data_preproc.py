

from public.audio import datasets
from public.toolkit.io import save_json, log, read_json
import os
from scipy.io import wavfile
import numpy as np
import json
import re
import wave
from random import shuffle
from tqdm import tqdm
from glob import glob
import torch
import librosa
from .mel_spec import spectrogram_torch
from .config import config

__sr = config["sampling_rate"]

__split_dir = f"/code/data/custom-datasets/kuaou/split-8.0-singer-{(__sr//1000)}k/"
def __resample():
    if os.path.exists(__split_dir): return
    datasets.split_audio_by_lrc(
        "/code/data/custom-datasets/kuaou/song", 
        __split_dir, 
        min_duration=int(8.0*1000), 
        is_recreate=False,
        separated_dir= "/code/data/custom-datasets/kuaou/separated"
    )
    files = [os.path.join(__split_dir, file) for file in os.listdir(__split_dir) if file.endswith(".wav")]
    
    for i, file in enumerate(files):
        wav = datasets.load_audio(file, __sr, volume_normalize=True)
        wavfile.write(
            file,
            __sr,
            (wav * np.iinfo(np.int16).max).astype(np.int16)
        )
        log(f"procossing ... {(i + 1)}/{len(files)}")

def __cache_vecs(out_dir:str):
    train_file = os.path.join(out_dir, "vits", "train.json")
    val_file =  os.path.join(out_dir, "vits", "val.json")
    if os.path.exists(train_file) and os.path.exists(val_file):
        return

    source_dir = os.path.join(out_dir, "vits", "44k-vecs")
    os.makedirs(source_dir, exist_ok=True)
    json_files = glob(f"{__split_dir}/*.json")

    content_embed = None
    spk_embed = None
    dio_f0 = None

    sr = config["sampling_rate"]
    hl = config["hop_length"]

    all_data = []

    for i, json_file in enumerate(json_files):
        data = read_json(os.path.join(__split_dir, json_file))
        singer = str(data["name"]).split(" - ")[0]
        if "、" in singer or " " in singer: continue
        for seg in data["segs"]:
            wav_file = f'{data["id"]}-{seg["index"]}'

            f0_file = os.path.join(source_dir, wav_file + ".f0.npy")
            spk_file = os.path.join(source_dir, wav_file + ".spk.npy")
            vec_file = os.path.join(source_dir, wav_file + ".vec.npy")
            spec_file = os.path.join(source_dir, wav_file + ".spec.npy")

            wav_file = os.path.join(__split_dir, wav_file + ".wav")
            all_data.append({
                "wav" : wav_file,
                "singer" : singer,
            })

            if (
                os.path.exists(f0_file) and 
                os.path.exists(spk_file) and
                os.path.exists(spec_file) and
                os.path.exists(vec_file)
            ):
                continue

            if content_embed is None:
                from public.audio.spk_embed import SpkEmbed
                from public.audio.hubert_vec import HubertVec
                from public.audio.dio_f0 import DioF0
                device = "cuda" if torch.cuda.is_available() else "cpu"
                content_embed = HubertVec(device)
                spk_embed = SpkEmbed(device, sample_rate=16000)
                dio_f0 =  DioF0(hop_length=hl, f0_min=50,f0_max=1100, sampling_rate=sr)
            
            wav, sr = librosa.load(wav_file, sr=sr)
            f0,uv = dio_f0.compute_f0_uv(  wav  )
            np.save(f0_file, np.asanyarray((f0,uv),dtype=object))
            
            wav16k = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            wav16k = torch.from_numpy(wav16k).to(device)
            c = content_embed.get_embedding(wav16k)#1x768x400
            np.save(vec_file, c.cpu().detach().numpy())

            spk_vec:torch.Tensor = spk_embed.get_embedding(wav16k.unsqueeze(0)) #1x192
            np.save(spk_file, spk_vec.cpu().detach().numpy())
            
            audio_norm = torch.FloatTensor(wav)
            audio_norm = audio_norm.unsqueeze(0)
            spec = spectrogram_torch(audio_norm )
            spec = torch.squeeze(spec, 0)
            np.save(spec_file, spec.numpy())
        log(f"Process dataset {i + 1} of {len(json_files)}")
    
    shuffle(all_data)
    save_json(train_file, all_data[:-20])
    save_json(val_file, all_data[-20:])

def data_prepoc():
    out_dir = "../audio-data/"
    __resample()
    __cache_vecs(out_dir)


