

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
from .config import config
import random

__split_dir = f"/code/data/custom-datasets/kuaou/split-8.0-singer-{(config['sampling_rate']//1000)}k/"
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
        wav = datasets.load_audio(file, config["sampling_rate"], volume_normalize=True)
        wavfile.write(
            file,
            config["sampling_rate"],
            (wav * np.iinfo(np.int16).max).astype(np.int16)
        )
        log(f"procossing ... {(i + 1)}/{len(files)}")

class Volume_Extractor:
    def __init__(self, hop_size = 512):
        self.hop_size = hop_size
        
    def extract(self, audio): # audio: 2d tensor array
        if not isinstance(audio,torch.Tensor):
           audio = torch.Tensor(audio)
        n_frames = int(audio.size(-1) // self.hop_size)
        audio2 = audio ** 2
        #对波形平方，得到功率（能量）
        audio2 = torch.nn.functional.pad(audio2, (int(self.hop_size // 2), int((self.hop_size + 1) // 2)), mode = 'reflect')
        #在前后补边，保证帧切分时对齐。
        volume = torch.nn.functional.unfold(audio2[:,None,None,:],(1,self.hop_size),stride=self.hop_size)[:,:,:n_frames].mean(dim=1)[0]
        #unfold: 把信号切成一帧一帧的窗口（每帧长度 = hop_size）. mean(dim=1)：对每一帧求平均能量。
        volume = torch.sqrt(volume)
        #从功率取平方根，得到 RMS 值。
        return volume
    
def __cache_vecs(out_dir:str):
    train_file = os.path.join(out_dir, "diff", "train.json")
    val_file =  os.path.join(out_dir, "diff", "val.json")
    if os.path.exists(train_file) and os.path.exists(val_file):
        return

    source_dir = os.path.join(out_dir, "diff", "44k-vecs")
    os.makedirs(source_dir, exist_ok=True)
    json_files = glob(f"{__split_dir}/*.json")

    content_embed = None
    spk_embed = None
    dio_f0 = None
    mel_extractor = None

    sr = config["sampling_rate"]
    hl = config["hop_length"]

    all_data = []

    volume_extractor = Volume_Extractor(hl)

    for i, json_file in enumerate(json_files):
        data = read_json(os.path.join(__split_dir, json_file))
        singer = str(data["name"]).split(" - ")[0]
        if "、" in singer or " " in singer: continue
        for seg in data["segs"]:
            wav_file = f'{data["id"]}-{seg["index"]}'

            f0_file = os.path.join(source_dir, wav_file + ".f0.npy")
            spk_file = os.path.join(source_dir, wav_file + ".spk.npy")
            vec_file = os.path.join(source_dir, wav_file + ".vec.npy")

            vol_file = os.path.join(source_dir, wav_file + ".vol.npy")
            mel_file = os.path.join(source_dir, wav_file + ".mel.npy")

            aug_mel_file = os.path.join(source_dir, wav_file + ".aug_mel.npy")
            aug_vol_file = os.path.join(source_dir, wav_file + ".aug_vol.npy")

            wav_file = os.path.join(__split_dir, wav_file + ".wav")
            all_data.append({
                "wav" : wav_file,
                "singer" : singer,
            })

            if (
                os.path.exists(f0_file) and 
                os.path.exists(spk_file) and
                os.path.exists(vol_file) and
                os.path.exists(mel_file) and
                os.path.exists(aug_mel_file) and
                 os.path.exists(aug_vol_file) and
                os.path.exists(vec_file)
            ):
                continue

            if content_embed is None:
                from public.audio.spk_embed import SpkEmbed
                from public.audio.hubert_vec import HubertVec
                from public.audio.dio_f0 import DioF0
                from vocode.vocoder import Vocoder
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                content_embed = HubertVec(device)
                spk_embed = SpkEmbed(device, sample_rate=16000)
                dio_f0 =  DioF0(hop_length=hl, f0_min=50,f0_max=1100, sampling_rate=sr)
                mel_extractor = Vocoder(device=device)
            
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
            
            volume = volume_extractor.extract(audio_norm)
            np.save(vol_file, volume.to('cpu').numpy())

            mel_t = mel_extractor.extract(audio_norm.to(device), sr)
            mel = mel_t.squeeze().to('cpu').numpy()
            np.save(mel_file, mel)

            max_amp = float(torch.max(torch.abs(audio_norm))) + 1e-5
            max_shift = min(1, np.log10(1/max_amp))#范围是(0到1)
            log10_vol_shift = random.uniform(-1, max_shift)#范围是(-1到1)
            keyshift = random.uniform(-5, 5)
            aug_mel_t = mel_extractor.extract(
                audio_norm * (10 ** log10_vol_shift), sr, keyshift = keyshift
            )#10 ** log10_vol_shift 范围是 (0.1到10)
            #keyshift在提取 mel 特征之前，把音频整体升高或降低若干半音。
            #keyshift = +5 → 把音频升高 5 个半音（大约升高一个纯四度）。, keyshift = -5 → 把音频降低 5 个半音。
                
            aug_mel = aug_mel_t.squeeze().to('cpu').numpy()
            aug_vol = volume_extractor.extract(audio_norm * (10 ** log10_vol_shift))

            np.save(aug_mel_file, np.asanyarray((aug_mel,keyshift),dtype=object))
            np.save(aug_vol_file, aug_vol.to('cpu').numpy())

        log(f"Process dataset {i + 1} of {len(json_files)}")
    
    shuffle(all_data)
    save_json(train_file, all_data[:-20])
    save_json(val_file, all_data[-20:])

def data_prepoc():
    out_dir = "../audio-data/"
    __resample()
    __cache_vecs(out_dir)


