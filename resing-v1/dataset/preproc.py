from public.io import log, save_json, read_json
from public.obj import Num
from config import config
import torch
import librosa
import soundfile
import soxr
import numpy as np
import os
from scipy.io import wavfile
from random import shuffle

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
    from .features.dio_f0 import DioF0
    from .features.hubert_vec import HubertVec
    from .features.spk_embed import SpkEmbed
    from .features.mel_spec import MelSpectrogram

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sr = config["sampling_rate"]
    hl = config["hop_length"]
    n_fft = config["filter_length"]

    global __cache
    if "completed" not in __cache:
        
        __cache["content_embed"] = HubertVec(device)
        __cache["spk_embed"] = SpkEmbed(device, sample_rate=16000)
        __cache["dio_f0"] =  DioF0(hop_length=hl, f0_min=50,f0_max=1100, sampling_rate=sr)
        __cache["melspec"] = MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hl, device=device)
        __cache["completed"] = True
    
    content_embed = __cache["content_embed"]
    spk_embed = __cache["spk_embed"]
    dio_f0 = __cache["dio_f0"]
    melspec = __cache["melspec"]

    wav = load_audio(file, sr, volume_normalize=True)
    wav = wav.astype(np.float32) 
    f0,uv = dio_f0.compute_f0_uv(  wav  )

    wav16k = soxr.resample(wav, sr, 16000, quality="VHQ")
    wav16k = torch.tensor(wav16k, device=device, dtype=torch.float32)
    content = content_embed.get_embedding(wav16k)#1x768x400
    content = content.cpu().detach().numpy()


    spk_vec:torch.Tensor = spk_embed.get_embedding(wav16k.unsqueeze(0)) #1x192
    spk_vec = spk_vec.cpu().detach().numpy()

    audio_norm = torch.tensor(wav, dtype=torch.float32, device=device)
    audio_norm = audio_norm.unsqueeze(0)
    spec = melspec.get_spec(audio_norm ).cpu().detach().numpy()

    return wav, f0, uv, content, spk_vec, spec

def preproc_folder(in_dir:str, out_dir:str):
    os.makedirs(out_dir, exist_ok=True)

    
    sr = config["sampling_rate"]
    files = [file for file in os.listdir(in_dir) if file.endswith(".wav")]
    log(f"start preproc folder {in_dir}")
    for i, file in enumerate(files):
        
        src_file = os.path.join(in_dir, file)
        out_file = os.path.join(out_dir, file)

        file_name = file[:-4]
        f0_file = os.path.join(out_dir, file_name + ".dio_f0.npy")
        spk_file = os.path.join(out_dir, file_name + ".sv_spk.npy")
        vec_file = os.path.join(out_dir, file_name + ".hubert_vec.npy")
        spec_file = os.path.join(out_dir, file_name + ".wav_spec.npy")

        duration =  get_wave_duration(src_file)
        if duration < 7 or duration > 18:
            continue

        
        recreate = next((True for file in [
            out_file, f0_file, spk_file, vec_file, spec_file
        ] if not os.path.exists(file) ), False)

        if not recreate: 
            continue
        try:
            wav, f0, uv, content, spk_vec, spec = preproc_file(src_file)

            np.save(f0_file, np.asanyarray((f0,uv),dtype=object))
            np.save(vec_file, content)
            np.save(spec_file, spec)
            np.save(spk_file, spk_vec)
            wavfile.write(out_file, sr, wav)
            log(f"processing {(i+1)}/{len(files)} ... ")
        except Exception as e:
            print(f"error at {src_file} {e}")
        #if i > 84590: break
    log("preproc folder  completed!")


def create_datasets(wav_dir:str):
    wav_files = [file for file in os.listdir(wav_dir) if file.endswith(".wav")]
    songs = [it for items in read_json(os.path.join(wav_dir, "..", "data.json")) for it in items]
    all_data = []
    for file in wav_files:
        sa = file.split("-")
        id = sa[0]
        song = next((it for it in songs if it["id"] == id), None)
        all_data.append(song | { "file" : os.path.join(wav_dir, file), "index" :  sa[1][0:-4]})

    data_map = {}
    singer_id = 0
    song_id = 0
    for it in all_data:
        if it["singer"] not in data_map:
            data_map[it["singer"]] = {"id" : singer_id, "songs" : {}}
            singer_id += 1
        singer = data_map[it["singer"]]
        it["singer_id"] = singer["id"]
        if it["name"] not in singer["songs"]:
            singer["songs"][it["name"]] = {"id" : song_id, "segs" : []}
            song_id += 1
        it["song_id"] = singer["songs"][it["name"]]["id"]
        singer["songs"][it["name"]]["segs"].append(it["index"])
    
    shuffle(all_data)

    train_file = os.path.join(wav_dir, "..", "train.json")
    val_file =  os.path.join(wav_dir, "..", "val.json")
    map_file =  os.path.join(wav_dir, "..", "map.json")

    save_json(train_file, all_data[:-10])
    save_json(val_file, all_data[-10:])
    save_json(map_file, data_map | {"count" : {"singers" : singer_id, "songs" : song_id}})

def create_spk_emb(wav_dir:str):
    from .features.spk_embed import SpkEmbed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    spk_emb = SpkEmbed(device, sample_rate=16000)
    
    wav_files = [ os.path.join(wav_dir, file) for file in os.listdir(wav_dir) if file.endswith(".wav")]
    for file in wav_files:

        spk_file = file[:-4] + ".sv_spk.npy"
        if os.path.exists(spk_file): continue

        wav = load_audio(file, 16000, volume_normalize=True)
        wav = torch.tensor(wav, device=device, dtype=torch.float32)
        spk_vec:torch.Tensor = spk_emb.get_embedding(wav.unsqueeze(0)) #1x192
        spk_vec = spk_vec.cpu().detach().numpy()
        np.save(spk_file, spk_vec)









        



