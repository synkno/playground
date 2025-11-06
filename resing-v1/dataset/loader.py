import os
import random

import numpy as np
import torch
import torch.utils.data
from torch.nn import functional as F

from public.io import read_str, read_json
from config import config

import multiprocessing
from torch.utils.data import DataLoader
from scipy.io.wavfile import read
from collections import defaultdict

def _expand_2d(content, target_len):
    h, src_len = content.shape
    x_new = np.linspace(0, 1, target_len)
    idx = np.round(x_new * (src_len - 1)).astype(int)
    return content[:, idx]



class __DataLoader(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items
        self.hop_length = config["hop_length"]
        self.spk_emb = None
        
    def get_spkemd(self, item:dict):
        file = item["speaker"]["file"]  if "speaker" in item else item["file"]
        spk_file = file[:-4] + ".sv_spk.npy"
        return np.load(spk_file)
            

        
    def get_audio(self, item):
        file_path = item["file"]
        file_part = file_path[:-4]

        f0_file = file_part + ".dio_f0.npy"
        vec_file = file_part + ".hubert_vec.npy"
        spec_file = file_part + ".wav_spec.npy"

        sampling_rate, audio = read(file_path)
        if sampling_rate != config["sampling_rate"]:
            raise ValueError( "Sample Rate not match. Expect {} but got {} from {}".format( config["sampling_rate"], sampling_rate, file_path))
        if audio.dtype != np.float32:
            audio_norm = audio / 32768.0
        else:
            audio_norm = audio

        f0, uv = np.load(f0_file,allow_pickle=True)
        spk = self.get_spkemd(item)
        vec = np.load(vec_file)
        spec = np.load(spec_file)[0]

        return vec, f0, spec, audio_norm, spk, uv

    def random_slice(self, vec:np.ndarray, f0:np.ndarray, spec:np.ndarray, audio_norm:np.ndarray, spk:np.ndarray, uv:np.ndarray):
        vec:np.ndarray = _expand_2d(vec[0], f0.shape[0])
        lmin = min(vec.shape[-1], spec.shape[-1])

        spec, vec, f0, uv = spec[:, :lmin], vec[:, :lmin], f0[:lmin], uv[:lmin]
        audio_norm = audio_norm[:lmin * self.hop_length]

        if spec.shape[1] > 800:
            start = random.randint(0, spec.shape[1]-800)
            end = start + 790
            spec, vec, f0, uv = spec[:, start:end], vec[:, start:end], f0[start:end], uv[start:end]
            audio_norm = audio_norm[start * self.hop_length : end * self.hop_length]
        return vec, f0, spec, audio_norm, spk, uv

    def __getitem__(self, index):
        item = self.items[index]
        vec, f0, spec, audio_norm, spk, uv =  self.random_slice(*self.get_audio(item))
        return {"content" : vec, "f0" : f0, "spec" : spec, "wav" : audio_norm, "spk_emb" : spk, "uv" : uv, "data" : item}
    def __len__(self):
        return len(self.items)


class __DataCollate:
    def __call__(self, batch):
        batch = [b for b in batch if b is not None]

        input_lengths = np.array([x["content"].shape[1] for x in batch])
        ids_sorted_decreasing = np.argsort(-input_lengths)  # 负号表示降序
        input_lengths = input_lengths[ids_sorted_decreasing]

        max_vec_len = max([x["content"].shape[1] for x in batch])
        max_wav_len = max([x["wav"].shape[0] for x in batch])

        lengths = np.zeros(len(batch), dtype=np.int64)

        vec_padded = np.zeros((len(batch), batch[0]["content"].shape[0], max_vec_len), dtype=np.float32)
        f0_padded = np.zeros((len(batch), max_vec_len), dtype=np.float32)
        spec_padded = np.zeros((len(batch), batch[0]["spec"].shape[0], max_vec_len), dtype=np.float32)
        wav_padded = np.zeros((len(batch), 1, max_wav_len), dtype=np.float32)
        spk_padded = np.zeros((len(batch),  batch[0]["spk_emb"].shape[1]), dtype=np.float32)
        uv_padded = np.zeros((len(batch), max_vec_len), dtype=np.float32)

        for i, idx in enumerate(ids_sorted_decreasing):
            row = batch[idx]

            vec = row["content"]
            vec_padded[i, :, :vec.shape[1]] = vec
            lengths[i] = vec.shape[1]

            f0 = row["f0"]
            f0_padded[i, :f0.shape[0]] = f0

            spec = row["spec"]
            spec_padded[i, :, :spec.shape[1]] = spec

            wav = row["wav"]
            wav_padded[i, :, :wav.shape[0]] = wav

            spk_padded[i, :] = row["spk_emb"]

            uv = row["uv"]
            uv_padded[i, :uv.shape[0]] = uv
        
        return { 
            "content"   :  vec_padded, 
            "f0"        :  f0_padded,
            "spec"      :  spec_padded,
            "wav"       :  wav_padded,
            "spk_emb"   :  spk_padded,
            "lengths"   :  lengths, 
            "uv"        :  uv_padded,
            "data"      : [row["data"] for row in batch]
        }

def create_loader(
    wav_files:list, 
    num_workers:int, 
    batch_size:int = 1,
):
    ds = __DataLoader(wav_files)
    loader = DataLoader(
        ds, num_workers=num_workers, shuffle=False,
        batch_size=batch_size, pin_memory=False,
        drop_last=False, collate_fn=__DataCollate()
    )
    return loader

def create_loaders(data_dir:str, rank:int, world_size:int, batch_size:int):
    train_file = os.path.join(data_dir, "train.json")
    val_file =  os.path.join(data_dir, "val.json")

    def read_lines(file:str, distributed = True):
        items = [it for it in read_json(file)]
        if distributed: return [items[i] for i in range(rank, len(items), world_size)]
        return items

    num_workers = 5 if multiprocessing.cpu_count() > 4 else multiprocessing.cpu_count()
    train_loader = create_loader(
        read_lines(train_file),
        num_workers=num_workers if not config["debug"] else 0,
        batch_size=batch_size
    )
    eval_loader = None
    if rank == 0:
        eval_loader = create_loader(
            read_lines(val_file, False),
            batch_size=1,
            num_workers=1
        )
    return train_loader, eval_loader

