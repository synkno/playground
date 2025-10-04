import os
import random

import numpy as np
import torch
import torch.utils.data
from torch.nn import functional as F

from .mel_spec import spectrogram_torch, config
from public.toolkit.io import read_str, read_json

import multiprocessing
from torch.utils.data import DataLoader
from scipy.io.wavfile import read

def _expand_2d(content, target_len):
    h, src_len = content.shape
    x_new = np.linspace(0, 1, target_len)
    idx = np.round(x_new * (src_len - 1)).astype(int)
    return content[:, idx]



class __AudioLoader(torch.utils.data.Dataset):
    def __init__(self, audiopaths, all_in_mem: bool = False):
        self.audiopaths = audiopaths
        self.hop_length = config["hop_length"]
        self.spk_map = config["spk_map"]
        random.seed(1234)
        random.shuffle(self.audiopaths)
        
        self.all_in_mem = all_in_mem
        if self.all_in_mem:
            self.cache = [self.get_audio(p) for p in self.audiopaths]

    def get_audio(self, file_path):
        wav_file = os.path.basename(file_path)[:-4]
        source_dir = "../audio-data/vits/44k-vecs/"

        f0_file = os.path.join(source_dir, wav_file + ".f0.npy")
        spk_file = os.path.join(source_dir, wav_file + ".spk.npy")
        vec_file = os.path.join(source_dir, wav_file + ".vec.npy")
        spec_file = os.path.join(source_dir, wav_file + ".spec.npy")

        sampling_rate, audio = read(file_path)
        if sampling_rate != config["sampling_rate"]:
            raise ValueError( "Sample Rate not match. Expect {} but got {} from {}".format( config["sampling_rate"], sampling_rate, file_path))
        audio_norm = audio / 32768.0

        f0, uv = np.load(f0_file,allow_pickle=True)
        spk = np.load(spk_file)
        vec = np.load(vec_file)
        spec = np.load(spec_file)

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
        if self.all_in_mem:
            return self.random_slice(*self.cache[index])
        else:
            return self.random_slice(*self.get_audio(self.audiopaths[index]))

    def __len__(self):
        return len(self.audiopaths)


class __AudioCollate:
    def __call__(self, batch):
        batch = [b for b in batch if b is not None]

        input_lengths = np.array([x[0].shape[1] for x in batch])
        ids_sorted_decreasing = np.argsort(-input_lengths)  # 负号表示降序
        input_lengths = input_lengths[ids_sorted_decreasing]

        max_vec_len = max([x[0].shape[1] for x in batch])
        max_wav_len = max([x[3].shape[0] for x in batch])

        lengths = np.zeros(len(batch), dtype=np.int64)

        vec_padded = np.zeros((len(batch), batch[0][0].shape[0], max_vec_len), dtype=np.float32)
        f0_padded = np.zeros((len(batch), max_vec_len), dtype=np.float32)
        spec_padded = np.zeros((len(batch), batch[0][2].shape[0], max_vec_len), dtype=np.float32)
        wav_padded = np.zeros((len(batch), 1, max_wav_len), dtype=np.float32)
        spk_padded = np.zeros((len(batch),  batch[0][4].shape[1]), dtype=np.float32)
        uv_padded = np.zeros((len(batch), max_vec_len), dtype=np.float32)

        for i, idx in enumerate(ids_sorted_decreasing):
            row = batch[idx]

            vec = row[0]
            vec_padded[i, :, :vec.shape[1]] = vec
            lengths[i] = vec.shape[1]

            f0 = row[1]
            f0_padded[i, :f0.shape[0]] = f0

            spec = row[2]
            spec_padded[i, :, :spec.shape[1]] = spec

            wav = row[3]
            wav_padded[i, :, :wav.shape[0]] = wav

            spk_padded[i, :] = row[4]

            uv = row[5]
            uv_padded[i, :uv.shape[0]] = uv

        return vec_padded, f0_padded, spec_padded, wav_padded, spk_padded, lengths, uv_padded

def create_loader(wav_files:list, num_workers:int, all_in_mem:bool = False, pin_memory:bool = False, batch_size:int = 1):
    ds = __AudioLoader(wav_files, all_in_mem=all_in_mem)
    loader = DataLoader(
        ds, num_workers=num_workers, shuffle=False,
        batch_size=batch_size, pin_memory=pin_memory,
        drop_last=False, collate_fn=__AudioCollate()
    )
    return loader

def create_loaders(rank:int, world_size:int, batch_size:int):
    all_in_mem = False
    def read_lines(file:str, distributed = True):
        items = [it["wav"] for it in read_json(file)]
        if distributed: return [items[i] for i in range(rank, len(items), world_size)]
        return items

    num_workers = 5 if multiprocessing.cpu_count() > 4 else multiprocessing.cpu_count()
    if all_in_mem: num_workers = 0
    train_loader = create_loader(
        read_lines("../audio-data/vits/train.json"),
        num_workers=num_workers,
        all_in_mem=all_in_mem,
        pin_memory=True,
        batch_size=batch_size
    )
    eval_loader = None
    if rank == 0:
        eval_loader = create_loader(
            read_lines("../audio-data/vits/val.json", False),
            all_in_mem=all_in_mem,
            batch_size=1,
            pin_memory=False,
            num_workers=1
        )
    return train_loader, eval_loader

