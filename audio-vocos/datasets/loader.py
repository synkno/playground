import os
import random

import numpy as np
import torch
import torch.utils.data
from torch.nn import functional as F

from config import config
from public.toolkit.io import read_json

import multiprocessing
from torch.utils.data import DataLoader
from scipy.io.wavfile import read
from collections import defaultdict


class __DataLoader(torch.utils.data.Dataset):
    def __init__(self, items, duration:float = 2.4):
        self.items = items
        self.duration = duration

        
    def get_audio(self, file_path):
        file_part = file_path[:-4]

        sr, audio = read(file_path)
        if sr != config["sampling_rate"]:
            raise ValueError( "Sample Rate not match. Expect {} but got {} from {}".format( config["sampling_rate"], sr, file_path))
        if audio.dtype != np.float32:
            wav = audio / 32768.0
        else:
            wav = audio

        hl = config["hop_length"]
        sr = config["sampling_rate"]

        mel_spec = np.load(file_part + ".mel_spec.npy")
        wav_length = len(wav)

        segment_length = int(self.duration * sr) if self.duration > 0 else wav_length
        segment_length = segment_length // hl * hl
        if segment_length > wav_length:
            wav = np.pad(wav, (0, int(segment_length - wav_length)))
            wav_length = len(wav)

        start = random.randint(0, wav_length - segment_length)
        end = start + segment_length

        return wav[start:end], mel_spec[:, :, start//hl:end//hl + 1]

    def __getitem__(self, index):
        item = self.items[index]
        return  self.get_audio(item)
    def __len__(self):
        return len(self.items)


class __DataCollate:
    def __call__(self, batch):
        wavs = np.array([b[0] for b in batch])
        mel_specs = np.array([b[1][0] for b in batch])
        return wavs, mel_specs
    

def create_loader(
    wav_files:list, 
    num_workers:int, 
    batch_size:int = 1,
    duration:float = 0,
):
    ds = __DataLoader(wav_files, duration=duration)
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
        batch_size=batch_size, duration=config["train_duration"]
    )
    eval_loader = None
    if rank == 0:
        eval_loader = create_loader(
            read_lines(val_file, False),
            batch_size=1,
            num_workers=1, duration=config["eval_duration"]
        )
    return train_loader, eval_loader

