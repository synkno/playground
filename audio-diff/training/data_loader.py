import os
import random

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from public.toolkit.io import read_json

from .config import config



def _expand_2d(content, target_len):
    h, src_len = content.shape
    x_new = np.linspace(0, 1, target_len)
    idx = np.round(x_new * (src_len - 1)).astype(int)
    return content[:, idx]


class AudioDataset(Dataset):
    def __init__(
        self,
        files,
        is_train:bool,
    ):
        super().__init__()
        self.files = files
        self.is_train = is_train
           

    def __getitem__(self, file_idx):
        file = self.files[file_idx]
        return self.__get_data(file)

    def __get_data(self, wav_file:str):
        file_name = os.path.basename(wav_file)[:-4]
        source_dir = "../audio-data/diff/44k-vecs/"

        f0_file = os.path.join(source_dir, file_name + ".f0.npy")
        spk_file = os.path.join(source_dir, file_name + ".spk.npy")
        vec_file = os.path.join(source_dir, file_name + ".vec.npy")

        vol_file = os.path.join(source_dir, file_name + ".vol.npy")
        mel_file = os.path.join(source_dir, file_name + ".mel.npy")

        aug_mel_file = os.path.join(source_dir, file_name + ".aug_mel.npy")
        aug_vol_file = os.path.join(source_dir, file_name + ".aug_vol.npy")

        resolution = config["hop_length"]/config["sampling_rate"]
    
        duration = librosa.get_duration(path= wav_file, sr = config["sampling_rate"])
        seg_secs = config["train_duraton"] if self.is_train else duration

        idx_from = 0 if not self.is_train else random.uniform(0, duration - seg_secs - 0.1)

        start_frame = int(idx_from / resolution)
        units_frame_len = int(seg_secs / resolution)
        aug_flag = random.choice([True, False])

        aug_shift = 0
        if aug_flag:
            mel, aug_shift = np.load(aug_mel_file, allow_pickle=True)
        else:
            mel = np.load(mel_file)
        
        volume = np.load(aug_vol_file if aug_flag else vol_file)
        
        f0, _ = np.load(f0_file, allow_pickle=True)
        f0 = f0.astype(float)
        units = np.load(vec_file)
        units:np.ndarray = _expand_2d(units[0], f0.shape[0])

        units = units[:, start_frame : start_frame + units_frame_len]
        f0 = 2 ** (aug_shift / 12) * f0[start_frame : start_frame + units_frame_len]
        mel = mel[start_frame : start_frame + units_frame_len, :]
        volume = volume[start_frame : start_frame + units_frame_len]

        units = units.T             #172x768
        f0 = f0.reshape(-1, 1)       #172x1
        volume = volume.reshape(-1, 1)  #172x1

        aug_shift = np.array([[aug_shift]])

        spk_emb = np.load(spk_file)

        return dict(mel=mel, f0=f0, volume=volume, units=units, spk_emb=spk_emb, aug_shift=aug_shift, wav_file=wav_file)

    def __len__(self):
        return len(self.files)



class AudioCollate:
    def __call__(self, batch):
        out = {}
        for key in batch[0].keys():
            if isinstance(batch[0][key], np.ndarray):
                out[key] = np.stack([d[key] for d in batch], axis=0)
            else:
                 out[key] = [d[key] for d in batch]
        return out



def create_loader(files:list, is_train:bool, batch_size:int,num_workers:int, shuffle = True):
    data_train = AudioDataset(
        files= files,
        is_train=is_train
    )
    loader_train = torch.utils.data.DataLoader(
        data_train ,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers= num_workers,
        persistent_workers = num_workers > 0,
        pin_memory=True,
        collate_fn=AudioCollate()
    )
    return loader_train


def create_loaders(rank:int, world_size:int, batch_size:int):
    def read_lines(file:str, distributed = True):
        items = [it["wav"] for it in read_json(file)]
        if distributed: return [items[i] for i in range(rank, len(items), world_size)]
        return items
    loader_train = create_loader(
        files= read_lines("/code/playground/audio-data/diff/train.json"),
        is_train=True,
        batch_size=batch_size,
        num_workers=0 if config["debug"] else  5,
        shuffle=True
    )
    loader_valid = create_loader(
        files = read_lines("/code/playground/audio-data/diff/val.json"),
        is_train=False,
        batch_size=1,
        num_workers=0,
        shuffle=False
    )
    return loader_train, loader_valid 