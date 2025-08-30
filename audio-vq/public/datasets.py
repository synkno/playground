
from public.audio_utils import split_audio_by_lrc, load_audio
from public.toolkit.io import read_json, save_json, read_jsonl
from public.toolkit.object import Num
import os
import random
from typing import Dict, Any
import numpy as np
import torch
from operator import itemgetter


class __DatasetConfig:
    sample_rate:int
    max_val_duration:int
    train_duration:int
    hop_length:int
    device:str
    def update(self, 
        sample_rate:int,
        max_val_duration:int,
        train_duration:int,
        hop_length:int,
        device:str,
    ):
        self.sample_rate = sample_rate
        self.max_val_duration = max_val_duration
        self.train_duration = train_duration
        self.hop_length = hop_length
        self.device = device
        

config = __DatasetConfig()

def __split_batch(all_data:list, batch_size:int):
    all_data = [all_data[i: i + batch_size] for i in range(0, len(all_data), batch_size)]
    return [data for data in all_data if len(data) == batch_size]

def __get_sample(file:str, is_train:bool) -> Dict[str, Any]:

    wav = load_audio(file, config.sample_rate, volume_normalize=True)
    wav_length = len(wav)
    wav_length = (wav_length // config.hop_length) * config.hop_length

    duration = config.train_duration
    if not is_train:
        duration = min(wav_length//config.sample_rate, config.max_val_duration)
    segment_length = int(duration * config.sample_rate)
    if segment_length > wav_length:
        wav = np.pad(wav, (0, int(segment_length - wav_length)))
        wav_length = len(wav)
    start_indice = random.randint(0, wav_length - segment_length) if is_train else 0
    end_indice = start_indice + segment_length
    return {
        "wav" : wav, 
        "start" : start_indice,
        "end" : end_indice,
        "length" : wav_length
    }

def update_batch(batch:list,  is_train:bool, return_tensor:bool):
    batch = [
        {
            "index" : it["index"],
            ** __get_sample(it["file"], is_train)
        }  
        for it in batch
    ]
    collate_batch = {
        "is_train" : is_train,
        "index" : [b["index"] for b in batch]
    }
    if return_tensor:
        x = torch.tensor(np.array([
            wav[start:end] for wav, start, end in map(itemgetter("wav", "start", "end"), batch)
        ]), dtype=torch.float32).to(config.device)
        collate_batch["wav"] = x
        return collate_batch
    collate_batch["data"] = batch
    return collate_batch

def create_datasets(batch_size:int, data_file:str, type:str = "jay", min_duration:float = 0):
    if os.path.exists(data_file):
        datasets = read_json(data_file)["items"]
        Num.count([it["duration"] for it in datasets], [0, 2.4, 100])
        train_datasets = __split_batch(datasets[:-20], batch_size)
        eval_datasets = __split_batch(datasets[-20:], 1)
        return train_datasets, eval_datasets
    if type == "jay":
        audio_dir = f"/code/data/custom-datasets/kuaou-jay/split-{min_duration}/"
        split_audio_by_lrc("/code/data/custom-datasets/kuaou-jay/song", audio_dir, min_duration=int(min_duration*1000), is_recreate=True)

        json_files = [read_json(os.path.join(audio_dir, file)) for file in os.listdir(audio_dir) if file.endswith(".json")]
        datasets = [{
            "file": os.path.join(audio_dir, f"{data['id']}-{seg['index']}.wav"),
            "name" : data["name"],
            "text" :  seg['text'],
            "type" : "music",
            "duration" :  (seg["end"] - seg["start"])/1000,
        } for data in json_files  for seg in data["segs"]]

        duration = 0
        for it in datasets:
            duration += it["duration"]

        random.shuffle(datasets)
        for i, it in enumerate(datasets): 
            it["index"] = i

        save_json(data_file, {
            "count" : len(datasets),
            "duration" : duration,
            "items":   datasets
        })
        return create_datasets(batch_size, data_file)
    
    datasets = read_jsonl("/code/data/datasets/voxbox_subset/metadata/aishell-3.jsonl")

    duration = 0
    for i, it in enumerate(datasets): 
        it["file"] = "/code/data/datasets/voxbox_subset/extracted/" + str(it["wav_path"]).replace(".flac", ".wav")
        duration += it["duration"]
        it["index"] = i
        
    save_json(data_file, {
        "count" : len(datasets),
        "duration" : duration,
        "items":   datasets
    })
    return create_datasets(batch_size, data_file)

    