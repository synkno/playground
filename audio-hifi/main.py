
import os
import sys
sys.dont_write_bytecode = True
code_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(code_dir)
sys.path.append(os.path.join(code_dir, "3rd-libs"))



from public.toolkit.io import read_json, save_json, read_jsonl, log
from public.toolkit.nn import size_of_model,start_update_params, stop_update_params, list_grad_params
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import random
from typing import Dict, Any
import numpy as np

from operator import itemgetter
from public.audio import datasets
from public.audio.training_logs import TrainingLogs
from public.audio.training_loss import MultiMelSpecLoss, stft_loss
from public.toolkit.training import WarmupLR

from typing import Tuple

import torch.nn.functional as F

from config import config
from log_melspec import LogMelSpec
from typing import List
from wav_trainer import WavTrainer, WavGen
from wav_disc import WaveDisc

"""
def test():
    from public.audio.hubert_vec import HubertVec
    hvec = HubertVec("cpu")
    wav = datasets.load_audio("/code/data/custom-datasets/kuaou/split-8.0-singer/0a89bea98ff55939959c699b37ca0403-5.wav")
    wav = torch.Tensor(wav)
    r = hvec.get_embedding(wav)
    exit()

test()
"""

def train_model(local_rank: int, out_dir:str, data_file:str, epochs:int, model_file:str = None):
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()

    device = torch.device(f"cuda:{local_rank}")
    is_main_process = local_rank == 0

    sample_rate = 16000
    os.makedirs(out_dir, exist_ok=True)
    datasets.config.update(
        sample_rate=sample_rate,
        max_val_duration=12, 
        train_duration=2.4,
        speaker_duration=6.0,
        hop_length=320,
        device=device,
    )
    train_datasets, eval_datasets = datasets.create_datasets_with_singers(
        batch_size=32,
        data_file=os.path.join(out_dir, "../", data_file),
        min_duration=8.0
    )
    if config["max_datasets_len"] > 0:
        train_datasets = train_datasets[:config["max_datasets_len"] ]

    len_datasets = len(train_datasets)//world_size

    warmup_lr       = {
        "warmup_step" : 500,
        "down_step" : int(epochs * len_datasets * 0.5),
        "min_lr"  : 1e-5,
        "max_lr" :  config["max_lr"]
    }
    loss_weights = {
        "mel_loss" : 1.0,
        "l1_loss" : 1.0,
    }
    if config["use_disc"]:
        loss_weights = {
            "mel_loss" : 15,
            "l1_loss" : 1.0,
            "adv_loss" : 1.0,
            "feature_map_loss" : 2.0
            
        }
    adamw_betas     = [0.8, 0.9]
    scale = (sample_rate//16000)
    mel_loss = MultiMelSpecLoss(
        sample_rate=sample_rate,
        n_mels=[5, 10, 20, 40, 80, 160, 320],
        window_lengths= [x * scale for x in [32, 64, 128, 256, 512, 1024, 2048]] ,
        clamp_eps=1.0e-5,
    )
    mel_spec = LogMelSpec(
        sample_rate=sample_rate,
        n_fft=1024,
        win_length=1024,
        hop_length=config["hop_length"],
        n_mels=128,
        center=True
    )
    wav_gen = WavGen(sample_rate=sample_rate).to(config["dtype"])
    wav_disc = WaveDisc(
        sample_rate=sample_rate,
        periods =  [2, 3, 5, 7, 11],
        bands = [[0.0, 0.1], [0.1, 0.25], [0.25, 0.5], [0.5, 0.75], [0.75, 1.0]],
        stft_params = {
            "fft_sizes": [2048, 1024, 512],
            "hop_sizes": [512, 256, 128],
            "win_lengths": [1200, 600, 300],
            "window": "hann_window",
        }
    ).to(config["dtype"])

    trainer = WavTrainer(
        wav_gen,
        disc= wav_disc if config["use_disc"] else None,
        mel_spec=mel_spec,
        mel_loss=mel_loss,
        sample_rate=sample_rate,
        adamw_betas=adamw_betas,
        warmup_lr=warmup_lr,
        loss_lambdas=loss_weights,
        device=device
    )
    if model_file is not None:
        state = torch.load(model_file)
        #state = { k: v for k, v in state.items()  if not k.startswith("disc.discriminators") }
        keys = trainer.load_state_dict(state, False)
    model = torch.nn.parallel.DistributedDataParallel(trainer, device_ids=[local_rank])
    if is_main_process:
        logs = TrainingLogs(
            optimizers = [p for p in [trainer.gen_optimizer, trainer.disc_optimizer] if p],
            save_dir=out_dir,
            sample_rate=sample_rate,
            log_steps=100
        )
    else:
        logs = None
    
    
    log(f"Start Trainning, epochs: {epochs}, len of data: {len(train_datasets)}, data file: {data_file}, model: {size_of_model(trainer)}, GPU: {local_rank}/{world_size} model_file:{model_file}")
    if is_main_process: log(f"Config: {config}")
    #1248, 20
    #11001, 
    
    
   
    def update_batch_data(data:dict, is_train:bool, epoch:int = 0):
        #return datasets.update_batch( data,  is_train, epoch, False)
        return trainer.prepare_batch_data( data,  is_train, epoch)

    @torch.no_grad()
    def eval():
        if not is_main_process: return
        model.eval()
        for di, data in enumerate(eval_datasets):
            data = update_batch_data(data, False)
            logs_dict, cached_demos = trainer.validation_step(data)
            logs.validation_step(logs_dict, cached_demos)
        logs.validation_end()
        model.train()
    def save():
        if not is_main_process: return
        model_dir = os.path.join(out_dir, "temp-training", f"{logs.current_epoch}-{logs.current_step}")
        os.makedirs(model_dir, exist_ok=True)
        torch.save(trainer.state_dict(), os.path.join(model_dir, "model.torch"))

    
    model.train()
    
    mem_data = [train_datasets[i] for i in range(local_rank, len(train_datasets), world_size)]
    if is_main_process: mem_data += eval_datasets

    trainer.cache_all_data(mem_data)
            
    eval()

    total_steps = len_datasets * epochs
    current_step = 0
    def lerp(start, end, time_start, time_end):
        p = current_step/total_steps
        if p < time_start: return start
        if p > time_end: return end
        p = (p - time_start)/(time_end - time_start)
        return (end - start) * p + start
    
    for epoch in range(epochs):
        for i in range(local_rank, len(train_datasets), world_size):
            data = train_datasets[i]
            data = update_batch_data(data, True, epoch)
            logs_dict = trainer.training_step(data)
            current_step += 1
            if not is_main_process: continue

            logs.training_step(logs_dict, epoch)
            if logs.current_step % 5000 == 0: eval()
            if logs.current_step % 5000 == 0: save()
    eval()
    save()

def train(local_rank:int, world_size:int):
    os.environ["RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] =  str(world_size)
    os.environ["LOCAL_RANK"] =  str(local_rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    out_dir = "../audio-data/"
    train_model(
        local_rank=local_rank,
        out_dir=os.path.join(out_dir, config["out_dir"]), 
        data_file="train_data_singer-8.0.json", 
        epochs=config["epochs"], 
        model_file=config["model_file"]
    )

if __name__ == "__main__":
    import argparse
    import subprocess
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--debug", type=bool, default=False)
    args = parser.parse_args()
    
    world_size = 2
    if args.local_rank != -1:
        train(args.local_rank, world_size)
    elif args.debug:
        train(0, 1)
    else:
        for i in range(world_size):
            subprocess.Popen(["python", "main.py", "--local_rank", str(i)])