
import os
import sys
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from public.toolkit.io import read_json, save_json, read_jsonl, log
from public.toolkit.nn import size_of_model,start_update_params, stop_update_params, list_grad_params
import torch
import torch.nn as nn
import torch.optim as optim
import random
from typing import Dict, Any
import numpy as np

from operator import itemgetter
from public.audio import datasets
from public.audio.training_logs import TrainingLogs
from public.audio.training_loss import MultiMelSpecLoss, stft_loss
from public.toolkit.training import WarmupLR

from typing import Tuple

from audio_vq import AudioVQ, build_model
import torch.nn.functional as F

from torchaudio.transforms import MelSpectrogram
from typing import List


def compute_loss(model: AudioVQ, batch:dict, loss_weights:dict, mel_loss:MultiMelSpecLoss = None):
    audios:torch.Tensor = batch["wav"].unsqueeze(1)
    out = model(audios)

    recons = out["recons"]
    out["l1_loss"] = F.l1_loss(recons, audios)
    out["stft_loss"] = stft_loss(recons, audios)  if mel_loss is None else mel_loss(recons, audios) 
   

    losses = {k : (v * out[k]).mean() for k, v in loss_weights.items() if k in out}
    loss = sum([v for _, v in losses.items()])

    out_dict = {
        "loss": loss,
        "audios" : audios,
        "recons" : recons,
    }
    logs = {
        "loss": loss.float(),
        "perplexity": out["perplexity"],
        "active_num": out["active_num"],
    } | losses 
    if model.vq.mode == "softmax": logs["softmax_temp"] = model.vq.softmax_temp
    return out_dict, logs

def train_model(model:AudioVQ, out_dir:str, data_file:str, epochs:int, use_mel_loss:bool):
    sample_rate = 16000
    os.makedirs(out_dir, exist_ok=True)
    device =  "cuda" if torch.cuda.is_available() else "cpu"
    datasets.config.update(
        sample_rate=sample_rate,
        max_val_duration=12, 
        train_duration=2.4,
        hop_length=320,
        device=device
    )
    

    train_datasets, eval_datasets = datasets.create_datasets(
        batch_size=8,
        data_file=os.path.join(out_dir, "../", data_file),
        min_duration=4.0
    )

    
    warmup_lr       = {
        "warmup_step" : 500,
        "down_step" : int(epochs * len(train_datasets) * 0.8),
        "min_lr"  : 1e-5,
        "max_lr" : 1e-3 
    }
    loss_weights = {
        "l1_loss" : 1.0,
        "stft_loss" : 5.0,
        "vq_loss" : 0.5,
        "kl_loss" : 0.0001,
        "commit_loss" :  0.25,
        "codebook_loss" : 1.0
    }
    adamw_betas     = [0.8, 0.9]
   
    
    
    mel_loss = MultiMelSpecLoss(
        sample_rate=sample_rate,
        n_mels=[5, 10, 20, 40, 80, 160],
        window_lengths=[32, 64, 128, 256, 512, 1024],
        clamp_eps=1.0e-5,
    ).to(device) if use_mel_loss else None

    
    eval_datasets += [[train_datasets[i][4]] for i in range(0, len(train_datasets), 1000)]
    eval_datasets = [it for it in eval_datasets if it[0]["duration"] > 1]

    log(f"Start Trainning, epochs: {epochs}, len of data: {len(train_datasets)}, data file: {data_file}, model: {size_of_model(model)}")
    #1248, 20
    #11001, 
    model = model.to(device)
    gen_optimizer = optim.AdamW(params=model.parameters(), betas=adamw_betas, lr=1.0)
    gen_scheduler = WarmupLR(optimizer=gen_optimizer, **warmup_lr)
    logs = TrainingLogs(
        optimizers = [gen_optimizer],
        save_dir=out_dir,
        sample_rate=sample_rate,
        log_steps=100
    )
    def update_batch_data(data:dict, is_train:bool):
        return datasets.update_batch( data,  is_train,  True)

    @torch.no_grad()
    def eval():
        model.eval()
        for di, data in enumerate(eval_datasets):
            data = update_batch_data(data, False)

            out_dict, logs_dict = compute_loss(model, data, loss_weights=loss_weights, mel_loss=mel_loss)
            raw_wavs = out_dict["audios"].squeeze(1).cpu().float().numpy()
            rec_wavs = out_dict["recons"].squeeze(1).detach().cpu().float().numpy()
            cached_demos = {}
            for raw_wav, rec_wav, index in zip(raw_wavs, rec_wavs, data["index"]):
                cached_demos[index] = (raw_wav, rec_wav)

            logs.validation_step(logs_dict, cached_demos)
        logs.validation_end()
        model.train()
    def save():
        model_dir = os.path.join(out_dir, "temp-training", f"{logs.current_epoch}-{logs.current_step}")
        os.makedirs(model_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(model_dir, "model.torch"))

    
    model.train()

    temp_index = 0
    temp_all = int(epochs * 0.9 * len(train_datasets))
    eval()
    for epoch in range(epochs):
        if model.vq.mode == "ema" and epoch > int(epochs * 0.5):
            model.vq.ema_strategy = "full"
            log("change vq ema  strategy to full.")

            

        for di, data in enumerate(train_datasets):
            temp_index = min(temp_all, temp_index + 1)
            model.vq.softmax_temp = 1.0 - (1.0 - 0.1) * temp_index/ temp_all

            data = update_batch_data(data, True)
            out_dict, logs_dict = compute_loss(model, data,loss_weights=loss_weights, mel_loss=mel_loss)
            loss:torch.Tensor = out_dict["loss"]
            gen_optimizer.zero_grad()
            loss.backward()
            gen_optimizer.step()
            gen_scheduler.step()
            logs.training_step(logs_dict, epoch)
            if logs.current_step % 5000 == 0: eval()
            if logs.current_step % 5000 == 0: save()
    eval()
    save()


def train():
    out_dir = "../audio-data/"

    model = build_model(
        base_channels = 128,
        codebook_size = 512,
        codebook_dim = 64,
        num_res_per_stage = 1,
        input_dim = 128, 
        strides = [5, 4, 2, 1],
        #model_file="/code/playground/audio-vq-data/temp-0/temp-training/46-30000/model.torch"
    )
    model.vq.mode = "ema"
    model.vq.ema_strategy = "full"
    train_model(
        model,
        os.path.join(out_dir, f"temp-1"), "train_data_music-4.0.json", 
        50, 
        use_mel_loss=True
    )
train()