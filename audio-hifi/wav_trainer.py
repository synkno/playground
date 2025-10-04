

import torch
import torch.nn as nn
import torch.optim as optim
import os
import soundfile
import numpy as np

from wav_gen import WavGen
from public.audio.training_loss import MultiMelSpecLoss
from public.toolkit.training import WarmupLR
from public.toolkit.nn import set_requires_grad, list_grad_params, start_update_params, stop_update_params
from public.toolkit.io import log
from public.audio import datasets

from typing import Dict, Any, List, Tuple
import torch.nn.functional as F
from wav_disc import WaveDisc
from operator import itemgetter
from config import config

from torchaudio.transforms import MelSpectrogram
from log_melspec import LogMelSpec


class WavTrainer(nn.Module):
    def __init__(self, 
        gen:WavGen, 
        disc:WaveDisc,
        mel_spec:LogMelSpec,
        mel_loss:MultiMelSpecLoss, 
        sample_rate:int,
        adamw_betas:List[float],
        warmup_lr:dict,
        loss_lambdas:dict,
        device
    ):
        torch.autograd.set_detect_anomaly(True)
        super().__init__()
        self.gen = gen
        self.mel_loss = mel_loss
        self.mel_spec = mel_spec
        self.loss_lambdas = loss_lambdas
        self.disc = disc

        gen_params = self.gen.parameters()
        self.gen_optimizer = optim.AdamW(params=gen_params, lr=1.0, betas=adamw_betas)
        self.gen_scheduler = WarmupLR(optimizer=self.gen_optimizer, **warmup_lr)

        if self.disc:
            disc_params = self.disc.parameters()
            self.disc_optimizer = optim.AdamW(params=disc_params, lr=1.0, betas=adamw_betas)
            self.disc_scheduler = WarmupLR(optimizer=self.disc_optimizer, **warmup_lr)
        else:
            self.disc_optimizer = None
            self.disc_scheduler = None
        self.sample_rate = sample_rate
        self.to(device)
        self.device = device
        self.data_dir = ""



    def cache_all_data(self, data:list):
        for batch in data:
            for it in batch:
                it["wav"] = datasets.load_audio(it["file"], sampling_rate=self.sample_rate, volume_normalize=True)

    @torch.no_grad()
    def prepare_batch_data(self, batch:Dict[str, Any], is_train:bool, epoch:int):
        def align(data:np.array, h_len:int):
            if data.shape[-1] < h_len:
                last_col = data[..., -1:] 
                repeat_count = h_len - data.shape[-1]
                padding = np.repeat(last_col, repeat_count, axis=-1)
                data = np.concatenate((data, padding), axis=-1)
            return data[..., :h_len]
        
        raw_wavs = []
        for it in batch:
            wav, start, end = itemgetter("wav", "start", "end")(datasets.get_sample(it["file"], is_train, False, audio_wav=it["wav"]))
            raw_wavs.append(wav[start:end])
        
        

        wav:torch.Tensor = torch.tensor(np.array(raw_wavs), device=self.device, dtype=config["dtype"]).unsqueeze(1) #8x1x38400
        mel:torch.Tensor = self.mel_spec(wav.to(torch.float32))
        hop_len = wav.shape[-1] // config["hop_length"]
        mel = mel[..., :hop_len].squeeze(1).to(config["dtype"])

        return {
            "index" :   [b["index"] for b in batch],
            "wav"       : wav,
            "mel"       : mel
        }
    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return self.gen(batch) 
    
    def training_step(self, batch: Dict[str, Any]) -> dict:
        output = self(batch)

        if self.disc:
            disc_optimizer = self.disc_optimizer
            disc_scheduler = self.disc_scheduler

            disc_loss = self.disc.discriminative_loss(output["wav"], output["hat_wav"])
            disc_optimizer.zero_grad()
            disc_loss.backward()
            disc_optimizer.step()
            disc_scheduler.step()

        gen_optimizer = self.gen_optimizer
        gen_scheduler = self.gen_scheduler

        if self.disc: set_requires_grad(self.disc, False)
        loss_dict = self.__compute_generator_loss(output)
        gen_loss = loss_dict["gen_loss"]
        gen_optimizer.zero_grad()
        gen_loss.backward()
        gen_optimizer.step()
        gen_scheduler.step()

        if self.disc:
            set_requires_grad(self.disc, True)
            loss_dict["disc_loss"] = disc_loss

        return loss_dict 
 
    def __compute_generator_loss(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Compute the generator loss."""
        loss_dict = {}

        mel_loss = self.mel_loss(
            inputs["wav"].squeeze(1), inputs["hat_wav"].squeeze(1)
        )
        l1_loss = F.l1_loss(inputs["wav"].squeeze(1), inputs["hat_wav"].squeeze(1))
        loss_dict["mel_loss"] = mel_loss
        loss_dict["l1_loss"] = l1_loss

        if self.disc:
            adv_loss = self.disc.adversarial_loss(inputs["wav"], inputs["hat_wav"])
            loss_dict["adv_loss"] = adv_loss["adv_loss"]
            loss_dict["feature_map_loss"] = adv_loss["feature_map_loss"]


        loss_lambdas = self.loss_lambdas
        loss = sum([v * loss_dict[k] for k, v in loss_lambdas.items()  if k in loss_dict])
        loss_dict["gen_loss"] = loss
        return loss_dict

    @torch.inference_mode()
    def validation_step(self, batch: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        output = self(batch)

        loss_dict = self.__compute_generator_loss(output)

        if self.disc:
            loss_dict["disc_loss"] = self.disc.discriminative_loss(output["wav"], output["hat_wav"])

        raw_wavs = output["wav"].squeeze(1).cpu().float().numpy()
        rec_wavs = output["hat_wav"].squeeze(1).detach().cpu().float().numpy()

        cached_demos = {}
        for raw_wav, rec_wav, index in zip(raw_wavs, rec_wavs, batch['index']):
            cached_demos[index] = (raw_wav, rec_wav)
        return loss_dict, cached_demos
        