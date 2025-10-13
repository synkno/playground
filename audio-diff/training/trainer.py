import argparse

from vocode.vocoder import Vocoder
from modelling.unit2mel import Unit2Mel
from . import utils
import librosa
import torch
from torch.optim import lr_scheduler
from public.toolkit.io import log
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from .config import config
from .data_loader import create_loaders
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from public.toolkit.nn import size_of_model

import os
import soundfile

def log_sw(writer:SummaryWriter, global_step:int,  epoch:int,  scalars=None, audios = None):
    audio_sampling_rate = config["sampling_rate"]
    if not writer: return
    if scalars:
        for k, v in scalars.items():
            writer.add_scalar(k, v, global_step)
        loss_msgs = "\n ".join([f"{k}: {v:.6f}" for k, v in scalars.items()])
        msg = f"Step {global_step}, Epoch {epoch}, \n {loss_msgs}\n"
        log(msg)  
    if audios:
        log_dir = writer.log_dir
        for k, v in audios.items():
            names = str(k).split("/")
            out_dir = f"{log_dir}/{'/'.join(names[:-1])}/"
            os.makedirs(out_dir, exist_ok=True)
            soundfile.write(f"{out_dir}/{names[-1]}.wav", v.squeeze(0).squeeze(0).detach().cpu().float().numpy(), audio_sampling_rate)

def train(rank, world_size):
    config["debug"] = world_size == 1
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] =  str(world_size)
    os.environ["LOCAL_RANK"] =  str(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl")
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(1234)
    sw = SummaryWriter(log_dir=os.path.join(config["out_dir"], "sw")) if rank == 0 else None

    device = torch.device(f"cuda:{rank}")
    dtype = config["dtype"]
    batch_size = 32
    global_step = 0
    log_steps = 200
    val_steps = 4000

    vocoder = Vocoder(device=device)

    unit2mel = Unit2Mel(
        input_channel=768, 
        n_spk = 1,
        out_dims = vocoder.dimension,
        n_layers = 20,
        n_chans = 512,
        n_hidden = 256,
        timesteps = 1000,
        k_step_max = 0
    )
    if utils.load_checkpoint(config["model_file"], unit2mel): log(f"load net_d from {config['model_file']}")

    unit2mel = unit2mel.to(device=device, dtype=dtype)

    optimizer = torch.optim.AdamW(unit2mel.parameters(), lr=1e-4)
    #initial_global_step, model, optimizer = utils.load_model(args.env.expdir, model, optimizer, device=args.device)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config["lr_step_size"], gamma=0.5,last_epoch=-1)

    train_loader, val_loader = create_loaders(rank=rank, world_size=world_size,  batch_size=batch_size)

    log(f'Start: model: {size_of_model(unit2mel)}, len data: {len(train_loader)}, config: {config}')
    model = torch.nn.parallel.DistributedDataParallel(unit2mel, device_ids=[rank])

    def np2torch(data):
        for k in data.keys(): 
            if not isinstance(data[k], np.ndarray):continue
            data[k] = torch.tensor( data[k], device=device, dtype=dtype)

    @torch.no_grad()
    def evaluate():
        model.eval()
        total_loss = 0
        audios = {}
        for bidx, data in enumerate(val_loader):
            np2torch(data)
            mel = model(
                units   = data['units'], 
                f0      = data['f0'], 
                volume  = data['volume'], 
                spk_emb = data['spk_emb'], 
                gt_spec=None if unit2mel.k_step_max == unit2mel.timesteps else data['mel'],
                infer=True, 
                k_step=unit2mel.k_step_max
            )
            signal = vocoder.infer(mel.to(torch.float32), data['f0'].to(torch.float32))
            audio = vocoder.infer(data["mel"].to(torch.float32), data['f0'].to(torch.float32))
            audios.update({
                f"eval-{global_step}/{bidx}-gen" : signal,
                f"eval-{global_step}/{bidx}-raw" : audio,
            })
            log(f"eval {(bidx + 1)}/{len(val_loader)}")

            total_loss +=  model(
                units   = data['units'], 
                f0      = data['f0'], 
                volume  = data['volume'], 
                spk_emb = data['spk_emb'], 
                gt_spec=data['mel'],
                infer=False,
                k_step=unit2mel.k_step_max
            )
            # log audi
            
        log_sw(sw, global_step, 0, {"eval/loss" : total_loss/len(val_loader)}, audios)
        
        model.train()
    def save():
        model_dir = os.path.join(config["out_dir"], "models")
        os.makedirs(model_dir, exist_ok=True)
        utils.save_checkpoint(model,  os.path.join(model_dir, f"{global_step}_model.pth"))
    #evaluate()
    for epoch in range(config["epochs"]):
        for batch_idx, data in enumerate(train_loader):
            global_step += 1
            optimizer.zero_grad()

            np2torch(data)
            
            loss = model(
                units   = data['units'], 
                f0      = data['f0'], 
                volume  = data['volume'], 
                spk_emb = data['spk_emb'], 
                aug_shift = data['aug_shift'], 
                gt_spec=data['mel'], infer=False,
                k_step=unit2mel.k_step_max
            )
            loss.backward()
            optimizer.step() 
            
            if rank == 0 and global_step % log_steps == 0:
                log_sw(sw, global_step, epoch, {"train/loss" : loss, "train/lr" : optimizer.param_groups[0]["lr"]})
            if rank == 0 and  global_step % val_steps == 0:
                evaluate()
                save()
        scheduler.step()
    if rank == 0:
        evaluate()
        save()
    
