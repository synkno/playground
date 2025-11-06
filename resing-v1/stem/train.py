import logging
import multiprocessing
import os
import time

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from public.obj import Nn
from public.io import log
from dataset.features.mel_spec import MelSpectrogram


from . import utils

from config import config
from dataset.loader import create_loaders

from .train_logs import TrainLogs
from .resing_vits import ResingVits
from .mp_disc import MultiPeriodDisc


def make_resing_vits(model_file:str, device):

    net_g = ResingVits( 
        spec_channels=config["filter_length"] // 2 + 1, 
        inter_channels = 384,
        hidden_channels = 384,
        filter_channels = 768,
        n_heads = 2,
        n_layers = 6,
        kernel_size = 3,
        p_dropout = 0.1,
        resblock = "1",
        resblock_kernel_sizes = [3, 5, 7, 9, 11, 13],
        resblock_dilation_sizes = [
            (1, 3, 5),
            (1, 4, 6),
            (1, 6, 9),
            (1, 2, 4),
            (1, 5, 7),
            (1, 8, 12)
        ],
        upsample_rates = [8, 8, 2, 2, 2],
        upsample_initial_channel = 512,
        upsample_kernel_sizes = [16,16,4,4,4],
        gin_channels = 768,
        ssl_dim = 768,
        sampling_rate=config["sampling_rate"],
        flow_share_parameter=False,
        n_flow_layer=4,
        melspec=MelSpectrogram(sample_rate=config["sampling_rate"], n_fft=config["filter_length"], hop_length=config["hop_length"], device=device)
    )
    if utils.load_checkpoint(model_file, net_g): log(f"load net_g from {model_file}")
    return net_g

def make_train_context(model:nn.Module, device, rank:int):
    model = model.to(device).to(config["dtype"])
    optimiter = torch.optim.AdamW( model.parameters(), config["learning_rate"], betas=config["betas"], eps=config["eps"])
    model = DDP(model, device_ids=[rank], gradient_as_bucket_view=False)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiter, gamma=config["lr_decay"], last_epoch=-1)
    return model, optimiter, scheduler

def kl_loss(z_poster, logs_poster, m_prior, logs_prior, spec_mask):
    #KL(q||p) = log(𝜎_p/𝜎_q) + (𝜎_q^2 + (m_q - m_p) ^ 2)/(2𝜎^2_p) - 0.5
    #q: posterior, p: prior
    #(z_p - m_p) ^ 2  = 𝜎_q^2 + (m_q - m_p)^2 
    #用样本来近视期望 所以上面的等式是成立的

    #KL(q||p) = log(𝜎_p) - log(𝜎_q) - 0.5 + (z_p - m_p) ^ 2/(2𝜎^2_p)
    #1/𝜎^2_p = exp(-2log(𝜎_p))
    z_poster = z_poster.float()
    logs_poster = logs_poster.float()
    m_prior = m_prior.float()
    logs_prior = logs_prior.float()
    spec_mask = spec_mask.float()
    kl = logs_prior - logs_poster - 0.5

    kl += 0.5 * ((z_poster - m_prior) ** 2) * torch.exp(-2.0 * logs_prior)
    kl = torch.sum(kl * spec_mask)
    l = kl / (torch.sum(spec_mask) +  1e-8)
    return l

def train(rank, world_size):
    if config["debug"]:
        torch.autograd.set_detect_anomaly(True)
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

    device = torch.device(f"cuda:{rank}")

    os.makedirs(config["out_dir"], exist_ok=True)

    logs = TrainLogs(config["out_dir"]) if rank == 0 else None
    global_step = 0

    train_loader, eval_loader = create_loaders(
        data_dir = config["data_dir"], 
        rank = rank, world_size = world_size, 
        batch_size = config["batch_size"],
    )
    net_g = make_resing_vits(config["net_g"], device)
    net_g, optim_g, scheduler_g = make_train_context(net_g, device, rank)

    net_d:MultiPeriodDisc = None
    scheduler_d:torch.optim.lr_scheduler.ExponentialLR = None
    optim_d:torch.optim.AdamW

    if False:
        net_d = MultiPeriodDisc()
        if utils.load_checkpoint(config["net_d"], net_d): log(f"load net_g from {config['net_d']}")
        net_d, optim_d, scheduler_d = make_train_context(net_d, device, rank)

    log(f'Start: net_g: {Nn.size_of_model(net_g)}, net_d: {(Nn.size_of_model(net_d) if net_d else "0")} len data: {len(train_loader)} ')


    dtype = config["dtype"]
    def upload_data(item):
        nit = {k: v for k, v in item.items()}
        for key in ["content", "f0", "spec", "wav", "spk_emb", "uv"]:
            nit[key] = torch.tensor(nit[key], device=device, dtype=dtype)
        nit["lengths"] = torch.tensor(nit["lengths"], device=device, dtype=torch.long)
        return nit

    @torch.no_grad()
    def evaluate():
        net_g.eval()
        audios = []
        for batch_idx, item in enumerate(eval_loader):
            item = upload_data(item)
            wav_hat = net_g.module.infer(content = item["content"], f0 = item["f0"], uv=item["uv"], spk_emb=item["spk_emb"])
            audios.append({"data" : item["data"][0], "wav" : item["wav"][0], "wav_hat" : wav_hat[0]} )
        logs.log_evaluate(global_step, audios)
        net_g.train()

    model_dir = os.path.join(config["out_dir"], "models")
    def save():
        os.makedirs(model_dir, exist_ok=True)
        utils.save_checkpoint(net_g,  os.path.join(model_dir, f"{global_step}_G.pth"))
        utils.save_checkpoint(net_d,  os.path.join(model_dir, f"{global_step}_D.pth"))
    if rank == 0: evaluate()
    
    for epoch in range(1, config["epochs"] + 1):
        net_g.train()
        for batch_idx, item in enumerate(train_loader):
            item = upload_data(item)
            (
                spec_mask,
                mel, mel_hat,
                wav, wav_hat,
                z_poster, logs_poster, m_prior, logs_prior
            ) = net_g(
                content = item["content"],  f0 = item["f0"], uv = item["uv"], spec = item["spec"], 
                spk_emb = item["spk_emb"],  lengths = item["lengths"], wav = item["wav"],
                segment_size = config["segment_size"],
                hop_length = config["hop_length"]
            )

            losses = {}

            if net_d:
                losses["loss_disc"] = MultiPeriodDisc.discriminator_loss(net_d, wav, wav_hat)
            
                optim_d.zero_grad()
                losses["loss_disc"].backward()
                optim_d.step()

                loss_gen, loss_fm = MultiPeriodDisc.generator_loss(net_d, wav, wav_hat)
                losses["loss_gen"] = loss_gen
                losses["loss_fm"] = loss_fm

            losses["loss_mel"] = F.l1_loss(mel, mel_hat) * config["loss_weights"]["mel"]
            losses["loss_kl"] = kl_loss(z_poster, logs_poster, m_prior, logs_prior, spec_mask) * config["loss_weights"]["kl"]

            
            losses["loss_gen_all"] = sum(losses[k] for k in ["loss_mel", "loss_kl", "loss_gen", "loss_fm"] if k in losses)

            optim_g.zero_grad()
            if config["debug"]:
                with torch.autograd.detect_anomaly():
                    losses["loss_gen_all"].backward()
            else:
                losses["loss_gen_all"].backward()
            optim_g.step()

            if rank == 0:
                if global_step % config["log_interval"] == 0:
                    log_dict = losses | {
                        "learning_rate" : optim_g.param_groups[0]['lr'],
                    }
                    logs.log_training(
                        global_step, epoch, log_dict, 
                    )
                if global_step % config["eval_interval"] == 0:
                    evaluate()
                    save()
                    

            global_step += 1
        scheduler_g.step()
        if scheduler_d: scheduler_d.step()
    if rank == 0:
        evaluate()
        save()
    dist.destroy_process_group()
    return  os.path.join(model_dir, f"{global_step}_G.pth")
