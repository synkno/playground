import logging
import multiprocessing
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from public.toolkit.nn import size_of_model
from public.toolkit.io import log


from . import utils

from .losses import discriminator_loss, feature_loss, generator_loss, kl_loss

from .config import config
from .mel_spec import mel_spectrogram_torch, spec_to_mel_torch
from .data_loader import create_loaders
from .train_logs import TrainLogs
from .mp_disc import MultiPeriodDisc
from modelling.synth_trn import SynthTrn
trncnf = {
    "segment_size": 10240,
    "lr_decay": 0.999875,
    "betas": [  0.8,  0.99  ],
    "eps": 1e-09,
    "log_interval": 200,
    "eval_interval": 1000,
    "batch_size": 32,
    "loss_weights" : {
        "mel": 45,
        "kl": 1.0,
    }
}

def train(rank, world_size):
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
    dtype = config["dtype"]

    
    os.makedirs(config["out_dir"], exist_ok=True)

    logs = TrainLogs(config["out_dir"]) if rank == 0 else None
    global_step = 0

    train_loader, eval_loader = create_loaders(rank, world_size, trncnf["batch_size"])

    net_g = SynthTrn( 
        spec_channels=config["filter_length"] // 2 + 1, 
        segment_size=trncnf["segment_size"] // config["hop_length"], 
        inter_channels = 192,
        hidden_channels = 192,
        filter_channels = 768,
        n_heads = 2,
        n_layers = 6,
        kernel_size = 3,
        p_dropout = 0.1,
        resblock = "1",
        resblock_kernel_sizes = [3, 7, 11],
        resblock_dilation_sizes = [[1,3,5],[1,3,5],[1,3,5]],
        upsample_rates = [8, 8, 2, 2, 2],
        upsample_initial_channel = 512,
        upsample_kernel_sizes = [16,16,4,4,4],
        gin_channels = 768,
        ssl_dim = 768,
        n_speakers = config["n_speakers"],
        sampling_rate=config["sampling_rate"],
        flow_share_parameter=False,
        n_flow_layer=4,
    )
    net_d = MultiPeriodDisc(False)

    if utils.load_checkpoint(config["net_d"], net_d): log(f"load net_d from {config['net_d']}")
    if utils.load_checkpoint(config["net_g"], net_g): log(f"load net_g from {config['net_g']}")

    net_d = net_d.to(device).to(dtype)
    net_g = net_g.to(device).to(dtype)

    optim_g = torch.optim.AdamW( net_g.parameters(), config["learning_rate"], betas=trncnf["betas"], eps=trncnf["eps"])
    optim_d = torch.optim.AdamW( net_d.parameters(), config["learning_rate"], betas=trncnf["betas"], eps=trncnf["eps"])

    log(f'Start: net_d: {size_of_model(net_d)}, net_g: {size_of_model(net_g)}, len data: {len(train_loader)} ')
    net_g = DDP(net_g, device_ids=[rank], gradient_as_bucket_view=False)  # , find_unused_parameters=True)
    net_d = DDP(net_d, device_ids=[rank], gradient_as_bucket_view=False)

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=trncnf["lr_decay"], last_epoch=-1)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=trncnf["lr_decay"], last_epoch=-1)

    @torch.no_grad()
    def evaluate():
        net_g.eval()
        audios = []
        for batch_idx, items in enumerate(eval_loader):
            vec, f0, spec, y, spk, _, uv = items
            vec  = torch.tensor(vec, device=device, dtype=dtype)
            f0   = torch.tensor(f0, device=device, dtype=dtype)
            spec = torch.tensor(spec, device=device, dtype=dtype)
            y    = torch.tensor(y, device=device, dtype=dtype)
            g    = torch.tensor(spk, device=device, dtype=dtype)
            uv   = torch.tensor(uv, device=device, dtype=dtype)

            mel = spec_to_mel_torch( spec )
            y_hat,_ = net_g.module.infer(vec, f0, uv, g=g)
            y_hat_mel = mel_spectrogram_torch(y_hat.squeeze(1).float())
            audios.append((y_hat[0], y[0]))
        logs.log_evaluate(global_step, audios, (y_hat_mel, mel))
        net_g.train()
    
    def save():
        model_dir = os.path.join(config["out_dir"], "models")
        os.makedirs(model_dir, exist_ok=True)
        utils.save_checkpoint(net_g,  os.path.join(model_dir, f"{global_step}_G.pth"))
        utils.save_checkpoint(net_d,  os.path.join(model_dir, f"{global_step}_D.pth"))
    #if rank == 0: evaluate()
    
    for epoch in range(1, config["epochs"] + 1):
        net_g.train()
        net_d.train()
        for batch_idx, items in enumerate(train_loader):
            vec, f0, spec, y, spk, lengths, uv = items
            vec  = torch.tensor(vec, device=device, dtype=dtype)
            f0   = torch.tensor(f0, device=device, dtype=dtype)
            spec = torch.tensor(spec, device=device, dtype=dtype)
            y    = torch.tensor(y, device=device, dtype=dtype)
            g    = torch.tensor(spk, device=device, dtype=dtype)
            uv   = torch.tensor(uv, device=device, dtype=dtype)
            lengths   = torch.tensor(lengths, device=device, dtype=torch.long)

            mel = spec_to_mel_torch(spec)
            
            y_hat, ids_slice, z_mask, \
            (z, z_p, m_p, logs_p, m_q, logs_q), pred_lf0, norm_lf0, lf0 = net_g(vec, f0, uv, spec, g=g, c_lengths=lengths,
                                                                                spec_lengths=lengths)

            y_mel = utils.slice_segments(mel, ids_slice, trncnf["segment_size"] // config["hop_length"])
            y_hat_mel = mel_spectrogram_torch(y_hat.squeeze(1) )
            y = utils.slice_segments(y, ids_slice * config["hop_length"], trncnf["segment_size"])  # slice

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            
            losses = {}

            loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
            losses["loss_disc"] = loss_disc
                    
            
            optim_d.zero_grad()
            losses["loss_disc"].backward()
            grad_norm_d = utils.clip_grad_value(net_d.parameters(), None)
            optim_d.step()
            
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            
            losses["loss_mel"] = F.l1_loss(y_mel, y_hat_mel) * trncnf["loss_weights"]["mel"]
            losses["loss_kl"] = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * trncnf["loss_weights"]["kl"]
            losses["loss_fm"] = feature_loss(fmap_r, fmap_g)
            loss_gen, losses_gen = generator_loss(y_d_hat_g)
            losses["loss_gen"] = loss_gen
            losses["loss_lf0"] = F.mse_loss(pred_lf0, lf0)
            losses["oss_gen_all"] = losses["loss_gen"] + losses["loss_fm"] + losses["loss_mel"] + losses["loss_kl"] + losses["loss_lf0"]

            optim_g.zero_grad()
            losses["oss_gen_all"].backward()
            grad_norm_g = utils.clip_grad_value(net_g.parameters(), None)
            optim_g.step()

            if rank == 0:
                if global_step % trncnf["log_interval"] == 0:
                    log_dict = losses | {
                        "learning_rate" : optim_g.param_groups[0]['lr'],
                        "grad_norm_d" : grad_norm_d,
                        "grad_norm_g" : grad_norm_g,
                    }
                    logs.log_training(
                        global_step, epoch, log_dict, 
                        y_mel=y_mel, y_hat_mel=y_hat_mel, mel=mel, 
                        lf0=lf0, norm_lf0=norm_lf0, pred_lf0=pred_lf0
                    )
                if global_step % trncnf["eval_interval"] == 0:
                    evaluate()
                    save()
                    

            global_step += 1
        scheduler_g.step()
        scheduler_d.step()
    if rank == 0:
        evaluate()
        save()
