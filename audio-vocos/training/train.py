import os

import torch
import torch.distributed as dist

from public.toolkit.nn import size_of_model
from public.toolkit.io import log

from .train_model import TrainModel
from .discriminator import Discriminator
from .train_logs import TrainLogs

from modelling.vocos import Vocos
from modelling.mel_spec import MelSpec
from config import config
from datasets.loader import create_loaders



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
    #dist.barrier()

    device = torch.device(f"cuda:{rank}")
    dtype = config["dtype"]

    
    os.makedirs(config["out_dir"], exist_ok=True)

    logs = TrainLogs(config["out_dir"]) if rank == 0 else None
    global_step = 0

    train_loader, eval_loader = create_loaders(config["data_dir"], rank, world_size, config["batch_size"])

    net_g = Vocos(
        input_channels = config["n_mels"],
        dim = config["model"]["dim"],
        intermediate_dim = config["model"]["intermediate_dim"],
        num_layers = config["model"]["num_layers"],

        n_fft = config["n_fft"], 
        hop_length = config["hop_length"], 
        padding = config["padding"],
    )
    net_d = Discriminator()
    net_melspec = MelSpec(
        sample_rate=config["sampling_rate"],
        n_fft=config["n_fft"], hop_length=config["hop_length"],
        n_mels=config["n_mels"], padding=config["padding"]
    ).to(device)
    log(f'Start: net_d: {size_of_model(net_d)}, net_g: {size_of_model(net_g)}, len data: {len(train_loader)}, config: {config} ')

    net_d = TrainModel(net_d, rank=rank, model_file=config["net_d"])
    net_g = TrainModel(net_g, rank=rank, model_file=config["net_g"])

    loss_weights = config["loss_weights"]

    @torch.no_grad()
    def evaluate():
        net_g.eval()
        audios = []
        for batch_idx, items in enumerate(eval_loader):
            wav, mel_spec = items
            wav         = torch.tensor(wav, device=device, dtype=dtype)
            mel_spec    = torch.tensor(mel_spec, device=device, dtype=dtype)
            wav_hat:torch.Tensor = net_g.model.module(features=mel_spec)
            audios.append({"wav" : wav, "wav_hat" : wav_hat})
        logs.log_evaluate(global_step, audios)
        net_g.train()
    
    def save():
        model_dir = os.path.join(config["out_dir"], "models")
        os.makedirs(model_dir, exist_ok=True)
        net_g.save(os.path.join(model_dir, f"{global_step}_G.pth"))
        net_d.save(os.path.join(model_dir, f"{global_step}_D.pth"))
    #if rank == 0: evaluate()
    
    for epoch in range(1, config["epochs"] + 1):
        net_g.train()
        net_d.train()
        for batch_idx, items in enumerate(train_loader):
            wav, mel_spec = items
            wav         = torch.tensor(wav, device=device, dtype=dtype)#32x57600
            mel_spec    = torch.tensor(mel_spec, device=device, dtype=dtype)#32x100x225

            wav_hat:torch.Tensor = net_g.forward(features=mel_spec)
            
            
            losses = {}

            loss_mp, loss_mr = Discriminator.disc_loss(net_d.model, wav, wav_hat.detach())
            loss_disc = loss_mp + loss_weights["loss_mr"] * loss_mr
            net_d.backward(loss_disc)
            losses.update({"loss_mp" : loss_mp, "loss_mr" : loss_mr, "loss_disc" : loss_disc})


            loss_gen_mp, loss_gen_mr, loss_fm_mp, loss_fm_mr = Discriminator.gen_loss(net_d.model, wav, wav_hat)
            losses.update({
                "loss_gen_mp" : loss_gen_mp,  "loss_gen_mr" : loss_gen_mr,
                "loss_fm_mp" : loss_fm_mp,  "loss_fm_mr" : loss_fm_mr
            })

            mel_loss = torch.nn.functional.l1_loss(
                net_melspec(wav), net_melspec(wav_hat)
            )
            losses["mel_loss"] = mel_loss
            loss = (
                loss_gen_mp
                + loss_weights["loss_gen_mr"] * loss_gen_mr
                + loss_fm_mp
                + loss_weights["loss_fm_mr"] * loss_fm_mr
                + loss_weights["mel_loss"] * mel_loss
            )
            losses["loss"] = loss
            net_g.backward(loss)

            #dist.barrier()
            if rank == 0:
                if global_step % config["log_interval"] == 0:
                    losses = {k : v.item() for k, v in losses.items()} | {
                        "learning_rate" : net_g.optimizer.param_groups[0]['lr'],
                    }

                    logs.log_training(  global_step, epoch, losses)
                if global_step % config["eval_interval"] == 0:
                    evaluate()
                    save()
            #dist.barrier()
            global_step += 1
        net_g.step_scheduler()
        net_d.step_scheduler()
    if rank == 0:
        evaluate()
        save()
