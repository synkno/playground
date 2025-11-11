import torch
import torch.nn as nn
from config import config
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from public.toolkit.io import log

def load_checkpoint(file, model):
    if not file or not os.path.exists(file):
        return False

    checkpoint_dict = torch.load(file, map_location="cpu")
    saved_state_dict = checkpoint_dict["model"]
    model = model.to(list(saved_state_dict.values())[0].dtype)
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            #old_k = str(k).replace("prior_enc.attn_enc", "prior_enc.enc_").replace("poster_enc.wav_net", "poster_enc.enc")
            #old_k = re.sub(r"(flow\.flows\.\d+)\.wav_net", r"\1.enc", old_k)
            old_k = k
            new_state_dict[k] = saved_state_dict[old_k]
            assert saved_state_dict[old_k].shape == v.shape, (
                saved_state_dict[old_k].shape,
                v.shape,
            )
        except Exception:
            if "enc_q" not in k or "emb_g" not in k:
                log(
                    "%s is not in the checkpoint,please check your checkpoint.If you're using pretrain model,just ignore this warning."
                    % k
                )
                new_state_dict[k] = v
    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    return True

def save_checkpoint(model, checkpoint_path):
    if not model: return
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save({"model": state_dict}, checkpoint_path)
class TrainModel:
    def __init__(self, model:nn.Module, rank, model_file:str):
        if load_checkpoint(model_file, model): log(f"load {type(model)} from {model_file}")

        dtype = config["dtype"]
        device = torch.device(f"cuda:{rank}")
        model = model.to(device).to(dtype)
        optimizer = torch.optim.AdamW( model.parameters(), config["learning_rate"], betas=config["betas"], eps=config["eps"])

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config["lr_decay"], last_epoch=-1)

        model = DDP(model, device_ids=[rank], gradient_as_bucket_view=False)

        self.model = model
        self.scheduler = scheduler
        self.optimizer = optimizer
    
    def save(self, file:str):
        save_checkpoint(self.model, file)
    
    def backward(self, loss:torch.Tensor):
        self.optimizer.zero_grad()
        if config["debug"]:
            with torch.autograd.detect_anomaly():
                loss.backward()
        else:
            loss.backward()
        self.optimizer.step()
    
    def forward(self, ** kwargs):
        return self.model(** kwargs)

    def train(self):
        self.model.train()
    def eval(self):
        self.model.eval()
    def step_scheduler(self):
        self.scheduler.step()
    

        