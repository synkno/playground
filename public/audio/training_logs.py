import torch
import torch.nn as nn
import torch.optim as optim
from public.toolkit.io import log
from torch.utils.tensorboard import SummaryWriter
import os
from typing import List, Dict, Any, Union
import soundfile

class TrainingLogs:
    def __init__(self, save_dir:str, log_steps:int,sample_rate:int, optimizers: List[optim.Optimizer] = None):
        self.save_dir = save_dir
        self.log_steps = log_steps
        self.sample_rate = sample_rate
        sw_log_dir = os.path.join(self.save_dir, "temp-sw")
        os.makedirs(sw_log_dir, exist_ok=True)
        self.sw = SummaryWriter(sw_log_dir)

        self.validation_step_outputs = []
        self.optimizers = optimizers

        self.current_step = 0
        self.current_epoch = 0
    
    def training_step(self, loss_dict:dict, epoch:int):
        self.current_epoch = epoch
        self.current_step += 1
        if self.current_step % self.log_steps == 0:
            self.__log_dict(loss_dict, "Train")
    
    def validation_step(self, loss_dict:dict, cached_demos:dict):
        step = self.current_step
        save_dir = os.path.join(self.save_dir, f"temp-val/{self.current_epoch}-{step}")
        os.makedirs(save_dir, exist_ok=True)

        for index, (raw_wav, rec_wav) in cached_demos.items():
            soundfile.write(f"{save_dir}/{index}_rec.wav", rec_wav, self.sample_rate)
            soundfile.write(f"{save_dir}/{index}_raw.wav", raw_wav, self.sample_rate)
        self.validation_step_outputs.append(loss_dict)
    
    def validation_end(self):
        step_outputs = self.validation_step_outputs
        aggregated_loss = {}
        for loss_dict in step_outputs:
            for key, value in loss_dict.items():
                if not torch.is_tensor(value):
                    continue
                
                if key not in aggregated_loss:
                    aggregated_loss[key] = []
                aggregated_loss[key].append(value)
        for key, values in aggregated_loss.items():
            if isinstance(values[0], torch.Tensor):
                aggregated_loss[key] = torch.stack(values).mean()
        
        self.__log_dict(aggregated_loss, "Eval")

        self.validation_step_outputs = []

    

    def __log_dict(self, loss_dict: Dict[str, Any], tag: str):
        lr_dict = self.__get_lr()
        lr_msgs = ", ".join(f"lr_{k}: {v:.2e}" for k, v in lr_dict.items())
        loss_msgs = "\n ".join([f"{k}: {v:.6f}" for k, v in loss_dict.items()])

        msg = f"{tag} | Step {self.current_step}, Epoch {self.current_epoch}, \n {loss_msgs}\n {lr_msgs}\n\n"
        log(msg)

        for name, value in loss_dict.items():
            self.sw.add_scalar( f"{tag}-{name}", value, self.current_step)
    
    def __get_lr(self) -> Union[Dict[str, float], float]:
        optimizers = self.optimizers
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
            
        return {
            str(i): optimizer.param_groups[0]["lr"]
            for i, optimizer in enumerate(optimizers)
            if optimizer is not None
        }