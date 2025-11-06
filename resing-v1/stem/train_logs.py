
import os
from torch.utils.tensorboard import SummaryWriter
from public.io import log
from config import config
import soundfile


class TrainLogs:
    def __init__(self, out_dir):
        self.sw = SummaryWriter(log_dir=os.path.join(out_dir, "sw"))

    def log_evaluate(self, global_step, audios):
        out_dir = f"{self.sw.log_dir}/{global_step}/"
        os.makedirs(out_dir, exist_ok=True)
        sr = config["sampling_rate"]

        for it in audios:
            data = it["data"]
            name = f"{data['singer']}-{data['name']}" 
            if "speaker" in data:
                name += f".{data['speaker']['singer']}"
            name = name.replace(" ", "").replace("&", "")

            soundfile.write(f"{out_dir}/{name}-raw.wav", it["wav"].squeeze(0).squeeze(0).detach().cpu().float().numpy(), sr)
            soundfile.write(f"{out_dir}/{name}-gen.wav", it["wav_hat"].squeeze(0).squeeze(0).detach().cpu().float().numpy(), sr)

    def log_training( self,  global_step, epoch, scalars):
        scalars = {"train/" + k: v for k, v in scalars.items()}

        for k, v in scalars.items():
            self.sw.add_scalar(k, v, global_step)
        self.__log_dict(scalars, global_step, epoch)

    def __log_dict(self, loss_dict, global_step, epoch):
        loss_msgs = "\n ".join([f"{k}: {v:.6f}" for k, v in loss_dict.items()])

        msg = f"Step {global_step}, Epoch {epoch}, \n {loss_msgs}\n"
        log(msg)
