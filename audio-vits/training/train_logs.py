
import os
from torch.utils.tensorboard import SummaryWriter
from public.toolkit.io import log
from .config import config
__matplotlib_flag = False
def __init_matplotlib():
    global __matplotlib_flag
    if not __matplotlib_flag:
        import matplotlib
        matplotlib.use("Agg")
        __matplotlib_flag = True

def _plot_spectrogram_to_numpy(spectrogram):
    __init_matplotlib()

    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    w, h = fig.canvas.get_width_height()
    data = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
    data = data[:, :, :3]

    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def _plot_data_to_numpy(x, y):
    __init_matplotlib()

    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 2))
    plt.plot(x)
    plt.plot(y)
    plt.tight_layout()

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    w, h = fig.canvas.get_width_height()
    data = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
    data = data[:, :, :3]
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def _summarize(
    writer:SummaryWriter,
    global_step:int,
    scalars={},
    histograms={},
    images={},
    audios={},
    audio_sampling_rate=22050,
):
    audio_sampling_rate = config["sampling_rate"]
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)

class TrainLogs:
    def __init__(self, out_dir):
        self.sw = SummaryWriter(log_dir=os.path.join(out_dir, "sw"))

    def log_evaluate(self, global_step, audios, images):
        y_hat_mel, mel = images
        image_dict = {
            f"eval/mel-{global_step}-gen": _plot_spectrogram_to_numpy(
                y_hat_mel[0].cpu().float().numpy()
            ),
            f"eval/mel-{global_step}-gt": _plot_spectrogram_to_numpy(
                mel[0].cpu().float().numpy()
            ),
        }
        audio_dict = {}
        for i, (y_hat, y) in enumerate(audios):
            audio_dict.update({
                f"eval/audio-{global_step}_{i}-gen": y_hat.cpu().float(),
                f"eval/audio-{global_step}_{i}-gt": y.cpu().float()
            })
        _summarize(self.sw, global_step, images=image_dict, audios=audio_dict)

    def log_training( self,  global_step, epoch, scalars,  
        y_mel,  y_hat_mel,   mel,
        lf0,  norm_lf0,  pred_lf0,
    ):
        scalars = {"train/" + k: v for k, v in scalars.items()}
        image_dict = {
            f"train/slice-mel-{global_step}-org": _plot_spectrogram_to_numpy(
                y_mel[0].data.cpu().float().numpy()
            ),
            f"train/slice-mel-{global_step}-gen": _plot_spectrogram_to_numpy(
                y_hat_mel[0].data.cpu().float().numpy()
            ),
            f"train/all-mel-{global_step}": _plot_spectrogram_to_numpy(
                mel[0].data.cpu().float().numpy()
            ),

            f"train/all-lf0-{global_step}": _plot_data_to_numpy(
                lf0[0, 0, :].cpu().float().numpy(), pred_lf0[0, 0, :].detach().cpu().float().numpy()
            ),
            f"train/all-lf0-{global_step}-norm": _plot_data_to_numpy(
                lf0[0, 0, :].cpu().float().numpy(), norm_lf0[0, 0, :].detach().cpu().float().numpy()
            ),
        }
        _summarize(
            writer=self.sw,
            global_step=global_step,
            images=image_dict,
            scalars=scalars,
        )
        self.__log_dict(scalars, global_step, epoch)

    def __log_dict(self, loss_dict, global_step, epoch):
        loss_msgs = "\n ".join([f"{k}: {v:.6f}" for k, v in loss_dict.items()])

        msg = f"Step {global_step}, Epoch {epoch}, \n {loss_msgs}n\n"
        log(msg)
