import os
import sys
sys.dont_write_bytecode = True
code_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(code_dir)
sys.path.append(os.path.join(code_dir, "3rd-libs"))


from training.trainer import train, config, trncnf, utils, spec_to_mel_torch, mel_spectrogram_torch
from training.data_preproc import data_prepoc
from training.data_loader import create_loader

from modelling.synth_trn import SynthTrn
import torch

def infer():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
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
    utils.load_checkpoint("/code/playground/audio-data/vits/49200_G.pth", net_g)
    net_g = net_g.to(device=device, dtype=dtype)

    loader = create_loader(
        [
            "/code/data/custom-datasets/kuaou/split-8.0-singer-44k/fd203e1119c3338600afbe836401922e-1.wav",
            "/code/data/custom-datasets/kuaou/split-8.0-singer-44k/fb2542f2e93a7be59fec10a9a3edc3f5-0.wav",
            "/code/data/custom-datasets/kuaou/split-8.0-singer-44k/8be1f8047ad8d615d69153e77db79347-0.wav"
        ],
        num_workers=1
    )
    from public.audio.spk_embed import SpkEmbed
    import soundfile
    from public.audio.datasets import load_audio
    spk_embed = SpkEmbed(device, sample_rate=16000)
    wav = load_audio("/code/data/custom-datasets/audio-examples/zaoanun-16k-mono.wav", sampling_rate=16000)
    
    spk_vec:torch.Tensor = spk_embed.get_embedding(torch.tensor(wav, dtype=torch.float, device=device).unsqueeze(0)) #1x192

    out_dir = "../audio-data/vits/"
    for batch_idx, items in enumerate(loader):
        vec, f0, spec, y, spk, _, uv = items
        vec  = torch.tensor(vec, device=device, dtype=dtype)
        f0   = torch.tensor(f0, device=device, dtype=dtype)
        spec = torch.tensor(spec, device=device, dtype=dtype)
        y    = torch.tensor(y, device=device, dtype=dtype)
        g    = spk_vec.to(dtype)
        uv   = torch.tensor(uv, device=device, dtype=dtype)

        mel = spec_to_mel_torch( spec )
        y_hat,_ = net_g.infer(vec, f0, uv, g=g)

        soundfile.write(f"{out_dir}{batch_idx}_rec.wav", y_hat[0].detach().cpu().float().numpy().reshape(-1), config["sampling_rate"])
        soundfile.write(f"{out_dir}{batch_idx}_raw.wav", y[0].detach().cpu().float().numpy().reshape(-1), config["sampling_rate"])
        
    
if __name__ == "__main__":
    #infer()
    #exit()

    import argparse
    import subprocess
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--debug", type=bool, default=False)
    args = parser.parse_args()
    
    world_size = 2
    if args.local_rank != -1:
        train(args.local_rank, world_size)
    elif args.debug:
        train(0, 1)
    else:
        for i in range(world_size):
            subprocess.Popen(["python", "main.py", "--local_rank", str(i)])

